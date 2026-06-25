#!/usr/bin/env python3
"""Extract frozen last-token hidden states from a pretrained GPT-2 model (elicitation readout input).

The activation we read is exactly the one the model's own classifier head consumes
(`weak_to_strong/model.py`): the final-layer hidden state at the **last non-pad token**
(`input_lens-1`, pad id = 0). We additionally keep a small set of intermediate layers via
`output_hidden_states` for a layer sweep.

`select_last_token_states` is a pure (numpy) function so it can be unit-tested without torch or a
GPU (see tests/test_extract_activations.py). The model forward (`extract_states`, `main`) needs
torch + transformers and runs on the GPU box.

Output (per model/ds/seed/split): an .npz with `acts_L<layer>` arrays [N, H], `hard_label` [N],
and `txt` (for joining). Usage on the box:
  python extract_activations.py --model_size=gpt2-xl --ds=boolq --seed=0 --split=test \
      --layers=last,half --out=results/elicitation/acts
"""
import numpy as np


def select_last_token_states(hidden, input_lens):
    """hidden: [B, S, H]; input_lens: [B] (count of non-pad tokens) -> [B, H].

    Returns the hidden state at the last non-pad position (input_lens-1), clamped to >=0.
    Pure numpy; mirrors TransformerWithHead.forward's `transformer_outputs[0][i, input_lens[i]-1]`.
    """
    hidden = np.asarray(hidden)
    input_lens = np.asarray(input_lens)
    assert hidden.ndim == 3, f"expected [B,S,H], got {hidden.shape}"
    assert input_lens.shape[0] == hidden.shape[0], "batch mismatch"
    idx = np.clip(input_lens - 1, 0, hidden.shape[1] - 1)
    return hidden[np.arange(hidden.shape[0]), idx]


def resolve_layers(spec, n_layers):
    """Map a layer spec like 'last,half,threequarter' to hidden_states indices (0=embeddings,
    n_layers=final). Accepts named anchors or explicit ints."""
    named = {"last": n_layers, "threequarter": (3 * n_layers) // 4,
             "half": n_layers // 2, "quarter": n_layers // 4, "first": 1}
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.append(named[tok] if tok in named else int(tok))
    return sorted(set(out))


# ----- everything below needs torch + transformers (GPU box) -----

def load_frozen_model(model_name, device="cuda", dtype="float32"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dt = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[dtype]
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dt)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def extract_states(model, tokenizer, texts, layers, max_ctx=1024, device="cuda", batch_size=16):
    """Forward `texts` through the frozen model; return {layer_index: np.ndarray [N, H]} of
    last-non-pad-token states. Right-pads with id 0 (causal model → position len-1 is unaffected)."""
    import torch
    out = {L: [] for L in layers}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = [tokenizer(t)["input_ids"][:max_ctx] for t in batch]
        enc = [e if len(e) > 0 else [tokenizer.eos_token_id or 0] for e in enc]
        maxlen = max(len(e) for e in enc)
        input_ids = torch.zeros(len(enc), maxlen, dtype=torch.long)  # pad id = 0
        for j, e in enumerate(enc):
            input_ids[j, : len(e)] = torch.tensor(e, dtype=torch.long)
        input_ids = input_ids.to(device)
        input_lens = (input_ids != 0).sum(-1).cpu().numpy()
        with torch.no_grad():
            res = model(input_ids, output_hidden_states=True)
        hs = res.hidden_states  # tuple len n_layers+1
        for L in layers:
            layer_np = hs[L].float().cpu().numpy()  # [B,S,H]
            out[L].append(select_last_token_states(layer_np, input_lens))
    return {L: np.concatenate(out[L], 0) for L in layers}


def main():
    import argparse, json, os
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from weak_to_strong.datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--n_test_docs", type=int, default=10000)
    ap.add_argument("--layers", default="last,half")
    ap.add_argument("--max_ctx", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "results/elicitation/acts"))
    # CCS contrast suffixes (empty => plain readout for the k-shot probe)
    ap.add_argument("--contrast", default="")  # e.g. "boolq" or "sciq" to also emit pos/neg states
    a = ap.parse_args()

    sizes = dict(train=a.n_docs, test=a.n_test_docs)
    ds = load_dataset(a.ds, seed=a.seed, split_sizes=sizes)[a.split]
    texts = list(ds["txt"]); labels = np.array(ds["hard_label"], dtype=np.int64)
    model, tok = load_frozen_model(a.model_size, device=a.device)
    n_layers = model.config.num_hidden_layers
    layers = resolve_layers(a.layers, n_layers)

    payload = {"hard_label": labels, "txt": np.array(texts, dtype=object), "layers": np.array(layers)}
    plain = extract_states(model, tok, texts, layers, a.max_ctx, a.device, a.batch_size)
    for L, arr in plain.items():
        payload[f"acts_L{L}"] = arr.astype(np.float32)

    if a.contrast:
        pos_suf, neg_suf = CONTRAST_SUFFIXES[a.contrast]
        pos = extract_states(model, tok, [t + pos_suf for t in texts], layers, a.max_ctx, a.device, a.batch_size)
        neg = extract_states(model, tok, [t + neg_suf for t in texts], layers, a.max_ctx, a.device, a.batch_size)
        for L in layers:
            payload[f"pos_L{L}"] = pos[L].astype(np.float32)
            payload[f"neg_L{L}"] = neg[L].astype(np.float32)

    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.ds}_{a.model_size}_s{a.seed}_{a.split}.npz")
    np.savez_compressed(fn, **payload)
    print(f"wrote {fn}  N={len(texts)} layers={layers} contrast={bool(a.contrast)}")


# contrast templates (pos = asserts the true class, hard_label==1)
CONTRAST_SUFFIXES = {
    "boolq": ("\nAnswer: Yes", "\nAnswer: No"),
    "sciq": ("\nThe answer is correct.", "\nThe answer is incorrect."),
}


if __name__ == "__main__":
    main()
