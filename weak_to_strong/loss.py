import torch


class LossFnBase:
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        This function calculates the loss between logits and labels.
        """
        raise NotImplementedError


# Custom loss function
class xent_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float, **kwargs
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.
        **kwargs: ignored (accepts the per-row aux tensors gt_mask/sample_weight the
            training loop now always passes, so the naive path is unaffected — Phase 2 A1).

        Returns:
        The mean of the cross entropy loss.
        """
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss.mean()


class WeightedXentLoss(LossFnBase):
    """Per-row weighted cross-entropy (Phase 2 plumbing's reference consumer; used by M1
    weighted-loss and M4 reliability-weighting).

    Row weight = base 1.0, scaled by `gt_weight` on GT rows (via `gt_mask`), then multiplied
    by an optional per-row `sample_weight`. Loss = Σ wᵢ·CEᵢ / Σ wᵢ.

    INVARIANT 5: with `gt_weight == 1.0` and `sample_weight is None` (or all ones), this reduces
    EXACTLY to `xent_loss` (`cross_entropy(logits, labels).mean()`).
    """

    def __init__(self, gt_weight: float = 1.0):
        self.gt_weight = gt_weight

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        gt_mask: torch.Tensor = None,
        sample_weight: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        w = torch.ones_like(ce)
        if gt_mask is not None and self.gt_weight != 1.0:
            w = w + (self.gt_weight - 1.0) * gt_mask.to(w.dtype)
        if sample_weight is not None:
            w = w * sample_weight.to(w.dtype)
        return (w * ce).sum() / w.sum()


class product_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for product of predictions and labels.

    Attributes:
    alpha: A float indicating how much to weigh the weak model.
    beta: A float indicating how much to weigh the strong model.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        alpha: float = 1.0,  # how much to weigh the weak model
        beta: float = 1.0,  # how much to weigh the strong model
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.alpha = alpha
        self.beta = beta
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        **kwargs,
    ) -> torch.Tensor:
        preds = torch.softmax(logits, dim=-1)
        target = torch.pow(preds, self.beta) * torch.pow(labels, self.alpha)
        target /= target.sum(dim=-1, keepdim=True)
        target = target.detach()
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        **kwargs,
    ) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.softmax(logits, dim=-1)
        mean_weak = torch.mean(labels, dim=0)
        assert mean_weak.shape == (2,)
        threshold = torch.quantile(preds[:, 0], mean_weak[1])
        strong_preds = torch.cat(
            [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
            dim=1,
        )
        target = labels * (1 - coef) + strong_preds.detach() * coef
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()
