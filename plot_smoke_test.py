"""
Plotting script for weak-to-strong results.
Loads all seeds and plots mean accuracy with min-max range shading.
Uses the same visual style as notebooks/Plotting.ipynb.
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

SEED_DIRS = [
    os.path.join(os.path.dirname(__file__), "results", "data", "baseline", d)
    for d in ["seed0", "seed1", "seed2"]
]
MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load results from all seeds ---
records = []
for seed_dir in SEED_DIRS:
    if not os.path.isdir(seed_dir):
        print(f"Skipping missing dir: {seed_dir}")
        continue
    for result_filename in glob.glob(os.path.join(seed_dir, "**/results_summary.json"), recursive=True):
        config_file = os.path.join(os.path.dirname(result_filename), "config.json")
        if not os.path.exists(config_file):
            continue
        config = json.load(open(config_file, "r"))
        if config["model_size"] not in MODELS_TO_PLOT:
            continue
        if 'seed' not in config:
            config['seed'] = 0
        record = config.copy()
        if 'weak_model' in config:
            for k in record['weak_model']:
                if k == 'model_size':
                    assert record['weak_model'][k] == record['weak_model_size']
                record['weak_' + k] = record['weak_model'][k]
            del record['weak_model']
        record.update(json.load(open(result_filename)))
        records.append(record)

df = pd.DataFrame.from_records(records).sort_values(['ds_name', 'model_size'])
print(f"Loaded {len(records)} result records across {df['seed'].nunique()} seeds")

# --- Plot per dataset ---
datasets = df.ds_name.unique()
for dataset in datasets:
    cur_df = df[(df.ds_name == dataset)].copy()
    # Drop logconf ground truth — ground truth baseline should always be xent
    cur_df = cur_df[~((cur_df['weak_model_size'].isna()) & (cur_df['loss'] == 'logconf'))]

    # Compute mean GT accuracy across seeds for x-axis positioning
    base_accuracies = (
        cur_df[cur_df['weak_model_size'].isna()]
        .groupby('model_size')
        .agg({'accuracy': 'mean'})
        .sort_values('accuracy')
    )
    base_accuracy_lookup = base_accuracies['accuracy'].to_dict()
    base_accuracies = base_accuracies.reset_index()

    # Assign x-position (mean GT accuracy of the strong model)
    cur_df['strong_model_accuracy'] = cur_df['model_size'].apply(lambda x: base_accuracy_lookup[x])
    cur_df.loc[~cur_df['weak_model_size'].isna(), 'weak_model_accuracy'] = (
        cur_df.loc[~cur_df['weak_model_size'].isna(), 'weak_model_size']
        .apply(lambda x: base_accuracy_lookup[x])
    )

    # Compute PGR
    valid_pgr_index = (
        (~cur_df['weak_model_size'].isna()) &
        (cur_df['weak_model_size'] != cur_df['model_size']) &
        (cur_df['strong_model_accuracy'] > cur_df['weak_model_accuracy'])
    )
    cur_df.loc[valid_pgr_index, 'pgr'] = (
        (cur_df.loc[valid_pgr_index, 'accuracy'] - cur_df.loc[valid_pgr_index, 'weak_model_accuracy'])
        / (cur_df.loc[valid_pgr_index, 'strong_model_accuracy'] - cur_df.loc[valid_pgr_index, 'weak_model_accuracy'])
    )

    cur_df.loc[cur_df['weak_model_size'].isna(), "weak_model_size"] = "ground truth"

    plot_df = cur_df.sort_values(['strong_model_accuracy']).sort_values(['loss'], ascending=False)

    print(f"\nDataset: {dataset} ({plot_df['seed'].nunique()} seeds)")

    # Median PGR table (across all seeds)
    pgr_results = plot_df[~plot_df['pgr'].isna()].groupby(['loss']).aggregate({"pgr": "median"})

    # Color palette matching original
    weak_models = plot_df['weak_model_size'].unique()
    palette = sns.color_palette('colorblind', n_colors=max(len(weak_models) - 1, 1))
    color_dict = {model: ("black" if model == 'ground truth' else palette.pop()) for model in weak_models}

    # Plot: mean line with min-max range shading
    fig, ax = plt.subplots(figsize=(10, 6))

    # sns.lineplot with errorbar=('pi', 100) gives min-max range as shading
    sns.lineplot(
        data=plot_df,
        x='strong_model_accuracy', y='accuracy',
        hue='weak_model_size', style='loss',
        markers=True, palette=color_dict,
        errorbar=('pi', 100),
        ax=ax,
    )

    # PGR inset table
    if not pgr_results.empty:
        pd.plotting.table(
            ax, pgr_results.round(4),
            loc='lower right', colWidths=[0.1, 0.1],
            cellLoc='center', rowLoc='center',
        )

    ax.set_xticks(base_accuracies['accuracy'])
    ax.set_xticklabels(
        [f"{e} ({base_accuracy_lookup[e]:.3f})" for e in base_accuracies['model_size']],
        rotation=90,
    )
    ax.set_title(f"Dataset: {dataset} (mean of {plot_df['seed'].nunique()} seeds, shaded=range)")
    ax.set_xlabel("Strong model accuracy (ground truth)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='upper left')

    out_path = os.path.join(OUTPUT_DIR, f"{dataset.replace('/', '-')}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
