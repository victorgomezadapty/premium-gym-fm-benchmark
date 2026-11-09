"""Figures from results/multi_anchor_summary.json. Bars show 95% bootstrap intervals."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figures"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "multi_anchor_summary.json")))
ORDER = [("R_multi", "Kumo Relational, raw tables", "#9a9a92"), ("R_feat", "Kumo Relational + features", "#2f8f83"),
         ("T_10k", "Kumo Tabular", "#0f766e"), ("LGB_same", "LightGBM, same rows", "#8fa8d8"), ("LGB_full", "LightGBM, all rows", "#1d4ed8")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def panel(ax, task, metric, xlabel):
    models = R[task]["models"]
    for y, (key, label, color) in enumerate(ORDER):
        m = models[key][metric]
        ax.errorbar(m["value"], y, xerr=[[m["value"] - m["ci95"][0]], [m["ci95"][1] - m["value"]]], fmt="o", color=color, ms=8, capsize=4, lw=2)
    ax.set_yticks(range(len(ORDER)), [o[1] for o in ORDER]); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.25)

fig, ax = plt.subplots(figsize=(9, 4.4))
panel(ax, "churn_30d", "roc_auc", "Churn in the next 30 days, ROC AUC (higher is better)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "churn-roc.png"), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
panel(axs[0], "revenue_30d", "mae", "Revenue MAE, next 30 days (lower is better)")
panel(axs[1], "visits_30d", "mae", "Visits MAE, next 30 days (lower is better)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "regression-mae.png"), dpi=160); plt.close(fig)
