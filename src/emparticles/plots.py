"""Figures for the report, made from results/results.json and results/examples.npz.

    python -m emparticles.plots results figures

training_curve.png   loss and validation AUC per epoch (model and positive control)
roc.png              ROC curves on the held-out test set: model, positive and negative controls
by_source.png        test AUC and IoU per source for the model and the controls
examples.png         test tiles: picture, true mask, predicted probability
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCES = ["emps", "hrtem", "co3o4"]
STYLE = {"model": ("trained U-Net", "#1f77b4"), "planted": ("positive control (mask given)", "#2ca02c"),
         "dummy": ("negative control (constant score)", "#7f7f7f"), "permuted": ("negative control (shuffled masks)", "#d62728")}


def training_curve(r, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for h, name in [(r["history"], "model"), (r["history_planted"], "planted")]:
        e = [x["epoch"] for x in h]
        label, c = STYLE[name]
        ax[0].plot(e, [x["train_loss"] for x in h], "-o", color=c, label=label)
        ax[1].plot(e, [x["val_loss"] for x in h], "-o", color=c, label=label)
        ax[2].plot(e, [x["val_auc"] for x in h], "-o", color=c, label=label)
    for a, t in zip(ax, ["training loss", "validation loss", "validation AUC"]):
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[2].set_ylim(0.4, 1.01)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=150); plt.close(fig)


def roc(r, out):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for k in ["model", "planted", "dummy", "permuted"]:
        t = r["test"][k]
        label, c = STYLE[k]
        ax.plot(t["roc"]["fpr"], t["roc"]["tpr"], color=c, label=f"{label}  AUC {t['overall']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="k", lw=.7)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC, held-out test pixels")
    ax.legend(fontsize=6.5, loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=150); plt.close(fig)


def by_source(r, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    srcs = [s for s in SOURCES if s in r["test"]["model"]["by_source"]]
    w = 0.2
    for j, (metric, title) in enumerate([("auc", "AUC"), ("iou", "IoU at probability 0.5")]):
        for i, k in enumerate(["model", "planted", "dummy", "permuted"]):
            v = [r["test"][k]["by_source"][s][metric] for s in srcs]
            ax[j].bar(np.arange(len(srcs)) + (i - 1.5) * w, v, w, color=STYLE[k][1], label=STYLE[k][0])
        ax[j].set_xticks(range(len(srcs))); ax[j].set_xticklabels(srcs); ax[j].set_title(title + " per source (test)")
        ax[j].grid(alpha=.3, axis="y")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "by_source.png", dpi=150); plt.close(fig)


def examples(res_dir, out):
    z = np.load(res_dir / "examples.npz")
    n = len(z["x"])
    fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.5 * n), squeeze=False)
    for i in range(n):
        for j, (img, t, cm) in enumerate([(z["x"][i], "picture", "gray"), (z["y"][i], "true mask", "gray"),
                                          (z["prob"][i], "predicted probability", "magma")]):
            ax[i, j].imshow(img, cmap=cm, vmin=0, vmax=1); ax[i, j].axis("off")
            ax[i, j].set_title(f"{SOURCES[int(z['source'][i])]}: {t}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "examples.png", dpi=130); plt.close(fig)


def main(res_dir: str, fig_dir: str):
    res, out = Path(res_dir), Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((res / "results.json").read_text())
    training_curve(r, out); roc(r, out); by_source(r, out); examples(res, out)
    print(f"plots: wrote training_curve.png roc.png by_source.png examples.png to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
