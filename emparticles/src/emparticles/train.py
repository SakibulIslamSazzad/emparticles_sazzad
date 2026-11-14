"""Train the segmentation model on data/inputs/train.safetensors and test it on eval.safetensors.

    python -m emparticles.train config.yaml data/inputs results

What it does (all seeded, every tile order is fixed):

1. Loads train.safetensors and eval.safetensors (made by `emparticles.tensors`).
2. Cross-validation split: tiles whose `fold` equals `train.val_fold` are VALIDATION; the other
   folds are TRAINING. Whole groups are in one fold, so validation never shares a group with training.
3. Trains the U-Net (random crops, flips) and prints the loss and validation AUC after every epoch.
   The epoch with the best validation AUC is kept.
4. Scores the held-out eval.safetensors (the TEST set, never used for any choice) and four controls:
     model      the trained U-Net
     dummy      NEGATIVE control: the same score for every pixel (the training foreground rate)
     permuted   NEGATIVE control: the trained model scored against test masks shuffled between tiles
     planted    POSITIVE control: a second U-Net that is GIVEN the true mask as an extra input
                channel; if training and scoring work it must reach AUC close to 1
5. Writes results/results.json (every number in the report), results/model.safetensors and
   results/examples.npz (a few test tiles with their predictions, for the figures).

Metrics are pixel-wise: ROC AUC, average precision (AP), and IoU / Dice / precision / recall at
probability 0.5. They are computed from histograms of the logits, so the 57 M test pixels never
have to be held in memory.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from .model import UNet, n_params

NBINS = 4096
LOGIT_RANGE = 12.0            # logits are clipped to [-12, 12] before binning
SOURCES = ["emps", "hrtem", "co3o4"]


# ---------------------------------------------------------------- metrics from histograms
class Hist:
    """Counts of positive and negative pixels per logit bin."""

    def __init__(self):
        self.pos = np.zeros(NBINS, np.float64)
        self.neg = np.zeros(NBINS, np.float64)

    def add(self, logit: torch.Tensor, y: torch.Tensor):
        idx = ((logit.clamp(-LOGIT_RANGE, LOGIT_RANGE) + LOGIT_RANGE) / (2 * LOGIT_RANGE) * (NBINS - 1)).round().long()
        yb = y.bool()
        self.pos += torch.bincount(idx[yb], minlength=NBINS).double().numpy()
        self.neg += torch.bincount(idx[~yb], minlength=NBINS).double().numpy()

    def roc(self):
        """(fpr, tpr) from the highest threshold down; ties inside a bin count as a diagonal."""
        tp = np.concatenate([[0.0], np.cumsum(self.pos[::-1])])
        fp = np.concatenate([[0.0], np.cumsum(self.neg[::-1])])
        return fp / max(fp[-1], 1), tp / max(tp[-1], 1)

    def metrics(self) -> dict:
        fpr, tpr = self.roc()
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        tp = np.cumsum(self.pos[::-1])
        fp = np.cumsum(self.neg[::-1])
        recall = tp / max(tp[-1], 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
        half = NBINS // 2                                     # logit >= 0  <=>  probability >= 0.5
        TP, FN = self.pos[half:].sum(), self.pos[:half].sum()
        FP, TN = self.neg[half:].sum(), self.neg[:half].sum()
        return dict(auc=auc, ap=ap,
                    iou=float(TP / max(TP + FP + FN, 1)), dice=float(2 * TP / max(2 * TP + FP + FN, 1)),
                    precision=float(TP / max(TP + FP, 1)), recall=float(TP / max(TP + FN, 1)),
                    accuracy=float((TP + TN) / max(TP + TN + FP + FN, 1)),
                    n_pixels=int(TP + TN + FP + FN), foreground_rate=float((TP + FN) / max(TP + TN + FP + FN, 1)))


class Scores:
    """Histograms for all pixels and for each source."""

    def __init__(self):
        self.all = Hist()
        self.by_source = {s: Hist() for s in SOURCES}

    def add(self, logit, y, source):
        self.all.add(logit, y)
        for k, s in enumerate(SOURCES):
            m = source == k
            if m.any():
                self.by_source[s].add(logit[m], y[m])

    def summary(self, curve_points: int = 200) -> dict:
        out = dict(overall=self.all.metrics(), by_source={s: h.metrics() for s, h in self.by_source.items()
                                                          if h.pos.sum() + h.neg.sum() > 0})
        fpr, tpr = self.all.roc()
        keep = np.unique(np.linspace(0, len(fpr) - 1, curve_points).round().astype(int))
        out["roc"] = dict(fpr=[round(float(v), 5) for v in fpr[keep]], tpr=[round(float(v), 5) for v in tpr[keep]])
        return out


# ---------------------------------------------------------------- helpers
def planted(x, y, on):
    """Model input: the picture, plus (positive control only) the true mask as a second channel."""
    return torch.cat([x, y.float()], dim=1) if on else x


def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


@torch.no_grad()
def score(model, x, y, source, batch, plant=False, permute=None, keep=None):
    """Run the model over tiles and return Scores (and optionally the probabilities of tiles in `keep`)."""
    model.eval()
    sc, sc_perm, kept = Scores(), Scores(), {}
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch].float()
        logit = model(planted(xb, yb, plant))
        sc.add(logit, yb, source[i:i + batch])
        if permute is not None:                       # same predictions, masks of OTHER tiles
            sc_perm.add(logit, y[permute[i:i + batch]].float(), source[i:i + batch])
        if keep:
            for j in range(len(xb)):
                if i + j in keep:
                    kept[i + j] = torch.sigmoid(logit[j, 0]).numpy()
    return sc, sc_perm, kept


def make_batches(n, batch, epoch, seed):
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    order = torch.randperm(n, generator=g)
    return [order[i:i + batch] for i in range(0, n, batch)]


def train_model(name, x, y, xv, yv, sv, cfg, in_ch, plant, epochs, seed, log):
    """Train one U-Net. Returns (model, history). Validation AUC picks the epoch to keep."""
    torch.manual_seed(seed)
    mc = cfg["model"]
    model = UNet(in_ch=in_ch, base=mc["base"], depth=mc["depth"])
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    crop, bs = mc["crop"], mc["batch_size"]
    gcrop = torch.Generator().manual_seed(seed + 7)
    best, best_state, history = -1.0, None, []
    print(f"[{name}] {n_params(model):,} parameters, {len(x)} training tiles, {len(xv)} validation tiles, "
          f"{epochs} epochs, batch {bs}, crop {crop}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        batches = make_batches(len(x), bs, epoch, seed)
        for b, idx in enumerate(batches, 1):
            oy, ox = (int(v) for v in torch.randint(0, x.shape[-1] - crop + 1, (2,), generator=gcrop))
            xb = x[idx][..., oy:oy + crop, ox:ox + crop]
            yb = y[idx][..., oy:oy + crop, ox:ox + crop].float()
            if int(torch.randint(0, 2, (1,), generator=gcrop)):
                xb, yb = xb.flip(-1), yb.flip(-1)
            if int(torch.randint(0, 2, (1,), generator=gcrop)):
                xb, yb = xb.flip(-2), yb.flip(-2)
            loss = loss_fn(model(planted(xb, yb, plant)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
            if b % log == 0 or b == len(batches):
                print(f"[{name}] epoch {epoch}/{epochs} step {b}/{len(batches)} loss {float(loss.detach()):.4f}", flush=True)
        sc, _, _ = score(model, xv, yv, sv, mc["eval_batch_size"], plant=plant)
        m = sc.all.metrics()
        model.eval()
        with torch.no_grad():                                           # validation loss on the first batch
            vb = slice(0, min(len(xv), mc["eval_batch_size"]))
            vloss = float(loss_fn(model(planted(xv[vb], yv[vb], plant)), yv[vb].float()))
        history.append(dict(epoch=epoch, train_loss=round(total / steps, 5), val_loss=round(vloss, 5),
                            val_auc=round(m["auc"], 5), val_ap=round(m["ap"], 5), val_iou=round(m["iou"], 5),
                            seconds=round(time.time() - t0, 1)))
        print(f"[{name}] epoch {epoch}/{epochs} done: train loss {history[-1]['train_loss']:.4f}  "
              f"val loss {vloss:.4f}  val AUC {m['auc']:.4f}  val IoU {m['iou']:.4f}  ({history[-1]['seconds']} s)",
              flush=True)
        if m["auc"] > best:
            best = m["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history[-1]["best"] = True
    model.load_state_dict(best_state)
    return model, history


def ascii_curve(history, key="val_auc", width=40):
    lines = [f"validation AUC per epoch (bar = 0.5 .. 1.0):"]
    for h in history:
        n = int(round(max(0.0, min(1.0, (h[key] - 0.5) / 0.5)) * width))
        lines.append(f"  epoch {h['epoch']:>2} | {'#' * n:<{width}} | {h[key]:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main(config_path: str, in_dir: str, out_dir: str):
    cfg = yaml.safe_load(open(config_path))
    mc = cfg["model"]
    torch.set_num_threads(int(mc.get("threads", 0)) or torch.get_num_threads())
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = int(cfg["seed"])
    d, out = Path(in_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((d / "manifest.json").read_text())

    tr, ev = load_file(str(d / "train.safetensors")), load_file(str(d / "eval.safetensors"))
    val_fold = int(mc["val_fold"])
    is_val = tr["fold"] == val_fold
    assert is_val.any() and (~is_val).any(), f"fold {val_fold} is empty or is the only fold"
    x_tr, y_tr = tr["x"][~is_val], tr["y"][~is_val]
    x_va, y_va, s_va = tr["x"][is_val], tr["y"][is_val], tr["source"][is_val]
    # the validation set used at every epoch can be thinned (every k-th tile) to save time
    step = max(1, int(mc.get("val_every", 1)))
    x_va, y_va, s_va = x_va[::step], y_va[::step], s_va[::step]
    x_te, y_te, s_te = ev["x"], ev["y"], ev["source"]
    del tr["x"], tr["y"]                           # free the full-size copies (RAM)
    fg_train = float(y_tr.float().mean())
    print(f"data: {len(x_tr)} train tiles (folds != {val_fold}), {len(x_va)} validation tiles (fold {val_fold}), "
          f"{len(x_te)} TEST tiles (held out); training foreground rate {fg_train:.3f}", flush=True)

    t_all = time.time()
    model, history = train_model("model", x_tr, y_tr, x_va, y_va, s_va, cfg, 1, False, int(mc["epochs"]), seed, int(mc["log_every"]))
    print(ascii_curve(history), flush=True)

    # ---- held-out test: model, negative controls, positive control
    g = torch.Generator().manual_seed(seed + 99)
    perm = torch.randperm(len(x_te), generator=g)
    # a few test tiles (first tile with particles of every source) for the example figure
    keep = {}
    for k in range(len(SOURCES)):
        for i in torch.nonzero(s_te == k).flatten().tolist():
            if int(y_te[i].sum()) > 200:
                keep[i] = k
                break
    sc, sc_perm, kept = score(model, x_te, y_te, s_te, mc["eval_batch_size"], permute=perm, keep=set(keep))
    dummy = Scores()
    logit_const = float(np.log(fg_train / (1 - fg_train)))
    for i in range(0, len(x_te), mc["eval_batch_size"]):
        yb = y_te[i:i + mc["eval_batch_size"]]
        dummy.add(torch.full(yb.shape, logit_const), yb, s_te[i:i + mc["eval_batch_size"]])

    cn = cfg["controls"]
    n_pc = min(int(cn["positive_tiles"]), len(x_tr))
    pc_idx = torch.arange(0, len(x_tr), max(1, len(x_tr) // n_pc))[:n_pc]
    pmodel, phist = train_model("planted", x_tr[pc_idx], y_tr[pc_idx], x_va, y_va, s_va, cfg, 2, True,
                                int(cn["positive_epochs"]), seed + 1, int(mc["log_every"]))
    sc_pos, _, _ = score(pmodel, x_te, y_te, s_te, mc["eval_batch_size"], plant=True)

    results = dict(
        format_version=1, seed=seed,
        data=dict(
            tile=manifest["tile"], folds=manifest["folds"], val_fold=val_fold,
            train_tiles=int(len(x_tr)), val_tiles=int(len(x_va)), test_tiles=int(len(x_te)),
            test_tiles_by_source={s: int((s_te == k).sum()) for k, s in enumerate(SOURCES)},
            train_tiles_by_source={s: int((tr["source"][~is_val] == k).sum()) for k, s in enumerate(SOURCES)},
            train_groups=int(len(set(tr["group"].tolist()))), test_groups=int(len(set(ev["group"].tolist()))),
            train_foreground_rate=round(fg_train, 5),
            test_foreground_rate=round(float(y_te.float().mean()), 5),
            n_images_total=len(manifest["images"]), file_sha256={k: v["sha256"] for k, v in manifest["files"].items()}),
        model=dict(name="U-Net", parameters=n_params(model), base=mc["base"], depth=mc["depth"], crop=mc["crop"],
                   batch_size=mc["batch_size"], epochs=int(mc["epochs"]), lr=mc["lr"], loss="BCE + soft Dice",
                   best_epoch=max(history, key=lambda h: h["val_auc"])["epoch"]),
        history=history, history_planted=phist,
        test=dict(model=sc.summary(), dummy=dummy.summary(), permuted=sc_perm.summary(), planted=sc_pos.summary()),
        timing=dict(seconds_total=round(time.time() - t_all, 1)),
        software=dict(torch=torch.__version__, threads=torch.get_num_threads()),
    )
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    idx = sorted(kept)
    np.savez_compressed(out / "examples.npz", source=np.array([keep[i] for i in idx]), x=x_te[idx, 0].numpy(),
                        y=y_te[idx, 0].numpy(), prob=np.stack([kept[i] for i in idx]))

    t = results["test"]
    print("\nTEST (held out, pixel-wise)            AUC     AP    IoU", flush=True)
    for name, label in [("model", "trained U-Net"), ("dummy", "negative: constant score"),
                        ("permuted", "negative: shuffled masks"), ("planted", "positive: mask given")]:
        m = t[name]["overall"]
        print(f"  {label:<30} {m['auc']:>7.4f} {m['ap']:>6.3f} {m['iou']:>6.3f}", flush=True)
    print(f"wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
