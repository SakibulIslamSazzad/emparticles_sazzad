"""Turn data/dataset.db into model-ready tensors: data/inputs/{train,eval}.safetensors.

    python -m emparticles.tensors config.yaml data/dataset.db data/inputs

What happens, in order (every query has an ORDER BY, nothing is random):

1. The split is the one stored in the database. It was decided per GROUP (EMPS: the paper,
   HRTEM: the acquisition session, Co3O4: the image), so all pictures of a group, and all
   tiles of a picture, land on the same side. `assert_no_group_overlap` refuses to write
   anything if a group ever appears on both sides.
2. Every picture is cut into non-overlapping `tile` x `tile` squares (centred, so the
   leftover border is split evenly). Tiles are the rows of the tensors. A picture smaller
   than one tile is skipped and listed in the manifest.
3. Intensities are stretched PER PICTURE (1st..99th percentile -> 0..1). Nothing is computed
   across pictures, so no statistic of the eval set can reach the training set.
4. The label is "particle" (1) versus background (0): EMPS instance masks become `mask > 0`,
   HRTEM and Co3O4 are binary already. Instance ids stay in the database.
5. Cross-validation: the TRAIN groups are dealt into `folds` folds, source by source (so each
   fold holds every source), largest group first, each to the currently smallest fold (ties ->
   lowest fold number). A group is never in two folds.
6. Two files plus `manifest.json` are written atomically (temporary file, then rename). The
   bytes depend only on the database and the config, so running it twice gives identical files.

Tensors in each file (N tiles, T = tile size):
    x              float32 [N, 1, T, T]  picture, 0..1
    y              uint8   [N, 1, T, T]  1 = particle
    source         uint8   [N]           0 emps, 1 hrtem, 2 co3o4
    group          int32   [N]           index into manifest["groups"]
    image          int32   [N]           index into manifest["images"]
    fold           int8    [N]           0..folds-1 for train, -1 for eval
    pixel_size_nm  float32 [N]           nm per pixel of the stored picture, NaN if unknown
    origin         int32   [N, 2]        (row, col) of the tile inside the stored picture
"""
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import yaml
from PIL import Image
from safetensors.numpy import save_file

FORMAT_VERSION = 1
SOURCES = ["emps", "hrtem", "co3o4"]          # position = integer code in the `source` tensor
SPLITS = ["train", "eval"]


def gray(png: bytes) -> np.ndarray:
    """Decode a PNG to a 2-D float64 array (RGB pictures are averaged to grey)."""
    a = np.asarray(Image.open(io.BytesIO(png)))
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a.astype(np.float64)


def stretch(a: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Per-picture contrast stretch to 0..1 (a flat picture becomes all zeros)."""
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(a.shape, np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def tile_origins(h: int, w: int, t: int):
    """Top-left corners of the non-overlapping t x t tiles, centred on the picture."""
    ny, nx = h // t, w // t
    y0, x0 = (h - ny * t) // 2, (w - nx * t) // 2
    return [(y0 + i * t, x0 + j * t) for i in range(ny) for j in range(nx)]


def assign_folds(group_tiles: dict, k: int) -> dict:
    """Deal groups ('<source>:<key>') into k folds, one source at a time so that every fold
    gets a share of every source. Inside a source: biggest group first, each to the fold with
    the fewest tiles of that source so far (ties: fewest tiles overall, then lowest number)."""
    total = [0] * k
    per_source = {}
    fold = {}
    for g in sorted(group_tiles, key=lambda g: (g.split(":", 1)[0], -group_tiles[g], g)):
        load = per_source.setdefault(g.split(":", 1)[0], [0] * k)
        f = min(range(k), key=lambda i: (load[i], total[i], i))
        fold[g] = f
        load[f] += group_tiles[g]
        total[f] += group_tiles[g]
    return fold


def assert_no_group_overlap(train_groups, eval_groups):
    both = set(train_groups) & set(eval_groups)
    if both:
        raise AssertionError(f"leakage: {len(both)} groups are in train and eval, e.g. {sorted(both)[:3]}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build(cfg: dict, db_path, out_dir) -> dict:
    """Write train/eval .safetensors and manifest.json into out_dir; return the manifest."""
    tc = cfg["tensors"]
    t, k = int(tc["tile"]), int(tc["folds"])
    dtype = np.dtype(tc.get("dtype", "float32"))
    lo_pct, hi_pct = tc.get("stretch_percentiles", [1, 99])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT image_id, source, group_key, split, width, height, pixel_size_nm "
        "FROM images ORDER BY source, group_key, image_id"
    ).fetchall()

    images, skipped = [], []                       # images that yield at least one tile
    for image_id, source, group_key, split, w, h, px in rows:
        origins = tile_origins(h, w, t)
        if not origins:
            skipped.append(image_id)
            continue
        images.append(dict(image_id=image_id, source=source, group=f"{source}:{group_key}",
                           split=split, px=px, origins=origins))

    groups = sorted({im["group"] for im in images})
    gidx = {g: i for i, g in enumerate(groups)}
    group_split = {}
    for im in images:
        if group_split.setdefault(im["group"], im["split"]) != im["split"]:
            raise AssertionError(f"group {im['group']} has images in both splits")
    assert_no_group_overlap([g for g, s in group_split.items() if s == "train"],
                            [g for g, s in group_split.items() if s == "eval"])

    group_tiles = {}
    for im in images:
        group_tiles[im["group"]] = group_tiles.get(im["group"], 0) + len(im["origins"])
    folds = assign_folds({g: n for g, n in group_tiles.items() if group_split[g] == "train"}, k)

    manifest = {
        "format_version": FORMAT_VERSION,
        "seed": cfg["seed"], "tile": t, "folds": k, "dtype": str(dtype),
        "stretch_percentiles": [lo_pct, hi_pct],
        "sources": SOURCES,
        "skipped_smaller_than_tile": skipped,
        "groups": [dict(index=gidx[g], name=g, split=group_split[g], fold=folds.get(g, -1),
                        n_tiles=group_tiles[g]) for g in groups],
        "images": [dict(index=i, image_id=im["image_id"], group=gidx[im["group"]], split=im["split"],
                        n_tiles=len(im["origins"])) for i, im in enumerate(images)],
    }
    img_index = {im["image_id"]: i for i, im in enumerate(images)}

    files = {}
    for split in SPLITS:
        sel = [im for im in images if im["split"] == split]
        n = sum(len(im["origins"]) for im in sel)
        x = np.zeros((n, 1, t, t), dtype)
        y = np.zeros((n, 1, t, t), np.uint8)
        source = np.zeros(n, np.uint8)
        group = np.zeros(n, np.int32)
        image = np.zeros(n, np.int32)
        fold = np.full(n, -1, np.int8)
        pix = np.full(n, np.nan, np.float32)
        origin = np.zeros((n, 2), np.int32)
        r = 0
        for im in sel:
            png, mpng = con.execute(
                "SELECT i.image_png, m.mask_png FROM images i JOIN masks m USING (image_id) "
                "WHERE i.image_id = ?", [im["image_id"]]).fetchone()
            pic = stretch(gray(png), lo_pct, hi_pct)
            mask = np.asarray(Image.open(io.BytesIO(mpng))) > 0
            if mask.shape != pic.shape:
                raise ValueError(f"{im['image_id']}: mask {mask.shape} != picture {pic.shape}")
            for (oy, ox) in im["origins"]:
                x[r, 0] = pic[oy:oy + t, ox:ox + t]
                y[r, 0] = mask[oy:oy + t, ox:ox + t]
                source[r] = SOURCES.index(im["source"])
                group[r] = gidx[im["group"]]
                image[r] = img_index[im["image_id"]]
                fold[r] = folds.get(im["group"], -1)
                pix[r] = np.nan if im["px"] is None else im["px"]
                origin[r] = (oy, ox)
                r += 1
        # ONE metadata key holding sorted JSON: safetensors writes several keys in random
        # order, which would make two identical builds differ byte-wise.
        meta = {"info": json.dumps({"format_version": FORMAT_VERSION, "split": split, "tile": t,
                                    "seed": cfg["seed"], "folds": k, "sources": SOURCES,
                                    "axes": "x,y: N,1,T,T"}, sort_keys=True)}
        path = out / f"{split}.safetensors"
        tmp = path.with_name(path.name + ".tmp")
        save_file(dict(x=x, y=y, source=source, group=group, image=image, fold=fold,
                       pixel_size_nm=pix, origin=origin), str(tmp), metadata=meta)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        files[split] = dict(file=path.name, n_tiles=n, sha256=sha256(path))
        print(f"tensors: {split}: {n} tiles from {len(sel)} pictures -> {path}")
        del x, y

    manifest["files"] = files
    mpath = out / "manifest.json"
    mtmp = mpath.with_name(mpath.name + ".tmp")
    mtmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(mtmp, mpath)
    return manifest


def main(config_path: str, db_path: str, out_dir: str):
    build(yaml.safe_load(open(config_path)), db_path, out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:4])
