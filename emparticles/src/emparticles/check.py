"""Audit data/inputs/{train,eval}.safetensors against data/dataset.db.

    python -m emparticles.check config.yaml data/dataset.db data/inputs

Every check prints one line and the script exits non-zero on the first failure:

  shapes      the tensors have the documented names, dtypes and shapes
  values      pictures are finite and in 0..1, masks are 0/1
  groups      no group is in both files; every group is in exactly one fold
  images      no picture is in both files; its split matches the database
  tiles       no identical (non-constant) tile occurs in both files
  folds       every fold is non-empty and folds never mix groups
  idempotent  rebuilding from the same database gives byte-identical files
  loading     the files load through safetensors.torch + DataLoader (numpy if torch is absent)
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import yaml
from safetensors import safe_open

from .tensors import SOURCES, build, sha256

KEYS = {"x": 4, "y": 4, "source": 1, "group": 1, "image": 1, "fold": 1, "pixel_size_nm": 1, "origin": 2}


def ok(name: str, msg: str):
    print(f"check {name:<11} ok  {msg}", flush=True)


def load(path: Path) -> dict:
    with safe_open(str(path), framework="numpy") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def tile_hashes(x: np.ndarray) -> set:
    """Hashes of the non-constant tiles (blank tiles legitimately look alike)."""
    out = set()
    for tile in x:
        if tile.min() != tile.max():
            out.add(hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest())
    return out


def main(config_path: str, db_path: str, in_dir: str):
    cfg = yaml.safe_load(open(config_path))
    t, k = int(cfg["tensors"]["tile"]), int(cfg["tensors"]["folds"])
    d = Path(in_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    data = {s: load(d / f"{s}.safetensors") for s in ("train", "eval")}

    # shapes
    for s, tens in data.items():
        assert set(tens) == set(KEYS), f"{s}: tensors are {sorted(tens)}"
        n = len(tens["x"])
        for key, ndim in KEYS.items():
            assert tens[key].ndim == ndim and len(tens[key]) == n, f"{s}.{key} has shape {tens[key].shape}"
        assert tens["x"].shape[1:] == (1, t, t) and tens["y"].shape[1:] == (1, t, t), f"{s}: wrong tile shape"
        assert n == manifest["files"][s]["n_tiles"] > 0, f"{s}: tile count differs from the manifest"
    ok("shapes", f"train {len(data['train']['x'])} tiles, eval {len(data['eval']['x'])} tiles, tile {t}x{t}")

    # values
    for s, tens in data.items():
        assert np.isfinite(tens["x"]).all() and tens["x"].min() >= 0 and tens["x"].max() <= 1, f"{s}: x outside 0..1"
        assert set(np.unique(tens["y"])) <= {0, 1}, f"{s}: y is not 0/1"
        assert (tens["source"] < len(SOURCES)).all(), f"{s}: bad source code"
    ok("values", "x in 0..1, y in {0,1}")

    # groups
    tg, eg = set(data["train"]["group"].tolist()), set(data["eval"]["group"].tolist())
    assert not (tg & eg), f"{len(tg & eg)} groups are in both train and eval"
    for g in manifest["groups"]:
        side = "train" if g["index"] in tg else "eval"
        assert g["split"] == side, f"group {g['name']} is {g['split']} in the manifest but in {side}.safetensors"
    ok("groups", f"{len(tg)} train groups and {len(eg)} eval groups, no overlap")

    # images, against the database
    ti, ei = set(data["train"]["image"].tolist()), set(data["eval"]["image"].tolist())
    assert not (ti & ei), "a picture is in both train and eval"
    con = duckdb.connect(db_path, read_only=True)
    db_split = dict(con.execute("SELECT image_id, split FROM images ORDER BY image_id").fetchall())
    for im in manifest["images"]:
        side = "train" if im["index"] in ti else "eval"
        assert db_split[im["image_id"]] == side == im["split"], f"{im['image_id']}: split differs from the database"
    straddle = con.execute(
        "SELECT source, group_key FROM images GROUP BY source, group_key "
        "HAVING count(DISTINCT split) > 1 ORDER BY source, group_key").fetchall()
    assert not straddle, f"database groups in both splits: {straddle[:3]}"
    ok("images", f"{len(ti)} train and {len(ei)} eval pictures agree with the database split")

    # tiles
    both = tile_hashes(data["train"]["x"]) & tile_hashes(data["eval"]["x"])
    assert not both, f"{len(both)} identical tiles occur in train and eval"
    ok("tiles", "no identical tile in both files")

    # folds
    f = data["train"]["fold"]
    assert set(np.unique(f)) == set(range(k)), f"folds present: {sorted(set(f.tolist()))}"
    assert (data["eval"]["fold"] == -1).all(), "eval tiles must have fold -1"
    for g in tg:
        assert len(set(f[data["train"]["group"] == g].tolist())) == 1, f"group {g} is in several folds"
    ok("folds", "tiles per fold " + str([int((f == i).sum()) for i in range(k)]) + ", each group in one fold")

    # idempotency
    with tempfile.TemporaryDirectory() as tmp:
        again = build(cfg, db_path, tmp)
        for s in ("train", "eval"):
            assert sha256(Path(tmp) / f"{s}.safetensors") == sha256(d / f"{s}.safetensors"), f"{s}: rebuild differs"
        assert (Path(tmp) / "manifest.json").read_bytes() == (d / "manifest.json").read_bytes(), "manifest differs"
    ok("idempotent", "a rebuild from the same database is byte-identical")

    # loading
    try:
        import torch
        from safetensors.torch import load_file
        from torch.utils.data import DataLoader, TensorDataset
        tr = load_file(str(d / "train.safetensors"))
        xb, yb = next(iter(DataLoader(TensorDataset(tr["x"], tr["y"]), batch_size=8)))
        assert xb.shape == (8, 1, t, t) and yb.shape == (8, 1, t, t)
        assert xb.dtype == getattr(torch, cfg["tensors"].get("dtype", "float32")), f"x has dtype {xb.dtype}"
        ok("loading", f"safetensors.torch + DataLoader gave a batch {tuple(xb.shape)}")
    except ImportError:
        ok("loading", "torch is not installed here; the files were read with safetensors (same format)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(*sys.argv[1:4])
