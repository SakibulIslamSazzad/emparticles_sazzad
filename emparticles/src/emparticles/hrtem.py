"""HRTEM nanoparticles (Au, Ag, CdSe...): raw OneView .dm3 images, binary label PNGs, CC0.

Files live on NERSC: <nersc_url>/<Folder>/<File name> (image) and
<nersc_url>/<Folder>/Labels/<name>_label.png (label). The list of images comes from the
dataset's Dataset_metadata.csv (kept in annotations/hrtem_metadata.csv, columns Folder and
File name). The same File name can occur in several session folders: those are different
acquisitions, so the image id uses Folder + name. Pixel size is read from the .dm3 header.
Labels are binary (touching particles are merged); particles are connected components.
"""
import csv
import hashlib
from urllib.parse import quote
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .common import Missing, assign_split, download, particle_rows, png_bytes

_TO_NM = {"nm": 1.0, "um": 1e3, "µm": 1e3, "µm": 1e3, "micron": 1e3, "a": 0.1, "å": 0.1,
          "angstrom": 0.1, "pm": 1e-3, "m": 1e9}


def read_dm3(path: Path):
    """Return (2-D float array, pixel size in nm or None)."""
    from ncempy.io import dm

    with dm.fileDM(str(path), verbose=False) as f:
        ds = f.getDataset(0)
        arr = np.asarray(ds["data"], dtype=np.float64)
        size, unit = ds.get("pixelSize"), ds.get("pixelUnit")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2:]) if arr.shape[0] == 1 else arr[0]
    px = None
    try:
        u = str(unit[0] if isinstance(unit, (list, tuple, np.ndarray)) else unit).strip().lower()
        s = float(size[0] if isinstance(size, (list, tuple, np.ndarray)) else size)
        if u in _TO_NM and s > 0:
            px = s * _TO_NM[u]
    except Exception:
        px = None
    return arr, px


def load(con, cfg: dict, raw: Path, root_cfg: dict, annotations: Path):
    root = raw / "hrtem"
    base = cfg["nersc_url"].rstrip("/")
    meta = annotations / cfg.get("metadata_csv", "hrtem_metadata.csv")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    csv_px = {(r["Folder"], r["File name"]): r.get("Pixel Scale (nm)") for r in rows}
    rows = sorted(csv_px)
    if cfg.get("limit"):    # deterministic subset: order by hash, not alphabetically
        rows = sorted(rows, key=lambda r: hashlib.sha256(f"{root_cfg['seed']}:{r[0]}/{r[1]}".encode()).hexdigest())
        rows = sorted(rows[: int(cfg["limit"])])
    k = int(cfg.get("downsample", 1))   # raw images are 4096x4096 (65 MB each): bin by k

    con.execute(
        "INSERT INTO sources VALUES ('hrtem', ?, 'CC0', ?, ?)",
        [
            "HRTEM images of Au/Ag/CdSe nanoparticles, raw .dm3 with binary labels",
            "https://doi.org/10.7941/D1SP93",
            "pixel size read from the .dm3 header",
        ],
    )
    seed, frac = root_cfg["seed"], root_cfg["eval_fraction"]
    n_img = n_part = 0
    skipped = []
    for folder, fname in rows:
        stem = Path(fname).stem
        try:
            img_p = download(f"{base}/{quote(folder)}/{quote(fname)}", root / folder / fname)
            lab_p = download(f"{base}/{quote(folder)}/Labels/{quote(stem)}_label.png", root / folder / "Labels" / f"{stem}_label.png")
        except Missing as e:                      # listed in the metadata but not on the server
            skipped.append(f"{folder}/{fname}")
            print(f"hrtem: SKIPPED (missing on server): {e}", flush=True)
            continue
        arr, px_nm = read_dm3(img_p)
        status = "header"
        if px_nm is None and csv_px[(folder, fname)]:
            px_nm, status = float(csv_px[(folder, fname)]), "metadata_csv"
        lab = np.array(Image.open(lab_p))
        if lab.ndim == 3:
            lab = lab[..., 0]
        binary = (lab > 0).astype(np.uint8)
        if binary.shape != arr.shape:
            raise ValueError(f"{folder}/{fname}: label {binary.shape} != image {arr.shape}")
        if k > 1:   # block-mean the image, majority vote the mask; pixel grows by k
            H, W = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
            arr = arr[:H, :W].reshape(H // k, k, W // k, k).mean(axis=(1, 3))
            binary = (binary[:H, :W].reshape(H // k, k, W // k, k).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
            px_nm = None if px_nm is None else px_nm * k
        lo, hi = float(arr.min()), float(arr.max())
        norm = np.zeros(arr.shape, np.uint16) if hi == lo else np.round((arr - lo) / (hi - lo) * 65535).astype(np.uint16)
        inst, _ = ndimage.label(binary)
        image_id = f"hrtem:{folder}/{stem}"
        rows_p = particle_rows(image_id, inst.astype(np.uint16), px_nm)
        h, w = arr.shape
        con.execute(
            "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                image_id, "hrtem", fname, folder, w, h, px_nm,
                status if px_nm else "no_scale", None, None,
                assign_split(folder, seed, frac), "float32->uint16",
                lo, hi, png_bytes(norm), len(rows_p),
            ],
        )
        con.execute("INSERT INTO masks VALUES (?,?,?)", [image_id, "binary", png_bytes(binary)])
        if rows_p:
            con.executemany("INSERT INTO particles VALUES (?,?,?,?,?,?,?,?,?)", rows_p)
        n_img += 1
        n_part += len(rows_p)
        if not cfg.get("keep_raw", True):           # save disk: raw .dm3 is 65 MB each
            img_p.unlink(missing_ok=True)
            lab_p.unlink(missing_ok=True)
    print(f"hrtem: {n_img} images, {n_part} particles, {len(skipped)} skipped (missing on server)")
    if skipped:
        Path(root_cfg["raw_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(root_cfg["raw_dir"]) / "hrtem_skipped.txt").write_text("\n".join(skipped) + "\n")
