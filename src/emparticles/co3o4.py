"""Co3O4 nanocrystals (Zenodo 14927582, CC BY 4.0): 256 hand-labelled 512x512 TEM images in HDF5.

training_images.h5 -> 'images'  (256, 512, 512) float64
training_labels.h5 -> 'labels'  (256, 512, 512, 2) float64, one-hot (channel `particle_channel` = particle)
The files carry no pixel size: the record only states a range (67-86 pm per pixel), so
pixel_size_nm is NULL with scale_status 'range_only' rather than an invented value.
Labels are semantic (touching particles are merged); particles are connected components.
There is no acquisition / group information, so every image is its own split group.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from .common import assign_split, download, particle_rows, png_bytes


def load(con, cfg: dict, raw: Path, root_cfg: dict, annotations: Path):
    root = raw / "co3o4"
    if cfg.get("images_path"):                       # local copies (tests)
        img_p, lab_p = Path(cfg["images_path"]), Path(cfg["labels_path"])
    else:
        img_p = download(cfg["images_url"], root / "training_images.h5")
        lab_p = download(cfg["labels_url"], root / "training_labels.h5")
    ch = int(cfg.get("particle_channel", 1))

    con.execute(
        "INSERT INTO sources VALUES ('co3o4', ?, 'CC BY 4.0', ?, ?)",
        [
            "256 hand-labelled TEM images of Co3O4 nanocrystals (HDF5, one-hot labels)",
            "https://zenodo.org/records/14927582",
            "not stored in the files; record states 67-86 pm per pixel, so pixel_size_nm is NULL",
        ],
    )
    seed, frac = root_cfg["seed"], root_cfg["eval_fraction"]
    n_img = n_part = 0
    with h5py.File(img_p, "r") as fi, h5py.File(lab_p, "r") as fl:
        di, dl = fi[cfg.get("images_key", "images")], fl[cfg.get("labels_key", "labels")]
        n = di.shape[0] if not cfg.get("limit") else min(int(cfg["limit"]), di.shape[0])
        for i in range(n):
            arr = np.asarray(di[i], dtype=np.float64)
            binary = (np.asarray(dl[i])[..., ch] > 0.5).astype(np.uint8)
            lo, hi = float(arr.min()), float(arr.max())
            norm = np.zeros(arr.shape, np.uint16) if hi == lo else np.round((arr - lo) / (hi - lo) * 65535).astype(np.uint16)
            inst, _ = ndimage.label(binary)
            key = f"{i:03d}"
            image_id = f"co3o4:{key}"
            rows = particle_rows(image_id, inst.astype(np.uint16), None)
            h, w = arr.shape
            con.execute(
                "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    image_id, "co3o4", key, f"co3o4-{key}", w, h, None, "range_only", None, None,
                    assign_split(f"co3o4-{key}", seed, frac), "float64->uint16", lo, hi,
                    png_bytes(norm), len(rows),
                ],
            )
            con.execute("INSERT INTO masks VALUES (?,?,?)", [image_id, "binary", png_bytes(binary)])
            if rows:
                con.executemany("INSERT INTO particles VALUES (?,?,?,?,?,?,?,?,?)", rows)
            n_img += 1
            n_part += len(rows)
    if not cfg.get("keep_raw", True) and not cfg.get("images_path"):
        img_p.unlink(missing_ok=True)
        lab_p.unlink(missing_ok=True)
    print(f"co3o4: {n_img} images, {n_part} particles")
