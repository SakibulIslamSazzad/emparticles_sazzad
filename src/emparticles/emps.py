"""EMPS: 465 literature images, uint16 instance masks, scale bar burnt into the picture.

Pixel size comes from annotations/emps_scale.csv: for every image a person identified the bar,
measured its length in pixels and read the printed value (nm or um). Images without a readable
scale keep pixel_size_nm = NULL and scale_status = 'no_scale'.
"""
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .common import assign_split, download, particle_rows, png_bytes, unzip

csv.field_size_limit(10**9)


def fetch(cfg: dict, raw: Path) -> Path:
    override = cfg.get("path")
    if override:                                   # a local copy (for tests)
        return Path(override)
    root = raw / "emps"
    if not (root / "images").exists():
        z = download(cfg["url"], raw / "emps.zip")
        unzip(z, raw / "_emps_unzip")
        inner = next((raw / "_emps_unzip").glob("*/images")).parent
        inner.rename(root)
    return root


def load(con, cfg: dict, raw: Path, root_cfg: dict, annotations: Path):
    root = fetch(cfg, raw)
    names = sorted(p.name for p in (root / "images").glob("*.png"))
    if cfg.get("limit"):
        names = names[: int(cfg["limit"])]
    doi = {}
    with open(root / "metadata.csv", newline="") as f:
        for r in csv.DictReader(f):
            doi[r["filename"]] = r["doi"]
    scale = {}
    with open(annotations / "emps_scale.csv", newline="") as f:
        for r in csv.DictReader(f):
            scale[r["filename"]] = r

    con.execute(
        "INSERT INTO sources VALUES ('emps', ?, 'MIT', ?, ?)",
        [
            "465 TEM/SEM images from the literature with hand-labelled instance masks",
            "https://github.com/by256/emps",
            "scale bar burnt into the image; bar length and printed value were read by hand "
            "(annotations/emps_scale.csv)",
        ],
    )
    seed, frac = root_cfg["seed"], root_cfg["eval_fraction"]
    n_img = n_part = 0
    for name in names:
        key = name[:-4]
        image_id = f"emps:{key}"
        img_path = root / "images" / name
        inst = np.array(Image.open(root / "segmaps" / name))
        if inst.ndim == 3:
            inst = inst[..., 0]
        inst = inst.astype(np.uint16)
        pic = Image.open(img_path)
        w, h = pic.size
        s = scale.get(name, {})
        status = s.get("status", "no_scale")
        ok = status in ("ok", "ok_manual_px") and s.get("pixel_size_nm")
        px_nm = float(s["pixel_size_nm"]) if ok else None
        group = doi.get(name, key)                 # images of one paper stay on one side
        rows = particle_rows(image_id, inst, px_nm)
        con.execute(
            "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                image_id, "emps", name, group, w, h, px_nm,
                "ok" if ok else "no_scale",
                int(s["bar_px"]) if ok and s.get("bar_px") else None,
                float(s["scale_nm"]) if ok and s.get("scale_nm") else None,
                assign_split(group, seed, frac), "uint8",
                None, None, img_path.read_bytes(), len(rows),
            ],
        )
        con.execute("INSERT INTO masks VALUES (?,?,?)", [image_id, "instance", png_bytes(inst)])
        if rows:
            con.executemany("INSERT INTO particles VALUES (?,?,?,?,?,?,?,?,?)", rows)
        n_img += 1
        n_part += len(rows)
    print(f"emps: {n_img} images, {n_part} particles")
