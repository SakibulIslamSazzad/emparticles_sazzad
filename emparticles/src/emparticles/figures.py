"""Preliminary-data figures and a summary table, straight from data/dataset.db.

    python -m emparticles.figures data/dataset.db figures results/summary.json
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(db: str, fig_dir: str, summary_path: str):
    con = duckdb.connect(db, read_only=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "images_per_source": dict(con.execute(
            "SELECT source, count(*) FROM images GROUP BY source ORDER BY source").fetchall()),
        "images_with_pixel_size": dict(con.execute(
            "SELECT source, count(pixel_size_nm) FROM images GROUP BY source ORDER BY source").fetchall()),
        "particles_per_source": dict(con.execute(
            "SELECT i.source, count(*) FROM particles p JOIN images i USING(image_id) "
            "GROUP BY i.source ORDER BY i.source").fetchall()),
        "split_counts": [list(r) for r in con.execute(
            "SELECT source, split, count(*) FROM images GROUP BY source, split ORDER BY 1, 2").fetchall()],
    }
    json.dump(summary, open(summary_path, "w"), indent=2)

    # 1) pixel size per source
    rows = con.execute("SELECT source, pixel_size_nm FROM images WHERE pixel_size_nm IS NOT NULL").fetchall()
    fig, ax = plt.subplots(figsize=(7, 4))
    for src in sorted({r[0] for r in rows}):
        v = np.array([r[1] for r in rows if r[0] == src])
        ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, 25), alpha=0.6, label=src)
    ax.set_xscale("log"); ax.set_xlabel("pixel size (nm / px)"); ax.set_ylabel("images"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{fig_dir}/pixel_size.png", dpi=150); plt.close(fig)

    # 2) particle diameter in nm, particles cut by the border excluded
    rows = con.execute(
        "SELECT i.source, p.equiv_diameter_nm FROM particles p JOIN images i USING(image_id) "
        "WHERE p.equiv_diameter_nm IS NOT NULL AND NOT p.touches_edge").fetchall()
    fig, ax = plt.subplots(figsize=(7, 4))
    for src in sorted({r[0] for r in rows}):
        v = np.array([r[1] for r in rows if r[0] == src])
        ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, 40), alpha=0.6,
                label=f"{src} (n={len(v)})")
    ax.set_xscale("log"); ax.set_xlabel("equivalent particle diameter (nm)"); ax.set_ylabel("particles")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{fig_dir}/particle_diameter_nm.png", dpi=150); plt.close(fig)

    # 3) particles per image
    rows = con.execute("SELECT source, n_particles FROM images").fetchall()
    fig, ax = plt.subplots(figsize=(7, 4))
    for src in sorted({r[0] for r in rows}):
        ax.hist([r[1] for r in rows if r[0] == src], bins=30, alpha=0.6, label=src)
    ax.set_xlabel("particles per image"); ax.set_ylabel("images"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{fig_dir}/particles_per_image.png", dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
