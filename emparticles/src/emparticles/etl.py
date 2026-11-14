"""Build data/dataset.db from the sources listed in config.yaml.

    python -m emparticles.etl config.yaml data/dataset.db

Idempotent: the database is built in a temporary file with deterministic ids, ordering and
splits, then renamed over the target, so running it twice gives the same content and an
interrupted run never leaves a half-written database. Downloads are cached in data/raw.
"""
import importlib
import sys
from pathlib import Path

import duckdb
import yaml

SOURCES = ["emps", "hrtem", "co3o4"]


sys.stdout.reconfigure(line_buffering=True)


def main(config_path: str, out_path: str):
    cfg = yaml.safe_load(open(config_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".db.tmp")
    for p in (tmp, Path(str(tmp) + ".wal")):
        p.unlink(missing_ok=True)
    raw = Path(cfg.get("raw_dir", "data/raw"))
    annotations = Path(cfg.get("annotations_dir", "annotations"))

    con = duckdb.connect(str(tmp))
    con.execute(open("schema.sql").read())
    for name in SOURCES:
        scfg = cfg["sources"].get(name, {})
        if not scfg.get("enabled", False):
            print(f"{name}: disabled in {config_path}")
            continue
        mod = importlib.import_module(f"emparticles.{name}")
        mod.load(con, scfg, raw, cfg, annotations)
    con.execute("CHECKPOINT")
    n = con.execute("SELECT source, count(*) FROM images GROUP BY source ORDER BY source").fetchall()
    con.close()
    out.unlink(missing_ok=True)
    tmp.rename(out)
    print("wrote", out, dict(n))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
