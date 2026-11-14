"""Compile report/report.typ to results/report.pdf (Typst, via the `typst` Python package).

    python -m emparticles.report report/report.typ results/report.pdf

The .typ file reads results/results.json and figures/*.png itself, so every number and figure in
the PDF comes from the latest run.
"""
import sys
from pathlib import Path

import typst


def main(src: str, out: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    typst.compile(src, output=out, root=".")
    print(f"report: wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
