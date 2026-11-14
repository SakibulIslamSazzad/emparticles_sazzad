"""Helpers shared by the source loaders."""
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def png_bytes(arr: np.ndarray) -> bytes:
    """Lossless PNG bytes of a uint8/uint16 2-D (or RGB uint8) array."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


MAX_TRIES = 22   # ~35 min of patience in total: the VM's network sometimes drops for minutes


class Missing(Exception):
    """The server says the file does not exist (HTTP 403/404/410): retrying will not help."""


def download(url: str, dest: Path) -> Path:
    """Download once; an existing file is reused (idempotent)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "emparticles-etl"})
    for attempt in range(MAX_TRIES):                       # transient network errors: retry with backoff
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                tmp.unlink(missing_ok=True)
                raise Missing(f"HTTP {e.code} for {url}") from e
            if attempt == MAX_TRIES - 1:
                raise
            print(f"retry {attempt + 1}/{MAX_TRIES - 1} for {url} (HTTP {e.code})", flush=True)
            time.sleep(min(15 * (attempt + 1), 120))
        except OSError as e:
            if attempt == MAX_TRIES - 1:
                raise
            print(f"retry {attempt + 1}/{MAX_TRIES - 1} for {url} (network: {e})", flush=True)
            time.sleep(min(15 * (attempt + 1), 120))
    tmp.rename(dest)
    return dest


def unzip(zip_path: Path, dest: Path) -> Path:
    """Extract once (a marker file records success)."""
    marker = dest / ".extracted"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    marker.write_text("ok\n")
    return dest


def assign_split(group_key: str, seed: int, eval_fraction: float) -> str:
    """Deterministic group split: the same key always lands in the same split."""
    h = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return "eval" if u < eval_fraction else "train"


def particle_rows(image_id: str, inst: np.ndarray, pixel_size_nm):
    """One row per instance id in an integer label image (0 = background)."""
    h, w = inst.shape
    ids = np.unique(inst)
    ids = ids[ids > 0]
    rows = []
    if len(ids) == 0:
        return rows
    flat = inst.ravel()
    ys, xs = np.divmod(np.arange(flat.size), w)
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    starts = np.searchsorted(sorted_ids, ids, side="left")
    ends = np.searchsorted(sorted_ids, ids, side="right")
    for pid, a, b in zip(ids, starts, ends):
        idx = order[a:b]
        px, py = xs[idx], ys[idx]
        area = int(b - a)
        d_px = float(2.0 * np.sqrt(area / np.pi))
        edge = bool(px.min() == 0 or py.min() == 0 or px.max() == w - 1 or py.max() == h - 1)
        rows.append(
            (
                image_id,
                int(pid),
                area,
                None if pixel_size_nm is None else float(area * pixel_size_nm**2),
                d_px,
                None if pixel_size_nm is None else float(d_px * pixel_size_nm),
                edge,
                float(px.mean()),
                float(py.mean()),
            )
        )
    return rows
