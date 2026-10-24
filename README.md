# emparticles — one database for EM nanoparticle segmentation

EMA 5937 *AI/ML for Materials Science*, Project A, **Parts 1 and 2 (ETL)**. Author: Sakibul Islam Sazzad
(UCF, sakibsazzad[at]ucf[dot]edu).

```bash
git clone https://github.com/SakibulIslamSazzad/emparticles_sazzad && cd emparticles_sazzad && just && just check
# or: tar xzvf emparticles.tar.gz && cd emparticles && just && just check
```

`just` runs `Justfile → Containerfile → Makefile → src/emparticles/*.py → data/dataset.db →
data/inputs/{train,eval}.safetensors`, plus the preliminary figures (`figures/*.png`) and
`results/summary.json`. `just check` audits the tensors (no leakage, folds, idempotency, loading).
Without podman: `uv sync && make all && make check`. The default build downloads about 4 GB
(see §3) and needs network access. **Part 2 (tensors for a model) is described in §10.**

---

## 1. What this builds

Three EM nanoparticle datasets arrive in three different conventions. This repository turns them into
**one DuckDB file, `data/dataset.db`**, with one schema, one unit (nm per pixel), one train/eval split
and one table of particles. Everything downstream (figures, model training, leaderboard evaluation)
reads from that file only.

| source | images | what the raw data looks like | labels | pixel size |
|---|---|---|---|---|
| **EMPS** (github.com/by256/emps, MIT) | 465 literature TEM/SEM images, PNG | uint16 **instance** masks | instance ids | **burnt into the image** as a scale bar (see §4) |
| **HRTEM** (doi:10.7941/D1SP93, CC0) | raw OneView `.dm3`, 4096×4096 float32; 407 listed in the metadata | **binary** PNG labels (touching particles are merged) | binary | in the `.dm3` header |
| **Co3O4** (zenodo 14927582, CC BY 4.0) | 256 hand-labelled 512×512 images in HDF5 | one-hot float64, 2 channels | binary (channel 1 = particle) | **not in the files** (record says 67–86 pm/px) |

### Schema (`schema.sql`)

```mermaid
erDiagram
  sources  ||--o{ images    : has
  images   ||--|| masks     : has
  images   ||--o{ particles : contains
  sources { text source PK  text license  text url  text pixel_size_origin }
  images  { text image_id PK  text source FK  text group_key  int width  int height  double pixel_size_nm  text scale_status  text split  blob image_png  int n_particles }
  masks   { text image_id PK  text mask_kind  blob mask_png }
  particles { text image_id PK  int particle_id PK  int area_px  double area_nm2  double equiv_diameter_nm  bool touches_edge  double centroid_x  double centroid_y }
```

* `images.pixel_size_nm` is **nm per pixel for every source**. It is `NULL` when no scale exists, and
  `scale_status` says why: `ok` (EMPS, read from the burnt-in bar), `no_scale` (EMPS, none), `header`
  (HRTEM `.dm3`), `metadata_csv` (HRTEM fallback), `range_only` (Co3O4).
* `masks.mask_kind` is `instance` (EMPS ids) or `binary` (HRTEM, Co3O4; particles are the connected
  components, so **touching particles count as one**).
* `particles.touches_edge` flags particles cut off by the image border; size statistics exclude them.
* `images.split` is `train` or `eval`, assigned **by group** (`group_key`) with a seeded hash:
  EMPS by paper DOI, HRTEM by acquisition-session folder, Co3O4 one group per image (the files carry
  no acquisition information, so this is the weakest split — see *Limitations*).
* Pixels are lossless PNG BLOBs (uint16 for float sources; the original min/max are kept in
  `intensity_min/max`).

### Why these tools

* **DuckDB, a single file**: no server, SQL for the figures, BLOBs for pixels, and the whole database
  is one artifact to hash and compare. (I first tried Zarr + Parquet, then LanceDB; LanceDB crashed in
  the course VM, and two stores doubled the places where metadata and pixels could drift apart.)
* **Build to `dataset.db.tmp`, then rename**: an interrupted build never leaves a half-written database.
* **Idempotent and deterministic**: sorted inputs, deterministic ids, seeded hash splits. A second run of
  the EMPS part gave a byte-identical content hash.
* **Downloads cached in `data/raw`** (HRTEM/Co3O4 raw files are deleted after use, `keep_raw: false`).

---

## 2. How to run, configure

`config.yaml` controls everything: seed, eval fraction, per-source switches, the HRTEM subset size
(`limit`, default 30), downsampling (`downsample: 4`), and `keep_raw`. To change the subset edit `limit` (use `null`
for all 407 HRTEM images, about 26 GB of downloads).

```
make all        # data/dataset.db, figures/, results/summary.json
make clean
```

---

## 3. Sizes and the choices they forced

* HRTEM raw images are 4096×4096 float32 (65 MB each; 26 GB for all 407). I bin them by 4 to
  1024×1024 (block mean for the image, majority vote for the mask) and **multiply the pixel size by
  4**, so physical sizes stay correct. The default build uses a deterministic, hash-ordered subset of
  **30** images (`limit: 30`) so a fresh `just` run stays practical (about 4 GB of downloads in total: 2 GB HRTEM, 1.6 GB Co3O4, 0.1 GB EMPS). Raise `limit` for more HRTEM images; larger values only change the download size.
* Raw HRTEM and Co3O4 files are deleted right after they are processed, so peak disk use stays small.

---

## 4. EMPS: reading the scale bar by hand (and why)

EMPS has masks but **no pixel size**. Each picture carries a scale bar and a label ("200 nm", "2 µm")
burnt into the pixels. Without it, particle sizes cannot be compared in nanometres across images. Pixel
size spans about 0.09 to 5,200 nm/px, and the bars come in every style: black, white, grey; left or
right; plain rectangles, two-sided arrows, whisker rulers.

![A sample of the review tiles: every image gets a tile with the candidate bar drawn on top](docs/img/label_sheet.png)

**Pixel size = printed value / bar length in pixels.** Both numbers have to be right.

### What did not work

Automatic detection was not reliable enough. Scored against my own marks on 30 banner images, the first
bar finder matched 3 of 30 and a thin-line finder 9 of 30; OCR of the labels was also too unreliable
(and a 1000× mix-up between nm and µm would silently ruin every size). I therefore used a
**human-in-the-loop** procedure:

1. A script proposes candidate bars (length in px) and draws them on review sheets.
2. I confirm correct ones, and mark the true bar **in green** on the sheet where the candidate is wrong
   (my hand-drawn marks are slightly slanted, so the script **snaps** each mark to the exact bar edges).
3. I read the printed value and unit from the tile itself.
4. A script (`annotations/emps_scale.csv`) stores `bar_px`, `scale_nm`, `pixel_size_nm` and a status.

Three passes covered all 465 images:

| pass | images | how |
|---|---:|---|
| Round 1 | 204 | detector line matched the bar; value read by eye |
| Excluded early | 50 | no scale bar or no readable value |
| Round 2, bottom strip | 101 | candidates reviewed, wrong ones re-marked |
| Round 2, banner | 40 | bars marked in green, snapped to pixels |
| Round 2, whole image | 70 | full-frame review; most have no bar |
| **total** | **465** | every image has a decision |

![Banner bars: magenta line = extracted bar, label = its length in pixels](docs/img/banner_check.png)

I cross-checked the extracted bars against the images tile by tile. On the bottom strips four bars had
to be corrected (images 126, 206, 289, 319):

![Corrected bars for images 126, 206, 289 and 319](docs/img/bottom_fix.png)

### Edge cases I decided by hand

* **Image 212**: the label is cut off by the left edge (only the letter "m" survives), so there is no
  readable scale. It is flagged `no_scale`, not guessed.
  ![Image 212](docs/img/img212.png)
* **Image 307**: the automatic candidate was a 25 px stub; the real "200 nm" bar runs tick to tick,
  242 px.
* **Image 376**: my table said 20 µm, the tile reads 2.0 µm. Units were re-checked on every tile.
* **Images without a bar** (many SEM pictures): kept, with `pixel_size_nm = NULL` and
  `scale_status = 'no_scale'`.
* **Instance ids** are not always consecutive (3 of 465 masks have gaps, from shapes fully covered or
  outside the image). Rule used everywhere: count the **unique non-zero ids**, never the maximum id.

**Result: 346 of 465 images have a pixel size (74%); 119 have none and keep a flag.** Median pixel size
1.8 nm/px.

![Pixel size distribution](docs/img/pixel_size_hist.png)

The labels live in `annotations/emps_scale_labels.txt` (raw) and `annotations/emps_scale.csv`
(processed); both are committed so the table is reproducible without repeating the manual work.

---

## 5. HRTEM: from raw `.dm3` to particles

* The list of images comes from the dataset's `Dataset_metadata.csv`
  (`annotations/hrtem_metadata.csv`, columns `Folder` and `File name`). Files are fetched from the NERSC
  mirror as `<Folder>/<File name>`, labels from `<Folder>/Labels/<name>_label.png`. Folder names contain
  spaces and a sub-path, so URLs are escaped.
* **The same file name appears in different session folders; those are different acquisitions.** The
  image id is `hrtem:<Folder>/<name>`, never the bare name.
* Pixel size is read from the `.dm3` header with `ncempy` (`pixelSize`, `pixelUnit` = nm). It differs
  between images (different magnifications), which is why a single constant would have been wrong.
  After the 4× binning the values I saw were about 0.10 to 0.17 nm/px.
* Labels are binary: a blob of touching particles is one object, so **particle counts are lower bounds**
  and **sizes of touching particles are overestimates**.
* The split is by session folder: images from one session are near-duplicates.

---

## 6. Co3O4

* `training_images.h5` → `images` (256, 512, 512) float64; `training_labels.h5` → `labels`
  (256, 512, 512, 2) one-hot. The channel means were 0.72 / 0.28, and an overlay confirmed that
  **channel 1 is the particle** (red mask on the cubic crystals, background untouched).
* The files contain **no pixel size**. The Zenodo record gives only a range, 67–86 pm/px, so
  `pixel_size_nm` is `NULL` with `scale_status = 'range_only'` rather than an invented number.

---

## 7. Problems I ran into

| problem | what happened | what I did |
|---|---|---|
| EMPS has no pixel size | scale bars are burnt into the pixels | human-in-the-loop measurement (§4) |
| Automatic bar detection | 3/30 and 9/30 correct on banners; OCR unreliable | candidates + manual confirmation + snapping |
| nm vs µm | a 1000× error if misread | every value checked against its tile |
| Instance ids with gaps | max id ≠ number of particles | count unique non-zero ids |
| Binary masks | touching particles merge | `mask_kind = 'binary'`, documented as a lower bound |
| HRTEM names not unique | same file name, different session | id includes the session folder |
| HRTEM size | 65 MB per image | bin by 4, subset of 30, delete raw after use |
| Co3O4 pixel size | not in the files | stored as NULL + `range_only` |
| LanceDB | crashed in the VM | switched to DuckDB |
| DuckDB 2.0.0 crashed the VM | inside the container `import duckdb` died with exit code 132 (illegal instruction) on the course VM's virtual CPU; numpy, scipy, h5py etc. were fine | pinned `duckdb==1.5.6` in `pyproject.toml` (the version that works in the VM) and re-locked `uv.lock`; bump the pin only after testing on the target machine |
| VM instability | disk I/O errors and the virtual network dropping about every 20–30 minutes while downloading | downloads retry with backoff for ~35 minutes; raw files cleaned as I go; builds done right after a reboot |

---

## 8. Verification (what has and has not been checked)

Coding assistants (Claude) helped write the code; I verified it as follows.

* **Part 1, full database build, 7 Oct 2026** (in the course QEMU VM, `limit: 30`, about 24 minutes):
  `wrote data/dataset.db {'co3o4': 256, 'emps': 465, 'hrtem': 30}`, 0 HRTEM files skipped.

| Source | Images | With pixel size | Particles | Train / eval images |
|---|---|---|---|---|
| EMPS | 465 | 346 | 11,535 | 349 / 116 |
| HRTEM | 30 | 30 | 641 | 17 / 13 |
| Co3O4 | 256 | 0 | 3,790 | 203 / 53 |

* EMPS: a second build gave an identical content hash. Overlays of the HRTEM and Co3O4 masks on their
  images line up with the particles (`check_overlay.py`).
* **Part 2, full pipeline, 24 Oct 2026** (course VM, `limit: 30`, a fresh `just` through podman, then `just check`):
  `train.safetensors` 2,556 tiles from 569 pictures, `eval.safetensors` 876 tiles from 182 pictures
  (453 train and 134 eval groups, no overlap); tiles per fold `[506, 506, 490, 520, 534]`;
  `ALL CHECKS PASSED`, including a byte-identical rebuild. The `safetensors.torch` + `DataLoader` check
  is skipped inside the container because torch is an optional extra (the files are then read with `safetensors`).
* I also broke things on purpose on a small synthetic three-source database (a group split over both
  sides, a training tile copied into eval, a wrong group index) and each was caught.

Not checked: a run with a larger `limit`, and the check on a machine other than my VM.

## 9. Limitations

* HRTEM and Co3O4 labels are binary, so particle counts and sizes of touching particles are biased.
* Co3O4 has no per-image pixel size and no group information (one group per image).
* The HRTEM default build is a 30-image subset, binned by 4: few sessions, so the HRTEM eval share and folds are coarse.
* 119 EMPS images have no usable scale and are `NULL` in nm columns.

---

## 10. Part 2: from the database to tensors

`data/dataset.db` holds pictures as PNG BLOBs of different sizes and masks of different kinds (EMPS has
instance ids, HRTEM and Co3O4 are binary). A model needs fixed-shape batches, so `src/emparticles/tensors.py`
turns the database into two files, `data/inputs/train.safetensors` and `data/inputs/eval.safetensors`,
plus `data/inputs/manifest.json` (every group, every picture, tile counts, file hashes).

**Loading** (the files are standard safetensors, written without needing torch):

```python
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

d = load_file("data/inputs/train.safetensors")
loader = DataLoader(TensorDataset(d["x"], d["y"]), batch_size=16, shuffle=True)

# 5-fold cross-validation, folds are made of whole groups:
k = 0
tr, va = d["fold"] != k, d["fold"] == k
train_set = TensorDataset(d["x"][tr], d["y"][tr])
val_set   = TensorDataset(d["x"][va], d["y"][va])
```

| tensor | dtype, shape (N tiles, T = 256) | meaning |
|---|---|---|
| `x` | float32 `[N,1,T,T]` | picture, stretched per picture to 0..1 |
| `y` | uint8 `[N,1,T,T]` | 1 = particle, 0 = background |
| `source` | uint8 `[N]` | 0 EMPS, 1 HRTEM, 2 Co3O4 |
| `group` | int32 `[N]` | index into `manifest["groups"]` (paper / session / image) |
| `image` | int32 `[N]` | index into `manifest["images"]` |
| `fold` | int8 `[N]` | 0..4 in train, -1 in eval |
| `pixel_size_nm` | float32 `[N]` | nm per pixel of the stored picture, NaN if unknown |
| `origin` | int32 `[N,2]` | (row, col) of the tile in the stored picture |

**No leakage.** The split is by group, never by row or tile: all pictures of one EMPS paper, one HRTEM
session, or one Co3O4 image, and all tiles of a picture, are on the same side (`assign_split` hashes the
group key with the seed; `tensors.py` refuses to write if a group is on both sides). Intensities are
stretched per picture, so no statistic is shared between train and eval. The cross-validation folds are
also built from whole groups of the train set, one source at a time so each fold contains every source
that has enough groups.

**Deterministic.** Every query has an `ORDER BY`; nothing uses random numbers except the seeded hash.
Files are written to a temporary name and renamed, and two builds from the same database are
byte-identical (`just check` rebuilds into a temporary folder and compares hashes).

**What `just check` verifies** (`src/emparticles/check.py`, one line per check): tensor names, dtypes and
shapes; value ranges; no group in both files and each group in exactly one fold; every picture's split
agrees with the database; no identical (non-blank) tile in both files; folds non-empty; byte-identical
rebuild; loading with `safetensors.torch` and a `DataLoader` batch when torch is installed.

**Choices and limits.**
* Pictures are cut into non-overlapping 256×256 tiles, centred; the border that does not fill a tile
  (under 256 px per side) is dropped, and pictures smaller than a tile are skipped and listed in the
  manifest. Tile size, folds, dtype and the stretch percentiles are in `config.yaml` (`tensors:`).
* The label is semantic (particle vs background). Instance ids (EMPS only) stay in the database.
* The pixel size differs between sources and is missing for 119 EMPS images and all of Co3O4: it is
  given as `pixel_size_nm` (NaN if unknown), the pictures are not resampled to a common scale.
* Co3O4 has one group per picture, so its split is the weakest against near-duplicate pictures.
* EMPS pictures with a burnt-in scale-bar banner keep it in their bottom tile.
* torch is optional (`uv sync --extra torch`); the container does not install it.
* Size: about 0.9 GB in total at `float32` with the default 30 HRTEM images (set `dtype: float16` to halve it).
