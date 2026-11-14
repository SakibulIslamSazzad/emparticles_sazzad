-- Schema of data/dataset.db (DuckDB). One row per source, per image, per mask, per particle.
-- Pixel size is always in nanometres per pixel (NULL when the image has no usable scale).

CREATE TABLE sources (
    source             TEXT PRIMARY KEY,   -- 'emps' | 'hrtem' | 'co3o4'
    description        TEXT,
    license            TEXT,
    url                TEXT,
    pixel_size_origin  TEXT                -- how nm/px was obtained for this source
);

CREATE TABLE images (
    image_id        TEXT PRIMARY KEY,      -- '<source>:<unique key>'
    source          TEXT NOT NULL REFERENCES sources(source),
    name            TEXT NOT NULL,         -- file name as published (not unique for hrtem)
    group_key       TEXT NOT NULL,         -- images sharing a key never straddle train/eval
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    pixel_size_nm   DOUBLE,                -- nm per pixel, NULL if unknown
    scale_status    TEXT NOT NULL,         -- 'ok' | 'no_scale' | 'header' | 'metadata_csv' | 'range_only'
    scale_bar_px    INTEGER,               -- emps only: measured bar length in pixels
    scale_bar_nm    DOUBLE,                -- emps only: value printed next to the bar
    split           TEXT NOT NULL,         -- 'train' | 'eval'
    pixel_dtype     TEXT NOT NULL,         -- dtype of the stored picture, e.g. 'uint8'
    intensity_min   DOUBLE,                -- original intensity range (for float sources)
    intensity_max   DOUBLE,
    image_png       BLOB NOT NULL,         -- lossless PNG bytes of the picture
    n_particles     INTEGER NOT NULL
);

CREATE TABLE masks (
    image_id   TEXT PRIMARY KEY REFERENCES images(image_id),
    mask_kind  TEXT NOT NULL,              -- 'instance' (one id per particle) | 'binary'
    mask_png   BLOB NOT NULL               -- uint16 PNG; instance ids 1..N, 0 = background
);

CREATE TABLE particles (
    image_id           TEXT NOT NULL REFERENCES images(image_id),
    particle_id        INTEGER NOT NULL,   -- instance id inside the mask (connected component if binary)
    area_px            INTEGER NOT NULL,
    area_nm2           DOUBLE,             -- NULL without pixel size
    equiv_diameter_px  DOUBLE NOT NULL,    -- diameter of the circle with the same area
    equiv_diameter_nm  DOUBLE,
    touches_edge       BOOLEAN NOT NULL,   -- cut by the image border (size is a lower bound)
    centroid_x         DOUBLE NOT NULL,
    centroid_y         DOUBLE NOT NULL,
    PRIMARY KEY (image_id, particle_id)
);
