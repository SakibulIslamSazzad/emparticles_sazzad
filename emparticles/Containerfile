# Debian-slim Python with uv. Everything else (duckdb, numpy, ...) is installed by `uv run`
# from pyproject.toml, so the container only needs make, CA certificates and a font for the report.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends make ca-certificates fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
WORKDIR /work
