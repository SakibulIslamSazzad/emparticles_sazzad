# Debian-slim Python with uv. Everything else (duckdb, numpy, ...) is installed by `uv run`
# from pyproject.toml, so the container only needs make and CA certificates.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends make ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
WORKDIR /work
