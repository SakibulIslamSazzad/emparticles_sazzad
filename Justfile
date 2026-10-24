# `just`        builds everything: data/dataset.db, figures/, data/inputs/{train,eval}.safetensors
# `just check`  audits the tensors (no leakage, folds, idempotency, loading)
podman := "podman run --rm -v .:/work -w /work $(podman build -q .)"

default:
    {{podman}} make all

check:
    {{podman}} make check
