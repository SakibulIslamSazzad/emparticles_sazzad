# make all    -> data/dataset.db, the preliminary figures, then data/inputs/{train,eval}.safetensors
# make check  -> audit the tensors (leakage, folds, idempotency, loading)
# Recipes: 'output: inputs' followed by a tab-indented command ($@ = output, $< = first input)
CONFIG ?= config.yaml
INPUTS = data/inputs

all: results/summary.json $(INPUTS)/train.safetensors $(INPUTS)/eval.safetensors

data/dataset.db: src/emparticles/etl.py src/emparticles/common.py src/emparticles/emps.py \
                 src/emparticles/hrtem.py src/emparticles/co3o4.py schema.sql $(CONFIG) \
                 annotations/emps_scale.csv
	uv run python -m emparticles.etl $(CONFIG) $@

results/summary.json: data/dataset.db src/emparticles/figures.py
	uv run python -m emparticles.figures $< figures $@

# one script writes both files and manifest.json; eval depends on train so make runs it once
$(INPUTS)/train.safetensors: data/dataset.db src/emparticles/tensors.py $(CONFIG)
	uv run python -m emparticles.tensors $(CONFIG) $< $(INPUTS)

$(INPUTS)/eval.safetensors: $(INPUTS)/train.safetensors
	@test -f $@ || { echo "$@ is missing: run 'rm $<' and make again"; exit 1; }

check: $(INPUTS)/train.safetensors $(INPUTS)/eval.safetensors src/emparticles/check.py
	uv run python -m emparticles.check $(CONFIG) data/dataset.db $(INPUTS)

clean:
	rm -rf data/dataset.db data/inputs results figures

.PHONY: all check clean
