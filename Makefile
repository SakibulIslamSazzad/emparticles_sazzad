# make all   -> data/dataset.db, then the preliminary figures and results/summary.json
# Recipes: 'output: inputs' followed by a tab-indented command ($@ = output, $< = first input)
CONFIG ?= config.yaml

all: results/summary.json

data/dataset.db: src/emparticles/etl.py src/emparticles/common.py src/emparticles/emps.py \
                 src/emparticles/hrtem.py src/emparticles/co3o4.py schema.sql $(CONFIG) \
                 annotations/emps_scale.csv
	uv run python -m emparticles.etl $(CONFIG) $@

results/summary.json: data/dataset.db src/emparticles/figures.py
	uv run python -m emparticles.figures $< figures $@

clean:
	rm -rf data/dataset.db results figures

.PHONY: all clean
