.PHONY: install test regimes shortcut all

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

regimes:
	python -m experiments.selection_regimes

shortcut:
	python -m experiments.shortcut_challenge

all:
	python -m experiments.run_all --config configs/reference.yaml
