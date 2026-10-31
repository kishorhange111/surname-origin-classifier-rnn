# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev run data train sync-readme lint format test check clean

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## pinned runtime dependencies + this package (editable)
	pip install -r requirements.txt

install-dev: install ## + lint/test tooling
	pip install -r requirements-dev.txt

run:            ## start the Streamlit app on http://localhost:8501
	streamlit run app.py

data:           ## download the surnames dataset into data/names/
	curl -O https://download.pytorch.org/tutorial/data.zip && unzip -o data.zip && rm data.zip

train:          ## train, evaluate and export models/ + results/
	python scripts/train.py --data data/names --epochs 27 --seed 2024

sync-readme:    ## copy the numbers in results/metrics.json into README.md
	python scripts/sync_readme.py

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the test suite (CPU, offline, uses the committed models/)
	pytest

check: lint test ## everything CI runs

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
