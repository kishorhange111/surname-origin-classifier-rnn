.PHONY: install install-dev run data train lint format test check clean

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

data:           ## download the surnames dataset into data/names/
	curl -O https://download.pytorch.org/tutorial/data.zip && unzip -o data.zip

train:
	python train.py --data data/names --epochs 27

lint:
	ruff check .

format:
	ruff check --fix .

test:
	pytest

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
