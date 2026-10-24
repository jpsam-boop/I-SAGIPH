.PHONY: setup ingest build-real demo run test app all
setup:      ## install dependencies and the package
	pip install -r requirements.txt && pip install -e .
ingest:     ## download automatable sources; lists manual steps
	python -m isagiph.ingest
build-real: ## your files in data/manual/ -> standard tables in data/interim/
	python -m isagiph.build_real
demo:       ## full pipeline on synthetic data (no downloads needed)
	python -m isagiph.pipeline --demo
run:        ## full pipeline on real data
	python -m isagiph.pipeline
test:
	pytest -q
app:
	streamlit run app/app.py
all: build-real run app
