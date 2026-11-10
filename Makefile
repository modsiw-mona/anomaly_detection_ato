.PHONY: install_env data intermediate features split feature_pipeline training_pipeline demo notebooks_run test test_verbose test_coverage docs docs_test clean_env switch_main clean_branchs check lint help pre-commit_update demo_bundle

####----Basic configurations----####

install_env: ## Install libs with UV and pre-commit
	@echo "Creating virtual environment using UV"
	uv sync --all-groups
	@echo "Installing pre-commit..."
	uv run pre-commit install
	@echo "Activate virtual environment..."
	@bash -c "source .venv/bin/activate"

data: ## Download, verify and extract the raw dataset (DATASET=rba by default)
	@echo "Downloading dataset $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m data.download --dataset $(or $(DATASET),rba)

intermediate: ## Build data/02_intermediate/<dataset>.parquet from the raw file (DATASET=rba)
	@echo "Building intermediate dataset $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m data.intermediate --dataset $(or $(DATASET),rba)

features: ## Build data/04_feature/<dataset>_features.parquet from the intermediate data (DATASET=rba)
	@echo "Building features for $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m data.features --dataset $(or $(DATASET),rba)

split: ## Assign users to dev/test (grouped by user, stratified by label) (DATASET=rba)
	@echo "Splitting $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m data.split --dataset $(or $(DATASET),rba)

feature_pipeline: ## Run the feature pipeline: data -> intermediate -> features -> split, validating each output (DATASET=rba; VALIDATE_ONLY=1 checks existing outputs)
	@echo "Running the feature pipeline for $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m pipelines.feature_pipeline.run --dataset $(or $(DATASET),rba) $(if $(VALIDATE_ONLY),--validate-only)

training_pipeline: ## Train, validate and save the selected model from the development side (DATASET=rba; REPEATS=3 for the 15-fold estimate)
	@echo "Running the training pipeline for $(or $(DATASET),rba)..."
	PYTHONPATH=src uv run python -m pipelines.training_pipeline.run --dataset $(or $(DATASET),rba) $(if $(REPEATS),--repeats $(REPEATS))

demo: ## Run the Streamlit risk demo (needs the model from notebooks 06 and 07)
	PYTHONPATH=src ATO_DATASET=$(or $(DATASET),rba) ATO_DEMO_OFFLINE=1 uv run streamlit run notebooks/7-deploy/app.py \
		--server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false

demo_bundle: ## Export the public demo bundle (RELEASE=demo-bundle-vN) and refresh notebooks/7-deploy/{bundle.json,requirements.txt}
	$(if $(RELEASE),,$(error RELEASE is required, e.g. make demo_bundle RELEASE=demo-bundle-v1))
	PYTHONPATH=src uv run python -m inference.bundle --dataset $(or $(DATASET),rba) \
		--out data/08_reporting/demo_bundle --release $(RELEASE) \
		--base-url https://github.com/modsiw-mona/anomaly_detection_ato/releases/download/$(RELEASE)/ \
		--manifest notebooks/7-deploy/bundle.json
	uv export --no-default-groups --no-emit-project --format requirements.txt \
		-o notebooks/7-deploy/requirements.txt

notebooks_run: ## Execute a notebook in place, e.g. make notebooks_run NB=notebooks/1-data/x.ipynb
	@echo "Executing $(NB)..."
	PYTHONPATH=src uv run jupyter nbconvert --to notebook --execute --inplace "$(NB)"


####----Tests----####
test: ## Test the code with pytest and coverage
	@echo "Testing code: Running pytest"
	@uv run pytest --cov

test_verbose: ## Test the code with pytest and coverage in verbose mode
	@echo "Testing code: Running pytest in verbose mode"
	@uv run pytest --no-header -v --cov

test_coverage: ## Test coverage report coverage.xml
	@echo "Testing code: Running pytest with coverage"
	@uv run pytest --cov --cov-report xml:coverage.xml

####----Pre-commit----####
pre-commit_update: ## Update pre-commit hooks
	@echo "Updating pre-commit hooks..."
	uv run pre-commit clean
	uv run pre-commit autoupdate

#
####----Docs----####
docs: ## Build and serve the documentation site locally (Zensical)
	@echo "Serving the documentation at http://127.0.0.1:8000 ..."
	uv run zensical serve

docs_test: ## Build the documentation site in strict mode (fails on warnings)
	@uv run zensical build --strict --clean
#

####----Clean----####
clean_env: ## Clean .venv virtual environment
	@echo "Cleaning the environment..."
	@[ -d .venv ] && rm -rf .venv || echo ".venv directory does not exist"

####----Git----####
switch_main: ## Switch to main branch and pull
	@echo "Switching to main branch..."
	@git switch main
	@git pull

clean_branchs: ## Clean local branches already merged on the remote
	@echo "Cleaning up merged branches..."
	@git fetch -p
	@for branch in $$(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v '^\*' | grep -v ' main$$'); do \
		if ! git show-ref --quiet refs/remotes/origin/$$branch; then \
			if git config --get branch.$$branch.remote > /dev/null 2>&1; then \
				echo "Deleting local branch $$branch"; \
				git branch -D $$branch; \
			fi \
		fi \
	done

####----Checks----####
check: ## Run all pre-commit hooks on all files (the same gate as CI)
	@echo "Linting, formating and Static type checking code: Running pre-commit"
	@uv run pre-commit run -a

lint: ## Run ruff only (lint and format) on all files
	@echo "Linting and formatting code: Running ruff"
	@uv run pre-commit run ruff-check -a
	@uv run pre-commit run ruff-format -a

####----Project----####
help:
	@printf "%-30s %s\n" "Target" "Description"
	@printf "%-30s %s\n" "-----------------------" "----------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
