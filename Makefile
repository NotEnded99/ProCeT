.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
PYTHON=$(ENV_PREFIX)python
PIP=$(ENV_PREFIX)pip

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(ENV_PREFIX)"
	@$(PYTHON) -V
	@$(PYTHON) -m site

.PHONY: install
install:          ## Install the project in dev mode.
	@echo "Don't forget to run 'make condaenv' if you got errors."
	$(PIP) install -e .[test]

.PHONY: fmt
fmt:              ## Format code using black & isort.
	$(PIP) install isort black
	$(PYTHON) -m isort lbp-neural-cbf/
	$(PYTHON) -m black -l 160 lbp-neural-cbf/
	$(PYTHON) -m black -l 200 tests/

.PHONY: lint
lint:             ## Run pep8, black, mypy linters.
	$(PYTHON) -m flake8 lbp-neural-cbf/
	$(PYTHON) -m black -l 160 --check lbp-neural-cbf/
	$(PYTHON) -m black -l 200 --check tests/

.PHONY: test-all
test-all:         ## Run all tests including slow dynamics models.
	$(PYTHON) -m pytest -v --cov-config .coveragerc --cov=lbp-neural-cbf -l --tb=short --maxfail=1 tests/
	$(PYTHON) -m coverage xml
	$(PYTHON) -m coverage html

.PHONY: test
test: lint test-all ## Run linting and fast tests (default for CI).

.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | entr $(PYTHON) -m pytest -s -vvv -l --tb=long --maxfail=1 tests/

.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: condaenv
condaenv:       ## Create a conda environment.
	@echo "creating conda environment ..."
	@conda env create --yes --f environment.yml

.PHONY: release
release:          ## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@echo "$${TAG}" > lbp-neural-cbf/VERSION
	@$(PYTHON) -m gitchangelog > HISTORY.md
	@git add lbp-neural-cbf/VERSION HISTORY.md
	@git commit -m "release: version $${TAG} 🚀"
	@echo "creating git tag : $${TAG}"
	@git tag $${TAG}
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."

.PHONY: docs
docs:             ## Build the documentation.
	@echo "building documentation ..."
	@$(PYTHON) -m mkdocs build
	URL="site/index.html"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL || open $$URL
