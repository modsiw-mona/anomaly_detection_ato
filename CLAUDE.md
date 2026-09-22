# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Master's thesis: **Account Take Over (ATO) detection in authentication records using anomaly
detection methodologies**.

This is a real data science project, scaffolded from
[JoseRZapata/data-science-project-template](https://github.com/JoseRZapata/data-science-project-template)
via `cruft`. It is NOT the template itself — do not edit it as if it were. `.cruft.json` records
the template commit; run `cruft check` / `cruft update` to pull upstream template improvements.

It follows the [Ciencia de Datos en Producción](https://joserzapata.github.io/courses/ciencia-datos-en-produccion/)
methodology: POC in notebooks first, then the same logic promoted into FTI pipeline scripts.

## Mandatory workflow

Every unit of work follows this, without exception:

1. Work is tracked as a **GitHub issue** (the 8 POC steps are issues #1–#8).
2. Create a branch **from the issue** (gitflow naming, e.g. `feat/1-descarga-de-datos`).
3. Commit with **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`…) — enforced by the
   commitizen pre-commit hook; a non-conforming message is rejected.
4. Open a **Pull Request into `main`**.
5. CI must pass, then merge.

Never commit directly to `main`.

## Commands

```bash
make install_env        # uv sync --all-groups + pre-commit install
make check              # run ALL pre-commit hooks (same gate as CI) — run before every PR
make lint               # ruff only
make test               # pytest + coverage
make test_coverage      # write coverage.xml
make docs               # serve mkdocs locally
uv add <pkg>            # add a runtime dep
uv add <pkg> --group dev  # add a dev dep
uv run pytest tests/test_x.py::test_y -v   # single test
```

Always use `uv run` / `uv add` — never bare `pip` or `python`.

## Architecture

Feature/Training/Inference (FTI) pipeline pattern:

```
src/
├── data/                    # extraction, validation, processing
├── model/                   # training, evaluation, validation, export
├── inference/               # prediction, serving, monitoring
└── pipelines/
    ├── feature_pipeline/    # raw data -> features & labels
    ├── training_pipeline/   # features & labels -> model
    └── inference_pipeline/  # features & model -> predictions
```

`conf/` holds Hydra configuration (`conf/config.yml` plus a folder per pipeline stage).

## Data conventions

8-stage layout (Kedro convention). **`data/` contents are gitignored**; only the folder structure
and `data/README.md` are tracked. Never commit datasets or model binaries.

`01_raw` → `02_intermediate` → `03_primary` → `04_feature` → `05_model_input` → `06_models`
→ `07_model_output` → `08_reporting`

Notebook folders mirror the stages, and map to the 8 issues:

| Issue | Step | Notebook folder |
| --- | --- | --- |
| #1 | Descarga de los datos | `notebooks/1-data/` |
| #2 | Exploración inicial de datos | `notebooks/2-exploration/` |
| #3 | Análisis exploratorio (EDA) | `notebooks/3-analysis/` |
| #4 | Feature Engineering | `notebooks/4-feat_eng/` |
| #5 | Modelo BaseLine | `notebooks/5-models/` |
| #6 | Selección del mejor modelo | `notebooks/6-interpretation/` |
| #7 | Interpretación del modelo | `notebooks/6-interpretation/` |
| #8 | Demo POC | `notebooks/7-deploy/` |

Start new notebooks from `notebooks/notebook_template.ipynb`.

Intermediate datasets are persisted as `.parquet`; the fitted pipeline + model as `.joblib`.

## Code quality

Configs live in `.code_quality/`:

- **ruff** (`ruff.toml`): line-length 100, target py3.10, notebooks included.
  Rules: B, C90, E, F, W, PL, I, S, UP, RUF, SIM, TRY
- **mypy** (`mypy.ini`): strict — `disallow_untyped_defs`, `disallow_untyped_calls`,
  `check_untyped_defs`

Pre-commit also enforces YAML validity, merge-conflict and large-file checks, private-key
detection, and conventional commits.

## Environment

- Python **3.12** (pinned in `.python-version`)
- **uv** for all dependency and venv management
