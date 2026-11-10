# Reproduce

Everything runs on a laptop (developed on a 16 GB laptop), with no cloud services and no credentials. The
dataset is downloaded from its official source and verified; nothing is uploaded anywhere.

## Set up

1. Install [uv](https://docs.astral.sh/uv/) and clone the repository:

    ```bash
    git clone https://github.com/modsiw-mona/anomaly_detection_ato.git
    cd anomaly_detection_ato
    ```

1. **macOS only:** LightGBM and XGBoost need the OpenMP runtime:

    ```bash
    brew install libomp
    ```

1. Create the environment (Python 3.14, exact versions from `uv.lock`) and the pre-commit hooks:

    ```bash
    make install_env
    ```

## Build the data

The **feature pipeline** runs the data steps in order and validates each output before the next step uses it.
Everything is written to `data/` (gitignored):

```bash
make feature_pipeline                  # all the steps below, with the checks
make feature_pipeline VALIDATE_ONLY=1  # only check the existing outputs (about 30 seconds)
```

| Step | Output | Checked before going on |
| --- | --- | --- |
| Download from Zenodo (9 GB unpacked) | `data/01_raw/` | the published checksum; safe extraction |
| Standard names, types, nulls, flags | `data/02_intermediate/` | types, required fields, unique rows, the dataset's value rules |
| Past-only behavioural features | `data/04_feature/` | types and ranges, nulls only on first logins, the label window, no catch-all account, the relations that follow from each definition |
| Development / test split by user | `data/05_model_input/` | every user on exactly one side, the pre-registered test share |
| Development against test | report | each feature's drift (Kolmogorov-Smirnov or total variation) below 0.2 |

A rebuilt output replaces the previous one only after it passes its checks, so a broken rule stops the pipeline
and leaves the earlier outputs in place; the log counts the failing rows of each rule. Each run writes a report
(row counts, warnings, the drift table, timings, and on failure the step and the error) to
`data/08_reporting/rba_feature_pipeline.json`. The steps can also run one at a time,
without the checks: `make data`, `make intermediate`, `make features`, `make split`.

## Train, evaluate and run the demo

```bash
make notebooks_run NB=notebooks/5-models/06-sap-model_selection-2026_11_03.ipynb          # about 70 minutes
make notebooks_run NB=notebooks/6-interpretation/07-sap-model_interpretation-2026_11_04.ipynb
make notebooks_run NB=notebooks/6-interpretation/08-sap-user_friction-2026_11_08.ipynb
make demo
```

## Checks

```bash
make test    # pytest with coverage, on toy data (no download needed)
make check   # ruff, mypy, secret scanning and file checks (commit messages are checked when committing)
```

Another login dataset can be used by writing its column mapping in `conf/dataset/<name>.yaml`
([how](datasets/README.md)) and passing `DATASET=<name>` to the commands above.
