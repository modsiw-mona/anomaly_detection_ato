# Reproduce

Everything runs on a laptop (developed on a 16 GB Apple M1 Pro), with no cloud services and no credentials. The
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

Each step is idempotent and writes to `data/` (gitignored):

```bash
make data           # download from Zenodo (9 GB unpacked), verify the published checksum -> data/01_raw/
make intermediate   # standard names, types, nulls, flags                       -> data/02_intermediate/
make features       # past-only behavioural features                            -> data/04_feature/
make split          # development / test split by user                          -> data/05_model_input/
```

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
make check   # every pre-commit hook: ruff, mypy, secret scanning, conventional commits
```

Another login dataset can be used by writing its column mapping in `conf/dataset/<name>.yaml`
([how](datasets/README.md)) and passing `DATASET=<name>` to the commands above.
