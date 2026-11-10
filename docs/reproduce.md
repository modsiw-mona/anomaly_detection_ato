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

## Train, score and run the demo

```bash
make training_pipeline      # about 6 minutes: the selected model, validated, with its thresholds
make inference_pipeline     # about 30 seconds: scores and response tiers of the test side
make demo
```

`make pipelines` runs the feature, training and inference pipelines in order. The notebooks are the record of the
proof of concept; the final evaluation and the friction analysis read what the pipelines save:

```bash
make notebooks_run NB=notebooks/6-interpretation/07-sap-model_interpretation-2026_11_04.ipynb
make notebooks_run NB=notebooks/6-interpretation/08-sap-user_friction-2026_11_08.ipynb
```

The **training pipeline** trains only the selected configuration, written in `conf/training/rba.yaml` (model,
features, sampling, seeds, folds, alert budgets and the limits below). Comparing model families is
experimentation and stays in notebook `5-models/06` (about 70 minutes), which chose that configuration. The
pipeline reads only the development side:

1. Out-of-fold scores on the same user folds as the notebook: 5 folds x 3 repeats, the 15-fold estimate.
2. The logistic baseline on the same folds.
3. The final fit on the whole development side.
4. The alert thresholds, from the out-of-fold scores of the first repeat.
5. The **model validation gates**. A model that fails a blocking gate is not saved, and the previous one stays:

| Gate | Condition | Blocking |
| --- | --- | --- |
| Selection rule of the model selection | lower end of the PR-AUC interval > 0.05; recall within the 1% budget >= 0.80 | yes |
| Gain over the logistic baseline | PR-AUC at least 10% higher, on the same folds | yes |
| Alert rate of the saved model | within 0.5 to 2 times each budget, on development logins | yes |
| Each group of takeovers (the dominant campaign, the rest, first logins, returning accounts) | recall within the 1% budget >= 0.5 | yes |
| Scoring time | at most 1 ms per login when scoring many at once (the course's condition); at most 0.2 s for a single login on its own, what a sign-in waits for | yes |
| Weakest group | PR-AUC at least 80% of the overall one (the course's condition) | reported: takeovers outside the dominant campaign are much harder |
| Overfitting gap, calibration (Brier score) | | reported |

It saves the model, its metadata and the out-of-fold scores where the notebooks saved them, so notebooks 07 and
08 and the demo read them unchanged. The three files are written first and renamed only then, the metadata last,
and the inference module refuses a model file whose SHA-256 differs from its metadata, so a model never runs with
another one's thresholds. `REPEATS=1` gives a quicker 5-fold run; notebooks 07 and 08 need the default 15 folds. Every run writes a record to `data/08_reporting/training_runs/`: the
configuration, metrics, gates, the SHA-256 of the data and of the model, the git commit, library versions, and
whether the model is identical to the previous one.

The **inference pipeline** scores a batch of logins with the saved model, the way a security team would score
the previous day's logins (the demo scores one login at a time with the same module). It loads the model only if
its SHA-256 matches the metadata, validates the batch with the feature table's schema, compares it with the
development side (drift is reported, not blocking: a service keeps answering), and writes every login's score
and response tier to `data/07_model_output/` with a report in `data/08_reporting/rba_inference_pipeline.json`. It
never uses the labels: measuring the model on the test side is the final evaluation's job, done once. On the
test side it flags 0.094% of logins for review or block and 0.95% for a second factor or stricter, against
budgets of 0.1% and 1%.

On the RBA data the training pipeline rebuilds the notebook's model **byte for byte** (same SHA-256), the 15-fold PR-AUC
(0.4475) and the thresholds. The forest averages its trees in parallel, so a few scores may differ in the last
bit (about 1e-16) from run to run; no threshold or response tier changes. The same holds for the inference
pipeline: its test-side scores match notebook 07's to 1e-16, with no tier changed.

## Checks

```bash
make test    # pytest with coverage, on toy data (no download needed)
make check   # ruff, mypy, secret scanning and file checks (commit messages are checked when committing)
```

Another login dataset can be used by writing its column mapping in `conf/dataset/<name>.yaml`
([how](datasets/README.md)) and passing `DATASET=<name>` to the commands above.
