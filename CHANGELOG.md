# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The entries up to the sign-in demo were written after the fact from the merged pull requests; each line links
to its pull request.

## [Unreleased]

### Added

- Data download from Zenodo with checksum verification and safe extraction, a standard field schema, minimum data
  requirements and a validated translation file per dataset (`conf/dataset/<name>.yaml`), so another dataset needs
  only a new file. Problem definition notebook. ([#18](https://github.com/modsiw-mona/anomaly_detection_ato/pull/18))
- Typed intermediate dataset (`make intermediate`): standard names, per-column null markers, strict types,
  artefact flags and an `event_id` per row. Initial exploration notebook.
  ([#19](https://github.com/modsiw-mona/anomaly_detection_ato/pull/19))
- Exploratory data analysis and Pandera validation rules for the intermediate dataset; dataset card.
  ([#20](https://github.com/modsiw-mona/anomaly_detection_ato/pull/20))
- Past-only behavioural features (`make features`), a seeded development/test split by user (`make split`), the
  label window (February to November 2020) and the preprocessing pipeline. Every feature is checked against a
  recomputation that only sees earlier events. ([#21](https://github.com/modsiw-mona/anomaly_detection_ato/pull/21))
- Evaluation protocol (5 folds x 3 repeats, grouped by user; PR-AUC, recall, precision and friction at 0.1% and
  1% alert budgets; bootstrap intervals) and baselines: rules, Freeman et al.'s RBA score and logistic regression.
  ([#23](https://github.com/modsiw-mona/anomaly_detection_ato/pull/23))
- Model selection on the development side with campaign-aware evaluation, the `campaign` and `travel` feature
  groups, and the selected model (random forest) saved with its metadata.
  ([#24](https://github.com/modsiw-mona/anomaly_detection_ato/pull/24))
- Alert thresholds from the out-of-fold development scores, split-conformal thresholds, the final evaluation on the
  test side, leakage checks, error costs and SHAP interpretation.
  ([#25](https://github.com/modsiw-mona/anomaly_detection_ato/pull/25))
- Inference module (`Detector`: score, response tier, rank and SHAP explanation) and a local Streamlit demo.
  ([#26](https://github.com/modsiw-mona/anomaly_detection_ato/pull/26))
- Friction per user (share of each legitimate user's logins that are challenged), compared with Freeman's score,
  and the check of the one-hour travel window. ([#33](https://github.com/modsiw-mona/anomaly_detection_ato/pull/33))
- Documentation site built with Zensical and published on GitHub Pages.
  ([#37](https://github.com/modsiw-mona/anomaly_detection_ato/pull/37))
- Public demo on Streamlit Community Cloud, which reads a small bundle (example logins, reference sample,
  thresholds and the model) attached to a GitHub Release.
  ([#41](https://github.com/modsiw-mona/anomaly_detection_ato/pull/41))
- The demo as a simulated sign-in page: the response the person would see (allow, second factor with its whole
  flow, block), feedback at every step and a guided tour.
  ([#46](https://github.com/modsiw-mona/anomaly_detection_ato/pull/46))
- A changelog entry in the pull request checklist.
- Feature pipeline (`make feature_pipeline`): download, intermediate dataset, features and split in order,
  validating each output with Pandera before the next step (new schemas for the feature table and the split),
  a drift check between the development and test sides, and a report per run.
  ([#51](https://github.com/modsiw-mona/anomaly_detection_ato/pull/51))
- Training pipeline (`make training_pipeline`): trains the selected configuration (`conf/training/rba.yaml`)
  on the development side, sets the alert thresholds, checks the model against validation gates before saving
  it, and writes a record of every run. The tuned forest moves from notebook 06 to `src/model/selection.py`.
  ([#53](https://github.com/modsiw-mona/anomaly_detection_ato/pull/53))

### Changed

- Python 3.14 and the latest release of every dependency.
  ([#16](https://github.com/modsiw-mona/anomaly_detection_ato/pull/16))
- The pre-commit auto-update runs on demand instead of every week.
  ([#14](https://github.com/modsiw-mona/anomaly_detection_ato/pull/14))
- The documentation workflow builds the site on pull requests without publishing it.
  ([#22](https://github.com/modsiw-mona/anomaly_detection_ato/pull/22))
- MkDocs and Material for MkDocs replaced by Zensical.
  ([#37](https://github.com/modsiw-mona/anomaly_detection_ato/pull/37))
- The `actionlint` job downloads a pinned release and checks its SHA-256, instead of running a script from the
  tool's default branch. ([#40](https://github.com/modsiw-mona/anomaly_detection_ato/pull/40))
- mypy runs in full strict mode. ([#45](https://github.com/modsiw-mona/anomaly_detection_ato/pull/45))

### Removed

- Internal planning notes and emojis. ([#28](https://github.com/modsiw-mona/anomaly_detection_ato/pull/28))
- Unused template leftovers (the empty Hydra configuration scaffold, the empty `models/` folder, Makefile targets)
  and dependencies nothing imported. ([#36](https://github.com/modsiw-mona/anomaly_detection_ato/pull/36))

### Fixed

- Networks never seen before were imputed as clean in preprocessing. Found in review; the model was selected again
  after the fix, and the earlier test run with the affected model is disclosed in the evaluation.
  ([#24](https://github.com/modsiw-mona/anomaly_detection_ato/pull/24))
- The public demo retries the bundle download on temporary server errors.
  ([#43](https://github.com/modsiw-mona/anomaly_detection_ato/pull/43))
- A test that could never fail, found by mypy strict mode.
  ([#45](https://github.com/modsiw-mona/anomaly_detection_ato/pull/45))

### Security

- Secret scanning with gitleaks before every commit and in CI; credential files ignored by git.
  ([#16](https://github.com/modsiw-mona/anomaly_detection_ato/pull/16))
- The demo checks every downloaded file against the SHA-256 committed in the repository before loading it, and
  never loads a model file that does not match. ([#41](https://github.com/modsiw-mona/anomaly_detection_ato/pull/41))
- Dependencies kept up to date by Dependabot, with a dependency review on pull requests.
