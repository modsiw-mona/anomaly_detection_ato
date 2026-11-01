# Development plan — ATO detection POC

> Status: **approved v2** · 2026-10-31
> Owner: Sofía Arango Patiño · Implementer: Claude Code (auto mode)

This plan makes the routine decisions in advance so development can run end to end without
stopping. Section 6 lists the few points where it *must* stop and ask. Section 9 is a glossary.

## 1. Goal

Deliver the 8 POC issues (#1–#8) as executed notebooks plus reusable, tested code in `src/`. The
work must satisfy both the course POC checklist and thesis objectives OE1–OE4, use the RBA
dataset, and keep a dataset-agnostic core. Everything runs **locally on a 16 GB M1 Pro**, with no
cloud services.

| Issue | Course step | CRISP-DM phase (anteproyecto 6.3) | Thesis objective |
| --- | --- | --- | --- |
| #1 | Descarga de los datos | F1 Comprensión del problema | OE1 |
| #2 | Exploración inicial | F2 Comprensión de datos | OE1 |
| #3 | EDA | F2 Comprensión de datos | OE1 (+ OE2 hypotheses) |
| #4 | Feature Engineering | F3 Preparación de datos | OE2 |
| #5 | Modelo BaseLine | F4 Modelado | OE3 |
| #6 | Selección del mejor modelo | F4 Modelado y selección | OE3 |
| #7 | Interpretación | F5 Evaluación e interpretación | OE4 |
| #8 | Demo POC | F6 Prototipo | OE4 |

## 2. Guiding principles

1. **Test, don't assume.** No variable is discarded without an empirical test (an ablation: train
   with and without it, compare). The dataset README tells us where to be sceptical and what to
   document, not what to delete.
2. **Current and supported only.** Latest stable Python and actively maintained libraries; no
   deprecated APIs (section 3).
3. **Local-first.** Develop on a sample, and run on the full data only where it fits in 16 GB
   (section 4.3).
4. **Reproducible.** Scripted download with checksum, pinned lockfile, seeded randomness, executed
   notebooks.
5. **No secrets, ever** (section 7).

## 3. Stack — current versions, verified on PyPI on 2026-10-31

- **Python 3.14** (3.14.7), the newest stable release (3.15 is only a release candidate).
  `.python-version`, ruff `target-version` and the CI interpreter all move from 3.12/py310 to 3.14.
- Every library is resolved to its latest release by `uv`, all of which declare Python 3.14 support:
  polars 2.1, pandas 3.1, numpy 2.5, scikit-learn 1.9, xgboost 3.4, lightgbm 4.7, pyod 3.6,
  numba 0.68, pandera 0.35, optuna 5.0, shap 0.54, pydantic 2.14, streamlit 1.66, jupyterlab 4.6.
- **Maintenance rule:** a library must be *actively maintained* — not archived, with commits or
  releases in the last 12 months — and support Python 3.14. Mature libraries that release rarely
  still qualify (PyYAML: last release 2025-09, commits through 2026-10). This rule rejected
  `detect-secrets` (no release since 2024) in favour of `gitleaks`.
- **Configuration format:** YAML for project and pipeline configuration (`conf/`), as in the
  course template and the ML ecosystem (Hydra, DVC, Kedro); TOML only for packaging and tool
  settings (`pyproject.toml`, `ruff.toml`). YAML files are read with `yaml.safe_load` and validated
  with pydantic; strings are quoted to avoid YAML 1.1 implicit booleans (`yes`, `no`, `on`, `off`).
- **Hydra** (`hydra-core` 1.3.8) is maintained, but its metadata only declares Python ≤ 3.11. It
  is used only if it runs cleanly on 3.14. Fallback: plain YAML files in `conf/`, validated with
  pydantic.
- **TabPFN** (a tabular foundation model) is an *optional* candidate, used only if its model
  weights are distributed in a safe format (safetensors, not pickle, which can execute code when
  loaded) and its licence allows academic use.

## 4. Data

### 4.1 Source and safety

- **Download from Zenodo**: `10.5281/zenodo.6782156`, v1.0.0, `rba-dataset.zip` (1.094 GB,
  `md5:cc1b1078b3929650e6c08678caffcc57`), CC BY 4.0. The Kaggle copy (`dasgroup/rba-dataset`) is
  referenced in the dataset card as a mirror.
  - Why Zenodo: a DOI-versioned record whose files cannot be silently replaced, a published
    checksum, no account and no API token needed (Kaggle requires a credential file), and the
    citation thesis committees expect.
- **A download script in `src/data/`** fetches the file, verifies the checksum, inspects the
  zip's file list (names and sizes) before extracting, and extracts only the expected CSV into
  `data/01_raw/`. Data files are never executed.
- **No third-party code is cloned, installed or executed.** The reference implementation
  (`das-group/rba-algorithm`, MIT) is read as text only, and its logic is re-implemented here.
  Python packages come only from PyPI via `uv`, pinned with hashes in `uv.lock`.
- Cite Wiefling et al. (2022), doi `10.1145/3546069`, as the licence requires.

### 4.2 Dataset-agnostic core

- **Standard column names ("canonical schema")** in `src/data/`: `user_id`, `timestamp`, `ip`,
  `asn`, `country`, `region`, `city`, `user_agent`, `os`, `browser`, `device_type`, `rtt_ms`,
  `login_success`, plus label columns. Feature and model code use only these names.
- **One translation file per dataset**: `conf/dataset/rba.yaml` maps the dataset's own column
  names to the standard ones, and records labels, data types, source, licence, citation and known
  artefacts.
- **A dataset card**, `docs/datasets/rba.md`: provenance, how the data was synthesised, what the
  labels mean, how faithful each signal is, and limitations.
- **Validation rules** (Pandera) generated from the translation file.
- **Minimum data requirements** for any dataset: `user_id`, `timestamp`, `login_success`, plus at
  least one context field among `ip`, `asn`, `device_type`, `user_agent`. Without `user_id` there
  is no per-user baseline, which is the core of the approach.

### 4.3 Running locally on 16 GB RAM

- The full table (31.3M rows) is processed with **polars** (lazy / streaming), stored as parquet
  with dictionary-encoded strings, and never loaded whole into pandas.
- **Development sample, drawn by user, not by row.** The sample contains every user with at least
  one ATO, plus a random 10% of all other users (seeded), with **each sampled user's complete
  history**. Sampling random rows would cut holes in users' histories and corrupt every
  "is this new for this user" feature.
- **Global counts come from the full data** ("how common is this ASN across all users"). These are
  cheap group-by counts, so they stay correct even when modelling on the sample.
- Because the sample over-represents ATO users, metrics are computed with **sample weights**
  (inverse sampling probability) so they reflect the real population.
- If the full run fits in memory once the pipeline is stable, the final numbers are also reported
  on the full data.

### 4.4 What the dataset authors say about signal fidelity

Source: the dataset's own README ("Data Creation" section). These are the authors' statements, not
our assumptions:

| Signal | Authors' statement | How we treat it |
| --- | --- | --- |
| Device type | Identical to the real data | Use |
| User agent, OS, browser | Logical relations identical; values regenerated | Use |
| IP, ASN, country | Per-user and global frequencies identical; values synthetic | Use via behaviour features (4.5) |
| City, region | Derived from the synthetic IPs; "do not reflect the real logical relations" | **Test** with an ablation |
| Timestamps | Order preserved, randomness added | Use; temporal features tested |
| RTT | Realistic values, reassigned per geolocation and login status | **Test** with an ablation |
| ASN ≥ 500000, IP in 10.0.0.0/8 | Placeholders from synthesis conflicts | Flag, quantify, sensitivity check |

### 4.5 Feature rules

1. **No peeking at the future.** A feature for a login at time *t* uses only events before *t*.
   History features are computed on each user's complete timeline *before* sampling or splitting.
   Unit tests on small hand-built timelines check this.
2. **Behaviour first.** The primary features describe *behaviour*: is this value new for this user,
   how often has the user used it, how rare is it globally, how many accounts share it, time since
   the last login, failed attempts just before a success. These keep working when values are
   scrambled or hashed, whether by this dataset's synthesis or by a real company's
   pseudonymisation.
3. **Identity-based encodings get tested too.** Encoding *which* ASN or country a login comes from
   is not ruled out; it is run as an ablation, and kept only if it helps on validation.

## 5. Modelling and evaluation

### 5.1 Target

- **Primary target:** `Is Account Takeover`, on successful logins (the only population where it is
  defined). Failed logins feed features, e.g. failed attempts on the account just before a success,
  which is the signature of credential stuffing.
- **`Is Attack IP` is never the target.** It is used for EDA, and for one sensitivity variant that
  removes "attack IP + successful + not flagged as ATO" rows (likely missed takeovers) from the
  clean negatives.

### 5.2 Splitting

- **Final test set ("final exam")**, decided by a fixed rule in #3, once we can see where the 140
  ATOs fall in time:
  - If the last ~20% of the timeline contains at least 20 ATOs → **time-based split**: train on
    the earlier months, test on the later ones. This is the preferred option because it mimics
    real use.
  - Otherwise → **random 20% split** that keeps the ATO share balanced between both sides.
  - Either way, **all logins of one user land on the same side**, so the model cannot memorise a
    user's habits from training and get an easy test.
  - The test set is used **once**, in #7.
- **Validation inside the training data:** repeated, user-grouped, stratified k-fold
  (5 folds × at least 3 repeats), reporting mean ± std and bootstrap 95% confidence intervals.

### 5.3 Metrics

PR-AUC (primary); Recall at alert budgets (top 0.1% and 1% of logins); Precision, reported as a
**lower bound** (PU view: unlabelled ≠ negative); ROC-AUC (secondary); **friction**, the share of
legitimate logins challenged at the chosen threshold.

### 5.4 Model families (OE3)

| Family | Candidates |
| --- | --- |
| Baseline | Dummy; **Freeman et al. (2016) RBA score**, re-implemented and vectorised |
| Unsupervised | Isolation Forest, ECOD (PyOD), others from PyOD as time allows |
| Supervised | Logistic regression, Random Forest, XGBoost, LightGBM — class weights vs undersampling vs SMOTE, compared |
| Hybrid / weakly supervised | Anomaly scores as features for supervised models |
| PU | PU bagging over a tree ensemble |
| Optional | TabPFN (only under the conditions in section 3) |

The course #6 minimum of 4 model types is covered. The Freeman re-implementation is validated
against the expected scores published in the reference notebook. A match is strong evidence it is
correct; a mismatch is investigated and documented.

### 5.5 Methods updated since the anteproyecto's sources

| Topic | Current practice | Source |
| --- | --- | --- |
| Class imbalance | Class weights + threshold tuning tend to beat SMOTE for strong classifiers, and oversampling can harm probability calibration. SMOTE is still tested, since the anteproyecto mentions it. | Elor & Averbuch-Elor (2022); van den Goorbergh et al. (2022, JAMIA) |
| Threshold choice | `TunedThresholdClassifierCV` (scikit-learn ≥ 1.5) | scikit-learn |
| Controlling false alarms | Conformal p-values give a statistical guarantee on the false-alarm rate among legitimate logins, e.g. "challenge at most 1%" | Bates et al. (2023, Annals of Statistics) |
| Few labels | A handful of labelled anomalies substantially improves on fully unsupervised detection, which supports the hybrid family | Han et al. (2022), ADBench, NeurIPS |
| Unlabelled ≠ negative | PU learning | Bekker & Davis (2020) |
| Small tabular data | Tabular foundation models | Hollmann et al. (2025), TabPFN, Nature |
| Hyper-parameters | Optuna 5 (TPE sampler, pruning) | Optuna |
| Explanations | SHAP (TreeSHAP) | Lundberg & Lee (2017) |

## 6. Stop-and-ask points

Claude stops and asks **only** when:

1. The data contradicts a thesis assumption materially, e.g. the ATOs are so concentrated that no
   valid split exists, or the labels turn out not to sit on successful logins.
2. A change would alter the **wording** of the objectives or methodology. Changing techniques is fine.
3. Before touching the final test set (#7).
4. Anything needing new accounts, credentials, paid services, or repo/security-settings changes.
5. Results look too good (PR-AUC > 0.95, or near-perfect recall): suspect leakage and investigate
   before reporting.

Everything else is decided by this plan and reported at the end of each issue.

## 7. Secrets — none, checked at five layers

1. **None needed.** The project uses no API keys, tokens or accounts (Zenodo needs none).
2. **Never written down.** No secret goes into code, notebooks, notebook outputs, commit messages
   or PR text. `.gitignore` also covers `.env*`, `*.pem`, `*.key`, `kaggle.json`, `credentials*`
   and `secrets*`. Notebooks use repository-relative paths, so outputs don't expose local folders.
3. **Scanned locally before every commit:** a `gitleaks` pre-commit hook (plus the existing
   `detect-private-key`).
4. **Scanned in CI** on every PR, because CI runs the same pre-commit hooks.
5. **Blocked at push by GitHub:** secret scanning and push protection are enabled on the repository.

Personal documents stay out of the repository. The anteproyecto PDF contains identity numbers and
must never be copied into it.

## 8. Engineering workflow

- One branch per issue via `gh issue develop <n>`; conventional commits; PR body `Closes #n`.
- `make check` and `make test` before every push. Claude opens the PR, waits for CI to go green,
  and reports. **Sofía merges, with "Create a merge commit"** (GitFlow style, keeps each conventional
  commit on `main`) — Claude never merges. The `protect-main` ruleset enforces PR +
  green `test`, `pre-commit` and `actionlint`, with 0 approvals.
- While a PR waits for merge, Claude continues the next issue on a branch **stacked** on it (PR
  base = the previous branch). Merge commits keep the stacked branch's commits identical to those on
  `main`, so no rebuild is needed: once the earlier PR is merged, GitHub retargets the next PR to
  `main` and Claude merges `main` into it if required (never a rebase or force-push).
- **Everything in English:** notebook text, code, docs, commits.
- Notebooks start from `notebooks/notebook_template.ipynb`, are executed headless
  (`jupyter nbconvert --execute`), and are committed with outputs so results are visible on GitHub.
  The data is public and synthetic.
- Reusable logic moves into `src/` with pytest tests, mypy-strict types and ruff, which prepares
  course part 2 (the FTI pipeline scripts).
- Dependencies are added with `uv add` inside the PR that first needs them.
- **Dependency hygiene** (where the real supply-chain risk is: packages run on this Mac): only
  well-known, maintained packages; exact versions and hashes in `uv.lock`; Dependabot alerts on;
  never `pip install` from anywhere other than PyPI via `uv`.
- *Optional, low priority:* replace CI's `bash <(curl …/actionlint/main/…)` step with a pinned
  install. Its risk is small (it runs on GitHub's servers with a read-only token), but it also
  breaks CI whenever upstream changes.

## 9. Sequence

| Step | Deliverables |
| --- | --- |
| 0 · setup | This plan; `CLAUDE.md` update; Python 3.14 upgrade; gitleaks hook; `.gitignore` hardening |
| #1 | Scripted, verified download; dataset card; translation file; canonical schema; data requirements; notebook `1-data` answering the 14 problem-definition questions |
| #2 | Load into the canonical schema, unify nulls, fix data types, flag artefacts → parquet in `data/02_intermediate/`; notebook `2-exploration` |
| #3 | Univariate / bivariate / multivariate EDA; deep dive on the 140 ATOs (timeline, users, failures before success, attack-IP overlap); artefact, geography and RTT checks; validation rules; split decision; sample definition; notebook `3-analysis` |
| #4 | Behaviour features (polars, no future peeking) + scikit-learn preprocessing pipeline → `data/04_feature/`; notebook `4-feat_eng` |
| #5 | Dummy + Freeman baseline, validation protocol, CV, learning curves; notebook `5-models` |
| #6 | Model-family comparison, ablations (geography, RTT, identity encodings, artefacts), tuning of the top 2–3, best pipeline → `.joblib`; notebook `5-models` |
| #7 | Final test evaluation, thresholds and alert budgets, conformal false-alarm control, SHAP, error analysis, FP/FN costs, friction; notebook `6-interpretation` |
| #8 | Streamlit demo: score one login → risk score, tier (allow / step-up / block), top signals; screenshots + run instructions |

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| 16 GB RAM vs 31.3M rows | polars streaming, categoricals, column pruning, cached parquet stages, user-based sample |
| Freeman score cost | vectorised cumulative counts instead of per-row loops |
| 140 positives → high-variance metrics | repeated grouped CV, bootstrap CIs, alert-budget recall |
| Synthetic artefacts | behaviour-first features, flags, ablations |
| Future information leaking into features | strict "past only" computation, unit-tested |
| A library failing on Python 3.14 | caught at `uv lock`; fall back to 3.13 for that library only if unavoidable, documented |

## 11. Glossary

- **ATO (account takeover):** an attacker logging in to someone else's account.
- **RTT (round-trip time):** milliseconds for a signal to go from the user's device to the server
  and back, like an echo. A login claiming to be in Medellín with a very slow echo may be routed
  through a distant proxy or VPN.
- **ASN (autonomous system number):** the ID of the network an IP belongs to (an ISP, a cloud
  provider, a university).
- **Holdout / test set:** data hidden until the end and used once, like a final exam.
- **Ablation:** removing one ingredient (a feature group) and measuring whether results get worse.
- **Data leakage:** the model accidentally seeing information it wouldn't have in real life (e.g.
  the future), which makes test results look better than reality.
- **PU learning:** learning from Positives plus Unlabelled data, where "unlabelled" may hide
  positives nobody found.
- **Conformal p-value:** a score with a statistical guarantee, e.g. "at most 1% of legitimate
  logins will be flagged".
