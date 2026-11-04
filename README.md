# Anomaly Detection ATO

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Repository for master's thesis project on **Account Take Over (ATO) detection** in authentication records using anomaly detection methodologies.

Scaffolded from [JoseRZapata/data-science-project-template](https://github.com/JoseRZapata/data-science-project-template) via [cruft](https://cruft.github.io/cruft/), following the [Ciencia de Datos en Producción](https://joserzapata.github.io/courses/ciencia-datos-en-produccion/) methodology.

Using the data science project template <https://github.com/JoseRZapata/data-science-project-template>

## Features and Tools

Information about all the features and tools used in this project: <https://joserzapata.github.io/data-science-project-template/#features-and-tools>

Features                                     | Package  | Why?
 ---                                         | ---      | ---
Dependencies and env                         | [UV] | [article](https://astral.sh/blog/uv)
Lint - Format, sort imports  (Code Quality)  | [Ruff] | [article](https://www.sicara.fr/blog-technique/boost-code-quality-ruff-linter)
Static type checking                         | [Mypy] | [article](https://python.plainenglish.io/does-python-need-types-79753b88f521)
code security                                | [bandit] | [article](https://blog.bytehackr.in/secure-your-python-code-with-bandit)
Code quality & security each commit          | [pre-commit] | [article](https://dev.to/techishdeep/maximize-your-python-efficiency-with-pre-commit-a-complete-but-concise-guide-39a5)
Test code                                    | [Pytest] | [article](https://realpython.com/pytest-python-testing/)
Test coverage                                | [coverage.py] [codecov] | [article](https://martinxpn.medium.com/test-coverage-in-python-with-pytest-86-100-days-of-python-a3205c77296)
Project Template                             | [Cruft] or [Cookiecutter] | [article](https://medium.com/@bctello8/standardizing-dbt-projects-at-scale-with-cookiecutter-and-cruft-20acc4dc3f74)
Folder structure for data science projects   | [Data structure] | [article](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
Template for pull requests                   | [Pull Request template] | [article](https://www.awesomecodereviews.com/pull-request-template/)
Template for notebooks                       | [Notebook template] |

## Set up the environment

1. Initialize git in local:

    ```bash
    make init_git
    ```

1. **macOS only:** LightGBM and XGBoost need LLVM's OpenMP runtime (official Homebrew formula,
   MIT licence). Linux CI runners already have it.

    ```bash
    brew install libomp
    ```

1. Set up the environment:

    ```bash
    make install_env
    ```

1. Activate virtual environment:

    ```bash
    source .venv/bin/activate
    ```

1. Install libraries for data science and machine learning:

    ```bash
    make install_data_libs
    ```

## Demo — account-takeover risk (issue #8)

A local [Streamlit](https://streamlit.io/) app scores a login with the selected model and shows the response tier
(allow / step-up / review-block) and the inputs behind the score. It runs only on your machine
(`127.0.0.1`), with Streamlit's usage statistics turned off (`.streamlit/config.toml`).

1. After [setting up the environment](#set-up-the-environment) (including `brew install libomp` on macOS), build
   the data and the model once (each step is idempotent; the notebooks take ~70 and ~2 minutes):

    ```bash
    make data intermediate features split
    make notebooks_run NB=notebooks/5-models/06-sap-model_selection-2026_11_03.ipynb
    make notebooks_run NB=notebooks/6-interpretation/07-sap-model_interpretation-2026_11_04.ipynb
    ```

1. Start the demo and open <http://127.0.0.1:8501>:

    ```bash
    make demo
    ```

Pick a kind of example in the sidebar (a takeover sent to review, one caught by step-up, a false alarm, an ordinary
login), then switch a *What if…* signal to see the score and the response change. `?example=<review|step-up|false-alarm|ordinary>&n=<k>`
in the URL opens a given example. Screenshots and details: [`notebooks/7-deploy/`](https://github.com/modsiw-mona/anomaly_detection_ato/blob/main/notebooks/7-deploy/README.md).

![Demo: a takeover sent to review](https://raw.githubusercontent.com/modsiw-mona/anomaly_detection_ato/main/notebooks/7-deploy/screenshots/01-takeover-review.png)

## Install dependencies

After init the environment to install a new package, run:

```bash
uv add <package-name>
```

Example to install [plotly](https://plotly.com/python/) in dev group:

```bash
uv add --group dev plotly
```

## Project structure

- [Data structure]
- [Pipelines based on Feature/Training/Inference Pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

```bash
.
├── codecov.yml                         # configuration for codecov
├── .code_quality
│   ├── mypy.ini                        # mypy configuration
│   └── ruff.toml                       # ruff configuration
├── data
│   ├── 01_raw                          # raw immutable data
│   ├── 02_intermediate                 # typed data
│   ├── 03_primary                      # domain model data
│   ├── 04_feature                      # model features
│   ├── 05_model_input                  # often called 'master tables'
│   ├── 06_models                       # serialized models
│   ├── 07_model_output                 # data generated by model runs
│   ├── 08_reporting                    # reports, results, etc
│   └── README.md                       # description of the data structure
├── docs                                # documentation for your project
├── .editorconfig                       # editor configuration
├── .github                             # github configuration
│   ├── dependabot.md                   # github action to update dependencies
│   ├── pull_request_template.md        # template for pull requests
│   └── workflows                       # github actions workflows
│       ├── ci.yml                      # run continuous integration (tests, pre-commit, etc.)
│       ├── dependency_review.yml       # review dependencies
│       ├── docs.yml                    # build documentation (mkdocs)
│       └── pre-commit_autoupdate.yml   # update pre-commit hooks
├── .gitignore                          # files to ignore in git
├── Makefile                            # useful commands to setup environment, run tests, etc.
├── models                              # store final models
├── notebooks
│   ├── 1-data                          # data extraction and cleaning
│   ├── 2-exploration                   # exploratory data analysis (EDA)
│   ├── 3-analysis                      # Statistical analysis, hypothesis testing.
│   ├── 4-feat_eng                      # feature engineering (creation, selection, and transformation.)
│   ├── 5-models                        # model training, evaluation, and hyperparameter tuning.
│   ├── 6-interpretation                # model interpretation
│   ├── 7-deploy                        # model packaging, deployment strategies.
│   ├── 8-reports                       # story telling, summaries and analysis conclusions.
│   ├── notebook_template.ipynb         # template for notebooks
│   └── README.md                       # information about the notebooks
├── .pre-commit-config.yaml             # configuration for pre-commit hooks
├── pyproject.toml                      # dependencies for the python project
├── README.md                           # description of your project
├── src                                 # source code for use in this project
│   ├── README.md                       # description of src structure
│   ├── tmp_mock.py                     # example python file
│   ├── data                            # data extraction, validation, processing, transformation
│   ├── model                           # model training, evaluation, validation, export
│   ├── inference                       # model prediction, serving, monitoring
│   └── pipelines                       # orchestration of pipelines
│       ├── feature_pipeline            # transforms raw data into features and labels
│       ├── training_pipeline           # transforms features and labels into a model
│       └── inference_pipeline          # takes features and a trained model for predictions
├── tests                               # test code for your project
│   ├── test_mock.py                    # example test file
│   ├── data                            # tests for data module
│   ├── model                           # tests for model module
│   ├── inference                       # tests for inference module
│   └── pipelines                       # tests for pipelines module
└── .vscode                             # vscode configuration
    ├── extensions.json                 # list of recommended extensions
    ├── launch.json                     # vscode launch configuration
    └── settings.json                   # vscode settings
```

## Credits

This project was generated from [@JoseRZapata]'s [data science project template] template.

---
[@JoseRZapata]: https://github.com/JoseRZapata

[bandit]: https://github.com/PyCQA/bandit
[codecov]: https://codecov.io/
[Cookiecutter]:https://cookiecutter.readthedocs.io/en/stable/
[coverage.py]: https://coverage.readthedocs.io/
[Cruft]: https://cruft.github.io/cruft/
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
[Data structure]: https://github.com/JoseRZapata/data-science-project-template/blob/main/anomaly_detection_ato/data/README.md
[Mypy]: http://mypy-lang.org/
[Notebook template]: https://github.com/modsiw-mona/anomaly_detection_ato/blob/main/notebooks/notebook_template.ipynb
[pre-commit]: https://pre-commit.com/
[Pull Request template]: https://github.com/modsiw-mona/anomaly_detection_ato/blob/main/.github/pull_request_template.md
[Pytest]: https://docs.pytest.org/en/latest/
[Ruff]: https://docs.astral.sh/ruff/
[UV]: https://docs.astral.sh/uv/
