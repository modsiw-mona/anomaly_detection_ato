# Datasets

The detection pipeline is **dataset-agnostic**: it works on any login-event dataset that meets the
minimum requirements below. Each dataset is described by two files:

| File | Audience | Content |
| --- | --- | --- |
| `conf/dataset/<name>.yaml` | the code | Translation of the dataset's column names to the standard names, where to download it, its checksum, licence and known artefacts. Validated on load. |
| `docs/datasets/<name>.md` | people | Dataset card: provenance, how it was produced, what the labels mean, how faithful each signal is, limitations. |

## Minimum data requirements

A dataset can be used if it provides **all three required fields** and **at least one context field**:

| Role | Standard name | Meaning |
| --- | --- | --- |
| Required | `user_id` | The account the login attempt targets. Pseudonymised IDs are fine. |
| Required | `timestamp` | When the attempt happened. |
| Required | `login_success` | Whether the attempt succeeded. |
| Context (≥ 1) | `ip` | Client IP address (raw or hashed). |
| Context (≥ 1) | `asn` | Network the IP belongs to (an identifier, not a number to do maths on). |
| Context (≥ 1) | `device_type` | Device class, e.g. mobile or desktop. |
| Context (≥ 1) | `user_agent` | Raw user agent string. |

Optional fields (`country`, `region`, `city`, `os`, `browser`, `rtt_ms`) and labels
(`label_ato`, `label_attack_ip`) are used when present.

**Why these:** the approach compares each login with *that user's own history*. That needs to know
whose login it is (`user_id`), when (`timestamp`), whether it worked (`login_success`), and at
least one piece of context that can be "new for this user" (network, device, ...).

The rule is enforced in code by `check_requirements()` in `src/data/schema.py`. A configuration
file that maps too few fields is rejected when it is loaded.

## Adding a dataset

1. Copy `conf/dataset/rba.yaml` to `conf/dataset/<name>.yaml` and fill in the source, the
   checksum and the column mapping. Only map columns the dataset actually has.
2. Write `docs/datasets/<name>.md` using `rba.md` as the model.
3. Run `PYTHONPATH=src uv run python -m data.download --dataset <name>`.

No pipeline code changes are needed.

## Available datasets

| Name | Card | Source |
| --- | --- | --- |
| `rba` | [rba.md](rba.md) | [Zenodo 10.5281/zenodo.6782156](https://doi.org/10.5281/zenodo.6782156) |
