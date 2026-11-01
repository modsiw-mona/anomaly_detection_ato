# Dataset card — Login Data Set for Risk-Based Authentication (`rba`)

> Translation file: [`conf/dataset/rba.yaml`](../../conf/dataset/rba.yaml) ·
> First look: [`notebooks/1-data/01-sap-data_download-2026_10_31.ipynb`](../../notebooks/1-data/01-sap-data_download-2026_10_31.ipynb)

## At a glance

| | |
| --- | --- |
| **What** | Login attempts to a large single sign-on (SSO) service in Norway, one row per attempt |
| **Size** | 31,269,264 login attempts; 3,304,502 users with at least one successful login (4,304,857 distinct user IDs overall); 15 columns + a row index; 9.05 GB CSV |
| **Period** | February 2020 – February 2021 (original collection) |
| **Labels** | `Is Account Takeover`: **141 rows (140 on successful logins, 138 users)**, confirmed by incident response — the paper reports 87, see *Labels* · `Is Attack IP`: 3,096,977 rows (9.90% of attempts) |
| **Nature** | **Synthesised** from real behaviour: frequencies preserved, values artificial |
| **Source** | Zenodo, DOI [10.5281/zenodo.6782156](https://doi.org/10.5281/zenodo.6782156), v1.0.0 · mirror: Kaggle `dasgroup/rba-dataset` · project page: [das-group/rba-dataset](https://github.com/das-group/rba-dataset) |
| **Licence** | CC BY 4.0. Citing the paper below is a condition of use. |
| **Download** | `make data` (verifies the published MD5 `cc1b1078b3929650e6c08678caffcc57`) |

**Cite as:** Wiefling, S., Jørgensen, P. R., Thunem, S., & Lo Iacono, L. (2022). Pump Up Password
Security! Evaluating and Enhancing Risk-Based Authentication on a Real-World Large-Scale Online
Service. *ACM Transactions on Privacy and Security*, 26(1). https://doi.org/10.1145/3546069

## Why this dataset

It is the only public dataset whose unit of analysis is a **consumer login event** and that carries
**confirmed account-takeover labels**. The thesis literature on risk-based authentication (Freeman
et al., 2016; Wiefling et al., 2021–2022) was built on it, and later detection work uses it too.
Alternatives were considered and rejected (development plan, 2026-10-21): LANL `cyber1` (enterprise
lateral movement), CERT insider threat (malicious *legitimate* users), and IEEE-CIS (transactions,
no authentication records).

## Columns

| Column in file | Standard name | Role | Notes |
| --- | --- | --- | --- |
| `User ID` | `user_id` | required | Random pseudonym. ~1 M IDs appear only in failed attempts (likely non-existent accounts targeted by attackers) |
| `Login Timestamp` | `timestamp` | required | Order preserved, randomness added |
| `Login Successful` | `login_success` | required | |
| `IP Address` | `ip` | context | Synthetic value; see artefacts |
| `ASN` | `asn` | context | Network ID — an identifier, never a quantity; see artefacts |
| `Device Type` | `device_type` | context | Identical to the real data; empty for 1,526 unparseable user agents |
| `User Agent String` | `user_agent` | context | Regenerated from OS and browser |
| `Country` | `country` | optional | Synthetic value, real frequencies |
| `Region` | `region` | optional | Derived from the synthetic IP; `-` used as a null marker |
| `City` | `city` | optional | Derived from the synthetic IP; `-` used as a null marker |
| `OS Name and Version` | `os` | optional | |
| `Browser Name and Version` | `browser` | optional | |
| `Round-Trip Time [ms]` | `rtt_ms` | optional | Realistic values, reassigned within each geolocation; often empty |
| `Is Account Takeover` | `label_ato` | label | Confirmed takeovers only |
| `Is Attack IP` | `label_attack_ip` | label | Proxy for attack *attempts* |

## How the data was produced

Statements from the dataset authors' README ("Data Creation"), not our assumptions:

- The statistical feature probabilities, **globally and per user, are identical to the real data**
  for the categorical features. Everything else was generated randomly while keeping logical
  relations and time order.
- IP addresses and user agent strings were generated from public data, so the values were very
  likely *not* present in the real data.
- **Country** was assigned randomly per unique feature value, then an **ASN** related to that
  country, then **IP addresses** for that ASN. **City and region** were derived from the generated
  IPs "and do not reflect the real logical relations from the original data set".
- **Device types are identical** to the real data. OS was assigned from the device type and the
  browser from the OS, then the user agent string was generated, so all logical relations around
  the user agent match the real data.
- **Timestamps** are not identical and contain randomness.
- **RTT** was drawn from the login-success status and the synthesised geolocation, so it is
  realistic per location but probably not in its original position.
- The authors state the synthesised data reproduces their published RBA results for the IP-based
  and user-agent-based features. Their warning: values are plausible but "totally artificial", so
  the data must not be used in productive systems such as intrusion detection.

### What this means for modelling

| Signal | Fidelity | Treatment |
| --- | --- | --- |
| Device type | Identical to the real data | Use |
| User agent, OS, browser | Relations identical, values regenerated | Use |
| IP, ASN, country | Frequencies identical, values synthetic | Use through behaviour features (new for this user, rarity, accounts sharing it) |
| City, region | Real relations not preserved | Included and **tested with an ablation** — not dropped untested |
| Timestamps | Order kept, noise added | Use; temporal features tested |
| RTT | Realistic per location, reassigned | Included and **tested with an ablation** |

## Labels

- **`Is Account Takeover`**: login attempts the service's incident response team identified as
  account takeovers. **The published file has 141 such rows: 140 on successful logins (0.0011% of
  12,541,442) and 1 on a failed login, belonging to 138 distinct users** (3 users have two rows).
  The companion paper reports *"87 successful logins were detected as account takeover"* (§4.1) for
  the original data, and does not describe how the label was carried into the synthesised release.
  The difference is **unexplained by the authors**; the published file is the source of truth for
  this project, and the discrepancy is reported as a limitation. True means *confirmed*; false means *not flagged*, which is not the same as legitimate. This
  is a **positive-unlabelled (PU)** label, so precision is reported as a lower bound.
- **`Is Attack IP`**: the IP appears in the FireHOL attacker lists, matched against the original
  data. It marks *attack attempts*, mostly failed, not confirmed takeovers. It is never the modelling
  target. Successful logins with an attack IP but no takeover flag are candidate missed takeovers.

## Known artefacts

Declared in `conf/dataset/rba.yaml` (`artefacts`), flagged as boolean columns in the intermediate
dataset (issue #2):

| Flag | Rule | Rows | Failed logins | Successful logins | Meaning |
| --- | --- | --- | --- | --- | --- |
| `asn_is_placeholder` | `asn ≥ 500000` | 1,457,824 (4.7%) | 5.1% | 4.1% | ASN that does not exist; assigned to resolve a synthesis conflict |
| `ip_is_placeholder` | `ip ∈ 10.0.0.0/8` | 7,291,335 (23.3%) | 28.7% | 15.3% | Private-range IP that cannot appear on the internet; synthesis placeholder |

According to the authors, these values do not affect Freeman RBA scores, since that model uses
frequencies only. Placeholder IPs look more frequent in failed logins than in successful ones, but
that is **confounded by a catch-all user ID** (see *Data quality findings* in the EDA, issue #3):
among real accounts the relation reverses (10.8% of failed vs 15.3% of successful logins). Issue #3
measures the link with the labels, and an ablation in issue #6 checks the effect.

## Intermediate dataset (`data/02_intermediate/rba.parquet`)

Built by `make intermediate` (`src/data/intermediate.py`) in ~25 s with ~7.5 GB peak memory; 676 MB
(zstd) from the 9.05 GB CSV. Verified against the raw file on 12 independent counts
(notebook `2-exploration`).

**Missing values**, decided from a scan of every column and declared per column (`null_markers`):

| Value | Column | Verdict |
| --- | --- | --- |
| `-` | `region`, `city` | Null marker |
| `Nan` | `region` (3 rows) | **Real value**: Nan province, Thailand (city Pua) |
| `unknown` | `device_type` (867,371 rows) | **Real category**, listed by the authors |
| empty | `device_type` (1,526 rows) | Kept as null: unparseable user agents (authors' README) |

**Coverage after unifying nulls:**

| Field | Missing | Notes |
| --- | --- | --- |
| `rtt_ms` | 95.9% | Present for 7.0% of successful and 2.1% of failed logins |
| `region` | 45.0% | Present for 62.4% of successful logins |
| `city` | 44.5% | Present for 63.0% of successful logins |
| `device_type` | 0.005% | Plus 867,371 `unknown` |
| all others | 0% | |

**Storage types:** `user_id` Int64 · `timestamp` Datetime[ms] · `login_success` and labels Boolean ·
`ip`, `asn`, `country`, `region`, `city`, `os`, `browser`, `device_type` Categorical (identities, not
quantities) · `user_agent` String · `rtt_ms` Float64 · plus `event_id` (row position in the raw file)
and the two artefact flags. Rows stay in time order.

## Data quality findings (EDA, issue #3)

Measured in notebook `3-analysis`:

- **No duplicates**: 0 identical rows, 0 repeated (user, millisecond) pairs.
- **A catch-all user ID** (`-4324475583306591935`, declared as `catch_all_user_ids`): 14,025,899
  attempts (45% of all rows, 75% of failed attempts) with 4 successes, from 227 countries, ~11,700
  ASNs and 14% bots. It is not a person; it is excluded from per-user behaviour and confounds any
  failed-login statistic that does not exclude it.
- **Takeover labels end in November 2020.** No takeover is labelled in December 2020 – February
  2021, although successful logins stay at ~1 M per month. Those months are *unlabelled*, not clean.
- **`Is Attack IP` is noisy:** one active legitimate account (~6,000 successes) logs in 97% of the
  time from "attack IPs" and has no takeover. The flag is still the strongest *global* signal for
  takeovers (×17 on successful logins).
- **Behaviour separates takeovers strongly** (logins with history, user-based sample): new country
  80% vs 1% (×83), new ASN 86% vs 10% (×8), attack IP 55% vs 6% (×10).
- **39% of takeovers happen on the user's first successful login**, so per-user novelty cannot see
  them.
- **Failed attempts just before a takeover are barely more frequent** than before a legitimate
  login (×1.1–1.2).
- **RTT is present for 1 of 140 takeovers.**

**Validation rules** (`src/data/validation.py` + `validation` in `conf/dataset/rba.yaml`) pass on
all 31.3 M rows in ~4 s, with one expected warning: the takeover on a failed login.

## Limitations

- **Synthetic:** suitable for methodological research and for comparing methods, not for deploying a
  model trained on it.
- **One service, one country, one year:** a Norwegian SSO in 2020–2021, during the pandemic.
  Generalisation to other services is not established.
- **Few, incomplete labels:** 140 confirmed takeovers make every metric noisy (confidence intervals
  are mandatory), and unconfirmed takeovers sit among the "negatives".
- **Possible label bias:** the incident response team probably caught the most visible takeovers, so
  the confirmed takeovers may under-represent subtle ones.
- **Unexplained label count:** 140 takeover rows on successful logins versus 87 in the paper (see
  *Labels*).
- **Labels cover February – November 2020 only** (see *Data quality findings*).
- **Weakened signals:** RTT and fine-grained geography were partly randomised during synthesis, and
  RTT is missing for 96% of attempts.
- **A catch-all user ID:** one ID holds 14 M failed attempts (45% of all rows) from 227 countries;
  it is not a person, and it distorts any per-user or failed-login statistic that does not exclude it
  (documented in the EDA, issue #3).

## Ethics and privacy

According to the authors, users agreed to data collection for research, and the synthesised release
was agreed with the data owner so that customers cannot be re-identified. IPs, browser identifiers,
timestamps and RTTs were randomly generated or assigned, so the dataset contains no real personal
data.
