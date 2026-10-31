# Dataset card — Login Data Set for Risk-Based Authentication (`rba`)

> Translation file: [`conf/dataset/rba.toml`](../../conf/dataset/rba.toml) ·
> First look: [`notebooks/1-data/01-sap-data_download-2026_10_31.ipynb`](../../notebooks/1-data/01-sap-data_download-2026_10_31.ipynb)

## At a glance

| | |
| --- | --- |
| **What** | Login attempts to a large single sign-on (SSO) service in Norway, one row per attempt |
| **Size** | 31.3 M login attempts, 3.3 M users (companion paper); 15 columns |
| **Period** | February 2020 – February 2021 (original collection) |
| **Labels** | `Is Account Takeover` (confirmed by incident response, 87 among successful logins per the paper); `Is Attack IP` (IP found in attacker lists) |
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
| `User ID` | `user_id` | required | Random pseudonym |
| `Login Timestamp` | `timestamp` | required | Order preserved, randomness added |
| `Login Successful` | `login_success` | required | |
| `IP Address` | `ip` | context | Synthetic value; see artefacts |
| `ASN` | `asn` | context | Network ID — an identifier, never a quantity; see artefacts |
| `Device Type` | `device_type` | context | Identical to the real data; empty for 1,526 unparseable user agents |
| `User Agent String` | `user_agent` | context | Regenerated from OS and browser |
| `Country` | `country` | optional | Synthetic value, real frequencies |
| `Region` | `region` | optional | Derived from the synthetic IP |
| `City` | `city` | optional | Derived from the synthetic IP |
| `OS Name and Version` | `os` | optional | |
| `Browser Name and Version` | `browser` | optional | |
| `Round-Trip Time [ms]` | `rtt_ms` | optional | Realistic values, reassigned within each geolocation |
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
  account takeovers. The paper reports **87 successful logins** — about 0.0007% of 12.5 M successful
  logins. True means *confirmed*; false means *not flagged*, which is not the same as legitimate. This
  is a **positive-unlabelled (PU)** label, so precision is reported as a lower bound.
- **`Is Attack IP`**: the IP appears in the FireHOL attacker lists, matched against the original
  data. It marks *attack attempts*, mostly failed, not confirmed takeovers. It is never the modelling
  target. Successful logins with an attack IP but no takeover flag are candidate missed takeovers.

## Known artefacts

Declared in `conf/dataset/rba.toml` (`[[artefacts]]`), flagged as boolean columns from issue #2:

| Flag | Rule | Meaning |
| --- | --- | --- |
| `asn_is_placeholder` | `asn ≥ 500000` | ASN that does not exist; assigned to resolve a synthesis conflict |
| `ip_is_placeholder` | `ip ∈ 10.0.0.0/8` | Private-range IP that cannot appear on the internet; synthesis placeholder |

According to the authors, these values do not affect Freeman RBA scores, since that model uses
frequencies only. The same holds for our behaviour features, and an ablation in issue #6 checks it.

## Limitations

- **Synthetic:** suitable for methodological research and for comparing methods, not for deploying a
  model trained on it.
- **One service, one country, one year:** a Norwegian SSO in 2020–2021, during the pandemic.
  Generalisation to other services is not established.
- **Few, incomplete labels:** 87 confirmed takeovers make every metric noisy (confidence intervals
  are mandatory), and unconfirmed takeovers sit among the "negatives".
- **Possible label bias:** the incident response team probably caught the most visible takeovers, so
  the 87 may under-represent subtle ones.
- **Weakened signals:** RTT and fine-grained geography were partly randomised during synthesis.

## Ethics and privacy

According to the authors, users agreed to data collection for research, and the synthesised release
was agreed with the data owner so that customers cannot be re-identified. IPs, browser identifiers,
timestamps and RTTs were randomly generated or assigned, so the dataset contains no real personal
data.
