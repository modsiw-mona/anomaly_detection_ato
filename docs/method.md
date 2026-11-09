# Method

The project follows CRISP-DM, one step per GitHub issue (#1 to #8), each delivered as an executed notebook plus
tested code in `src/`. Every result on this site comes from those notebooks.

## The problem

* **Unit:** one successful login. The question is whether the person logging in is the account's owner.
* **Label:** `Is Account Takeover`, confirmed by the service's incident response: 140 takeovers on successful logins,
  138 users. Labels stop at the end of November 2020, so the study uses February to November 2020 (about 9.8 million
  successful logins); later takeovers would be invisible, not absent.
* **Imbalance:** about **1 takeover in 70,000** logins. A model that never flags anything is 99.999% accurate,
  and ROC-AUC is optimistic at this rate, so the main metric is **PR-AUC** (average precision), with recall at
  fixed alert budgets.

## Features: only the past

Every feature of a login is computed from events **strictly before it**, so the model never sees the future.
A test recomputes every feature with all later events removed and requires identical values.

| Group | What it measures | Why |
| --- | --- | --- |
| `history` | earlier logins of the account, first login, hours since the previous one | dormant accounts, cold start |
| `user_novelty` | first time this account uses this country, network (ASN), IP, device, OS, browser, user agent, region, city; how many at once | deviation from the owner's own pattern |
| `global_rarity` | share of all earlier logins with the same value (country, ASN, IP, device, OS, browser, user agent) | works even on a first login |
| `sharing` | distinct accounts that logged in before from the same IP or network | shared attacker infrastructure |
| `failures` | failed attempts on the account in the last hour and day | password guessing |
| `campaign` | the network's share of earlier failed and attack-IP attempts, its attempts in the previous 24 hours, accounts that already used the exact user agent | attackers reuse infrastructure across victims |
| `travel` | country change within an hour of the previous login; number of earlier countries | impossible travel |
| context | IP on an attack list, missing region or city, device type, hour and weekday | |

The pipeline is **dataset-agnostic**: a YAML file maps any login dataset's columns to standard names
([how to add a dataset](datasets/README.md)).

## Evaluation protocol

* **Split by user, fixed before any model:** 80% of users for development, 20% for test, stratified so takeover
  users fall on both sides. All logins of a user stay on one side, so the model is always judged on people it has
  never seen. No choice was made on the test side (see [Results](results.md) for one earlier test run).
* **Cross-validation by user** on the development side (5 folds, repeated 3 times), with the takeover users spread
  evenly across folds.
* **Alert budgets:** the share of logins a service is willing to challenge, fixed before any model was trained (issue #5):
  ask for 2FA at the riskiest 1%, block at the riskiest 0.1%. The score thresholds are set on development scores.
* **Uncertainty:** bootstrap intervals that resample the takeovers, the scarce class.

## Baselines

| Baseline | PR-AUC | Recall at 1% |
| --- | --- | --- |
| Freeman et al. (2016) risk score | 0.0002 | 22% |
| Rule: new country | 0.0006 | 50% |
| Rule: new country and attack IP | 0.004 | 37% |
| Logistic regression (balanced weights) | 0.023 | 80% |

Baselines report PR-AUC pooled over the out-of-fold scores; the family comparison below reports the mean over 5
folds, hence 0.023 and 0.033 for the same logistic regression.

Freeman's model assumes an attacker imitates *common* values; in this service takeovers come from globally *rare*
ones, so its popularity factor works against it.

## Model selection

Seven families were compared on development users (5 folds), from linear to boosted trees, plus an unsupervised
detector and positive-unlabelled learning:

| Family | PR-AUC | Recall at 1% |
| --- | --- | --- |
| Logistic regression | 0.033 | 81% |
| Isolation Forest (no labels) | 0.011 | 56% |
| Gradient boosting (scikit-learn) | 0.366 | 85% |
| Random forest | 0.398 | 88% |
| LightGBM / XGBoost | 0.405 / 0.400 | 87% / 87% |
| PU bagging | 0.428 | 87% |

Trees beat the linear bar by 11 to 13 times; without labels the detector falls below it. One attack campaign (from
Romania) holds most of the takeovers, so a weighting that gives it the same total weight as all other takeovers was
tested, along with leave-campaign-out checks, ablations and tuning of the two best families.

**Selected** by a rule coded before the run (the highest mean PR-AUC over the 15 folds, among models above the
logistic-regression bar with recall at 1% of at least 80%): a random forest (minimum leaf size 2) with the
`campaign` and `travel` features and campaign-balanced weights. **PR-AUC 0.448** over 15 development folds, recall
81% at 0.1% and 89% at 1%. It ties LightGBM in a paired comparison and ranks the takeovers outside the main campaign
best (PR-AUC 0.159, against 0.11 to 0.14).
