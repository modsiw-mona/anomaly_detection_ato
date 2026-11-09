# Limitations

## Of the dataset

* **Synthetic values.** The RBA dataset keeps the real frequencies of each value, globally and per user, but the
  values themselves were regenerated. It is fit to compare methods, not to deploy the trained model.
* **Synthetic geography.** Countries were assigned by value, so distances between them mean nothing: legitimate
  logins move between Brazil and Norway within an hour 699 times on the development side. Impossible travel can only
  be tested by time (a country change within an hour), not by distance. Varying that window from 15 minutes to 12
  hours changes little.
* **Synthetic networks.** Network numbers (ASNs) are real, but which login falls on which network is artificial, so
  external reputation (hosting, VPN, block lists) would describe the real network, not the original login. Only the
  dataset's own history is used.
* **The log starts mid-way through accounts' lives.** A "first login" is the first one seen in the log, not the
  account's creation. In February 2020, when the log starts, 52% of logins are "first logins"; by October, about
  19%. That inflates the cold start, the friction on first logins and the takeovers without history. A real service
  keeps per-account state (first-seen date, known devices and countries) and knows when each account was created.
* **Few and incomplete labels.** 140 confirmed takeovers; unconfirmed ones count as legitimate, so precision is a
  lower bound. Labels stop at the end of November 2020.
* **Attack-IP flag.** If the dataset authors matched attacker lists after the fact, the attack-IP signals carry
  knowledge a live system would only partly have; the test numbers are then an upper bound.

## Of the evaluation

* **28 test takeovers.** Every test number is uncertain, hence the bootstrap intervals and the development estimate
  over 15 folds (112 takeovers). More test takeovers would require taking them from training.
* **One attack campaign** holds most takeovers. With the other takeovers hidden from training, the model generalises
  much less to them: a new kind of attack is the honest limit of this work.

## Of the design

* **Missing signals.** What would help most with first logins is not in the data: account creation date, a
  persistent device identifier, events after the login (email, password or 2FA changes), leaked-password checks.
* **Real-time cost not measured.** Scoring is fast; computing the history features in real time was not measured.
