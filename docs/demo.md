# Demo

**Live:** <https://anomalydetection-ato.streamlit.app> (the first visit may take a minute while the app wakes up and downloads its files).

A small web app ([Streamlit](https://streamlit.io/)) presented as a simulated sign-in: a real login attempt on
the left, a sign-in card on the right. Pressing **Log in** shows the response in proportion to the risk, as the
person would see it (let in, asked for a code, or held until they verify their identity), and below it what the
system saw: the takeover score, how often such scores were takeovers, the thresholds and how the score is built.

## How to use it

1. **Pick a login:** the kind of record (a takeover the model blocks or asks for 2FA, a real user asked for 2FA
   by mistake, a real user let in) and the account (one that had logged in before, or its first login). A card
   shows the selected login at a glance. Each is a real test login.
2. **Look at the login attempt** on the left: every control is one input the model receives. Each slider's "?"
   says where takeovers and legitimate logins usually are; inputs that follow from others are greyed out (a new
   network is always a new IP address, for example).
3. **Press Log in.** The credentials are valid in every scenario, since a takeover uses the real password; only
   where and how the login happens can tell the owner from an attacker.
4. **Change the attempt** and log in again to see how the response changes. **Back to the real login** undoes the
   changes.
5. **Compare in the table:** each input's real value next to yours, and how much it raises or lowers the score (tree
   SHAP contributions, which, added to a typical ordinary login's score, give this login's score).

New to it? **Take the guided tour**: it dims the page and walks through each part.

## Screenshots

![A takeover: sign-in held](https://raw.githubusercontent.com/modsiw-mona/anomaly_detection_ato/main/notebooks/7-deploy/screenshots/01-takeover-block.png)

**A takeover, sign-in held.** A returning account logs in from a network seen by only 7 accounts before, with a 70%
attack-IP history, from a rare country and browser: score 0.5866, above the block threshold.

![A takeover asked for a second factor](https://raw.githubusercontent.com/modsiw-mona/anomaly_detection_ato/main/notebooks/7-deploy/screenshots/02-takeover-2fa.png)

**A takeover asked for a second factor.** The code goes to the account's phone, which the attacker does not have.

![An ordinary login allowed](https://raw.githubusercontent.com/modsiw-mona/anomaly_detection_ato/main/notebooks/7-deploy/screenshots/03-ordinary-allow.png)

**An ordinary login, allowed.** Common network, country and browser: score 0.0000.

![The guided tour](https://raw.githubusercontent.com/modsiw-mona/anomaly_detection_ato/main/notebooks/7-deploy/screenshots/04-guided-tour.png)

**The guided tour**, highlighting what the person sees.

## Run it

The [public version](https://anomalydetection-ato.streamlit.app) runs on Streamlit Community Cloud from a small published bundle (the eight example logins,
the reference sample, the summaries and the model, each checked against its SHA-256 before it is loaded).

On your machine, after building the data and the model (see [Reproduce](reproduce.md)):

```bash
make demo
```

It listens only on `127.0.0.1`, with Streamlit's usage statistics turned off.
