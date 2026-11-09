# Fictional premium gym: Kumo vs LightGBM

Can a foundation model with no training match a trained gradient boosting model on gym retention?

**Status: the classic baselines are done. The Kumo comparison has not been run on this data, so there is no answer yet.**

## Where the data comes from

The data describes a fictional premium gym with two sites and 2,400 members, from January 2022 to August 2025. `generate.py` creates all of it with a fixed seed (42).

Prices, plan mix, hours, seasonality, renewal rules, behavior parameters and labels are round, illustrative numbers chosen by hand. They were not measured from a real operator, and they were not calibrated to one. The general idea of what a premium gym sells comes from the author's years working in the industry, which is why the tables look familiar. The numbers themselves are made up, so nothing here describes how a real gym behaves.

An earlier version of this benchmark was retired because its parameters were not independent of real data. Its results do not carry over to this version.

## Tables

| Table | Rows | Content |
|---|---:|---|
| members | 2,400 | join date, age, home site, gender, corporate flag |
| plans | 8 | plan name, type, duration, net price |
| contracts | 8,957 | start and end date of each membership |
| payments | 10,055 | net, tax, gross, method, source, discount, refund |
| visits | 270,734 | swipes, allowed or denied with a reason |
| trainers | 18 | role and hire date, pseudonymous |
| pt_packages | 1,568 | coaching packages bought |
| pt_sessions | 7,282 | coaching sessions, completed or cancelled |

## Baseline results

Task: for each member with at least one visit in the previous 30 days, predict no visit in the next 30 days. Anchors are monthly. Training anchors run from March 2022 to December 2024 and test anchors from January to August 2025, with no random split.

The test set has 7,748 member-months, and 11.73% of them churned.

| Model | ROC AUC | PR AUC | Precision in top 10% |
|---|---:|---:|---:|
| LightGBM | 0.765 | 0.565 | 0.590 |
| Logistic regression | 0.753 | 0.507 | 0.527 |
| Days since last visit | 0.720 | 0.348 | 0.442 |

LightGBM is ahead on all three measures, most clearly at the top of the list, which is what a retention team would act on.

## Limits

- Every pattern was written into the generator by hand, so these scores measure how much of that design a model can recover. They say nothing about real members.
- LightGBM was not tuned.
- `kumo_trial.py`, `fair_trial.py` and `multi_anchor.py` have not been run against this version of the data.

## Reproduce

```bash
pip install -r requirements.txt
python generate.py
python baseline.py
```

Running `generate.py` twice gave identical members, payments and visits files on the same machine and library versions.

To run the Kumo comparison, use your own NVIDIA API key. Never commit it.

```bash
export NVIDIA_API_KEY=<your key>
python kumo_trial.py
python fair_trial.py
python multi_anchor.py   # eight anchors, about 30 minutes
```

## License

Code: MIT. Data: CC BY 4.0, see `data/LICENSE`.
