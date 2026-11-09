# Fictional premium gym: Kumo vs LightGBM

Can a foundation model with no training match a trained gradient boosting model on gym retention?

**Short answer: with no training, Kumo Tabular ties a trained LightGBM when it gets LightGBM's prepared feature table. On the raw relational tables, Kumo is clearly behind.**

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

## Kumo vs LightGBM

Setup, the same for every model:

- Eight monthly test anchors, 2025-01-01 to 2025-08-01. At each one, 300 members who visited in the previous 30 days, picked with a fixed seed. Every model scores those same members, which gives 2,400 member-months from 1,125 members.
- Tables are cut at the anchor before they reach Kumo. Labels come from the full tables.
- LightGBM trains only on rows whose 30 day label window closes before the anchor. Kumo's examples come from those same training anchors, and their labels matched LightGBM's training table in every row.
- 95% intervals come from 2,000 bootstrap resamples of members, paired across models.

Models:

- **LightGBM, all rows**: all training rows and the engineered features from `baseline.py`.
- **LightGBM, same rows**: trained only on the roughly 3,000 rows Kumo Relational got as context.
- **Kumo Relational, raw tables**: the eight tables and nothing else.
- **Kumo Relational + features**: the same, with LightGBM's features added to each member row.
- **Kumo Tabular**: 10,000 random rows of LightGBM's training table, the endpoint's limit.

Churn in the next 30 days (11.63% positive):

| Model | ROC AUC [95% CI] | PR AUC [95% CI] | Precision top 10% [95% CI] |
|---|---|---|---|
| LightGBM, all rows | 0.786 [0.752, 0.819] | 0.581 [0.524, 0.634] | 0.588 [0.515, 0.662] |
| LightGBM, same rows | 0.768 [0.733, 0.804] | 0.555 [0.495, 0.613] | 0.567 [0.483, 0.641] |
| Kumo Relational, raw tables | 0.696 [0.662, 0.729] | 0.294 [0.242, 0.348] | 0.329 [0.264, 0.388] |
| Kumo Relational + features | 0.775 [0.740, 0.810] | 0.556 [0.494, 0.613] | 0.571 [0.500, 0.641] |
| Kumo Tabular | 0.776 [0.742, 0.812] | 0.587 [0.530, 0.640] | 0.596 [0.521, 0.664] |

Paired difference against LightGBM, all rows:

| Model | ROC AUC | PR AUC | Precision top 10% |
|---|---|---|---|
| Kumo Relational, raw tables | -0.090 [-0.123, -0.057] | -0.288 [-0.338, -0.234] | -0.258 [-0.327, -0.204] |
| Kumo Relational + features | -0.011 [-0.037, +0.015] | -0.025 [-0.045, -0.003] | -0.017 [-0.050, +0.013] |
| Kumo Tabular | -0.009 [-0.033, +0.016] | +0.006 [-0.010, +0.021] | +0.008 [-0.021, +0.030] |

Revenue and visits in the next 30 days, MAE (lower is better). Mean revenue per member-month was 100.4 units and mean visits 8.95.

| Model | Revenue MAE [95% CI] | vs LightGBM | Visits MAE [95% CI] | vs LightGBM |
|---|---|---|---|---|
| LightGBM, all rows | 84.3 [73.3, 95.3] | | 4.35 [4.18, 4.52] | |
| LightGBM, same rows | 86.6 [75.2, 97.7] | +2.3 [+0.4, +4.1] | 4.49 [4.33, 4.67] | +0.15 [+0.08, +0.21] |
| Kumo Relational, raw tables | 95.1 [82.6, 107.5] | +10.8 [+7.3, +14.1] | 4.73 [4.56, 4.91] | +0.39 [+0.29, +0.49] |
| Kumo Relational + features | 89.1 [77.0, 101.0] | +4.8 [+1.8, +8.0] | 4.54 [4.37, 4.72] | +0.19 [+0.11, +0.27] |
| Kumo Tabular | 80.5 [69.7, 92.0] | -3.8 [-10.1, +2.8] | 4.32 [4.16, 4.49] | -0.02 [-0.07, +0.01] |

What this says:

- **Kumo Tabular ties LightGBM on all three tasks.** Every interval on its difference includes zero. It needs the feature table, so the feature work does not go away.
- **Kumo on raw tables is clearly behind on all three tasks.** It does not recover on its own what the hand built features capture.
- **Kumo Relational + features is close on churn ranking but behind elsewhere.** Its PR AUC, revenue and visits differences all exclude zero, though they are small.
- **Less data costs LightGBM too.** With about 3,000 rows it loses some accuracy, so part of Kumo Relational's gap is the row limit.

Raw numbers are in `results/multi_anchor_summary.json` and every prediction is in `results/multi_anchor_predictions.csv.gz`.

## Limits

- Every pattern was written into the generator by hand, so these scores measure how much of that design a model can recover. They say nothing about real members.
- LightGBM was not tuned.
- Kumo's settings were not tuned either.
- The features given to Kumo in two variants are LightGBM's own, so those variants test Kumo as a learner on the same inputs, not as a replacement for feature work.
- Each anchor has 300 members because of request size.
- The NVIDIA hosted endpoint is a preview service. Its model can change, so these numbers belong to the November 2026 run.

## Reproduce

```bash
pip install -r requirements.txt
python generate.py
python baseline.py
```

Running `generate.py` twice gave identical members, payments and visits files on the same machine and library versions. A rerun from a clean copy on another machine gave the same baseline numbers, except precision in the top 10% for days since last visit: 0.4432 instead of 0.4419, because many members tie on that value and the sort breaks ties differently.

To run the Kumo comparison, use your own NVIDIA API key. Never commit it.

```bash
export NVIDIA_API_KEY=<your key>
python kumo_trial.py
python fair_trial.py
python multi_anchor.py   # eight anchors, about 30 minutes
python make_figures.py
```

## License

Code: MIT. Data: CC BY 4.0, see `data/LICENSE`.
