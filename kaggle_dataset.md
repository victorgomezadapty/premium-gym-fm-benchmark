# Fictional Premium Gym Relational Dataset

A multi table dataset of a fictional premium gym with two sites and 2,400 members, from January 2022 to August 2025. It is built for churn, revenue and visit prediction, and for benchmarking relational and tabular models.

**Where it comes from.** A seeded script generates every row. Prices, plan mix, hours, seasonality, renewal rules, behavior parameters and labels are round, illustrative numbers chosen by hand. They were not measured from a real operator, and they were not calibrated to one. Amounts are in unnamed currency units. The data does not describe how a real gym behaves.

**Tables.** members, plans, contracts, payments, visits, trainers, pt_packages and pt_sessions, plus churn_features, a monthly feature table with a 30 day churn label.

**License.** CC BY 4.0.
