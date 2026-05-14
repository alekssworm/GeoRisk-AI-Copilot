# Model Comparison

Generated with `ml.classic.compare_models` using spatial block CV on the original
`Radiation_Dose_Rate_Prediction` nuclide training table.

| Candidate | Model | Feature set | Rows | Features | CV R2 mean | CV RMSE mean | CV MAE mean |
|---|---|---|---:|---:|---:|---:|---:|
| MVP-B primary from Radiation_Dose_Rate_Prediction | extra_trees | env_plus_no_ratio | 545 | 18 | 0.2698 | 0.01166 | 0.00888 |
| Current real-data reference without nuclides | random_forest | env_only | 545 | 13 | -0.0197 | 0.01378 | 0.01049 |

- Current synthetic demo model: test R2 `0.9578`, RMSE `0.03628`, MAE `0.02905`. This is measured on generated synthetic data, so it is not directly comparable to the real-data spatial CV table.

Conclusion: for the original real dataset, the MVP-B primary model is the
production candidate because it uses the original radionuclide and environmental
feature set. The synthetic demo model remains useful for local UI/API demos, but
its score is not evidence of real-world accuracy.
