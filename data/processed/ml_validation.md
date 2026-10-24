# Machine-learning validation and diagnostics

- Target: The target is a modelled spatial-accessibility indicator derived from the project's E2SFCA pipeline; it is not an independent observation of staffed beds, patient utilisation, or clinical outcomes.
- Observations: 1,634; distinct regions: 17.
- Candidate validation: GroupKFold grouped by region; entire region groups remain outside each training fold.
- Top single candidate by pooled outer-fold log-scale R² (exploratory; selection can be optimistic): ridge.
- Nested model-selection evaluation available: True.
- Nested model-selection procedure: Nested GroupKFold: model family selected using training-region folds, then evaluated on outer held-out region folds.
- Nested-selection MAE: 0.4736 beds per 1,000.
- Nested-selection RMSE: 0.7160 beds per 1,000.
- Baseline MAE: 0.5525; comparison MAE: 0.4736 beds per 1,000.
- Empirical coverage on outer held-out rows: 95.8%.
- Prediction interval status: Bands use training-fold inner out-of-fold residuals and are evaluated only on the corresponding outer held-out rows, so each row's own outcome is not used to set its width. They are empirical spatial-validation bands, not a formal distribution-free conformal guarantee; geographic dependence, region heterogeneity, model selection, and limited regional sample size may affect coverage.
- Interval warning: intervals are training-fold calibrated empirical spatial-validation bands, not formal distribution-free conformal intervals.
- Preprocessing: feature imputation and scaling are fitted inside validation training folds.
- Feature importance: random-forest permutation importance is calculated on held-out fold rows.
- Audit files: `ml_predictions.csv`, `ml_importance.csv`, `ml_region_metrics.csv`, and `ml_subgroup_metrics.csv`.

## Interpretation limits

The target is produced by the application's E2SFCA accessibility pipeline, so this model predicts a derived spatial-accessibility indicator. It does not independently validate staffed beds, patient utilisation, health outcomes, or causality. Region-blocked CV assesses transfer to held-out regional groups, but should not be mistaken for prospective validation. Group-specific errors and travel-time-source differences should be reviewed before policy use.
