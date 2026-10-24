"""Spatially validated models for the I-SAGIPH LGU table.

The regression target is ``beds_per_1000_all``: an E2SFCA-derived *modelled*
accessibility measure, not observed patient utilisation or staffed-bed availability.
Candidate models are evaluated on identical region-blocked folds. Preprocessing
(imputation and scaling) is fitted inside each training fold. A nested grouped-CV
pass estimates the performance of the model-selection procedure itself and
produces cross-fitted LGU predictions. Empirical residual bands are explicitly
reported as empirical intervals, not as distribution-free conformal guarantees:
spatial observations are dependent and cross-fitted residuals are reused.

The existing public API and output column names are retained for compatibility.
Additional metrics and CSV outputs are additive.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "poverty_incidence": "Poverty incidence",
    "gida_share": "GIDA share of barangays",
    "pct_age_0_4": "% aged 0–4",
    "pct_age_60_plus": "% aged 60+",
    "log_density": "Population density (log)",
    "log_population": "Population (log)",
}
TARGET = "beds_per_1000_all"
MODEL_NAMES = {
    "mean_baseline": "Mean baseline",
    "linear": "Linear regression",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
ALPHA = 0.10  # 90% empirical out-of-fold residual band
N_BOOTSTRAP = 200
PERMUTATION_REPEATS = 5


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a numeric input without silently converting missing values to zero."""
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    return s.replace([np.inf, -np.inf], np.nan)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features; all learned imputation happens inside CV folds."""
    X = pd.DataFrame(index=df.index)
    for c in ("poverty_incidence", "gida_share", "pct_age_0_4", "pct_age_60_plus"):
        if c in df:
            X[c] = _numeric(df, c)
    if "pop_density" in df:
        X["log_density"] = np.log1p(_numeric(df, "pop_density").clip(lower=0))
    pop = _numeric(df, "population").clip(lower=0)
    X["log_population"] = np.log1p(pop)
    X = X.replace([np.inf, -np.inf], np.nan)
    # A feature that is entirely missing cannot teach a model anything. Drop it
    # deterministically and document the omission; partial missingness is imputed
    # by the fold's fitted SimpleImputer in _new_model.
    return X.loc[:, X.notna().any(axis=0)]


def _new_model(name: str, seed: int = 42, params: dict | None = None) -> Pipeline:
    """Build a fresh estimator. Every imputer is trained only on its fold's train data."""
    params = params or {}
    rf_cfg = params.get("random_forest", {})
    gb_cfg = params.get("gradient_boosting", {})
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "mean_baseline":
        est = DummyRegressor(strategy="mean")
        return Pipeline([("imputer", imputer), ("model", est)])
    if name == "linear":
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()), ("model", LinearRegression())])
    if name == "ridge":
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()),
                         ("model", Ridge(alpha=float(params.get("ridge_alpha", 1.0))))])
    if name == "random_forest":
        est = RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 300)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
            random_state=seed, n_jobs=int(rf_cfg.get("n_jobs", 1)),
        )
        return Pipeline([("imputer", imputer), ("model", est)])
    if name == "gradient_boosting":
        est = HistGradientBoostingRegressor(
            max_iter=int(gb_cfg.get("max_iter", 300)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
            min_samples_leaf=int(gb_cfg.get("min_samples_leaf", 20)),
            l2_regularization=float(gb_cfg.get("l2_regularization", 1.0)),
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("model", est)])
    raise KeyError(f"unknown model: {name}")


def _folds(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, max_splits: int = 5,
           seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds when geography permits; explicit random fallback for one-region data."""
    n_groups = pd.Series(groups).nunique(dropna=False)
    if n_groups >= 2:
        splitter = GroupKFold(n_splits=min(max_splits, int(n_groups)))
        return list(splitter.split(X, y, groups))
    if len(X) < 2:
        return []
    n_splits = min(max_splits, len(X))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


def _scores(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """Report predictive errors on both the fitted log scale and interpretable bed scale."""
    y_log = np.asarray(y_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    y_beds = np.maximum(0.0, np.expm1(y_log))
    pred_beds = np.maximum(0.0, np.expm1(pred_log))
    if len(y_log) > 1 and np.nanstd(y_log) > 0:
        r2 = float(r2_score(y_log, pred_log))
    else:
        r2 = float("nan")
    return {
        "r2": r2,                         # legacy key; R² is on log1p(target)
        "r2_log": r2,
        "mae_beds": float(mean_absolute_error(y_beds, pred_beds)),
        "rmse_beds": float(np.sqrt(mean_squared_error(y_beds, pred_beds))),
        "bias_beds": float(np.mean(pred_beds - y_beds)),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
    }


def conformal_halfwidth(y: np.ndarray, oof_pred: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample residual quantile used for an *empirical* OOF prediction band.

    This utility is retained for compatibility. Using pooled, cross-fitted residuals
    from spatially dependent LGUs does not itself establish a formal conformal
    coverage guarantee; the report and dashboard state that limitation explicitly.
    """
    r = np.sort(np.abs(np.asarray(y, dtype=float) - np.asarray(oof_pred, dtype=float)))
    r = r[np.isfinite(r)]
    if not len(r):
        return float("nan")
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(r[min(max(k, 0), n - 1)])


def _block_bootstrap_ci(y_log: np.ndarray, pred_log: np.ndarray, groups: np.ndarray,
                        metric: str, seed: int = 42, n_boot: int = N_BOOTSTRAP) -> list[float]:
    """Percentile interval from resampling whole regions rather than individual LGUs."""
    rng = np.random.default_rng(seed)
    y_log = np.asarray(y_log, float)
    pred_log = np.asarray(pred_log, float)
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        # A row bootstrap is a fallback only; the output documents lack of spatial blocks.
        blocks = [np.arange(len(y_log))]
        row_fallback = True
    else:
        blocks = [np.flatnonzero(groups == g) for g in uniq]
        row_fallback = False
    values = []
    for _ in range(n_boot):
        if row_fallback:
            idx = rng.integers(0, len(y_log), size=len(y_log))
        else:
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            idx = np.concatenate([blocks[i] for i in chosen])
        sc = _scores(y_log[idx], pred_log[idx])
        if metric == "mae_improvement_vs_baseline":
            raise ValueError("Use _bootstrap_mae_improvement for two-model comparisons")
        v = sc.get(metric)
        if v is not None and np.isfinite(v):
            values.append(float(v))
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _bootstrap_mae_improvement(y_log: np.ndarray, baseline_log: np.ndarray, model_log: np.ndarray,
                               groups: np.ndarray, seed: int = 42,
                               n_boot: int = N_BOOTSTRAP) -> list[float]:
    rng = np.random.default_rng(seed)
    y_beds = np.maximum(0, np.expm1(np.asarray(y_log, float)))
    base_beds = np.maximum(0, np.expm1(np.asarray(baseline_log, float)))
    model_beds = np.maximum(0, np.expm1(np.asarray(model_log, float)))
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        blocks = [np.arange(len(groups))]
        row_fallback = True
    else:
        blocks = [np.flatnonzero(groups == g) for g in uniq]
        row_fallback = False
    values = []
    for _ in range(n_boot):
        if row_fallback:
            idx = rng.integers(0, len(groups), size=len(groups))
        else:
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            idx = np.concatenate([blocks[i] for i in chosen])
        values.append(float(mean_absolute_error(y_beds[idx], base_beds[idx])
                             - mean_absolute_error(y_beds[idx], model_beds[idx])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def residual_moran(df: pd.DataFrame, resid: np.ndarray, seed: int = 0) -> dict:
    """Global Moran's I for out-of-fold log-scale residuals."""
    from .spatial_stats import knn_weights, morans_i
    W = knn_weights(df["lat"].to_numpy(), df["lon"].to_numpy(), k=min(6, max(1, len(df) - 1)))
    m = morans_i(np.asarray(resid, float), W, permutations=499, seed=seed)
    return {"I": float(m["I"]), "p": float(m.get("p_sim", m.get("p", np.nan))),
            "residual_scale": "log1p(beds per 1,000)"}


def _region_error_table(df: pd.DataFrame, y_log: np.ndarray, pred_log: np.ndarray,
                        groups: np.ndarray, model_labels: np.ndarray | None = None) -> list[dict]:
    rows: list[dict] = []
    for region in pd.unique(groups):
        idx = np.flatnonzero(np.asarray(groups).astype(str) == str(region))
        vals = _scores(y_log[idx], pred_log[idx])
        rows.append({"region": str(region), "n": int(len(idx)), **vals})
    return rows


def _subgroup_error_table(df: pd.DataFrame, y_log: np.ndarray, pred_log: np.ndarray,
                          groups: np.ndarray) -> list[dict]:
    """Auditable subgroup errors by region, GIDA share and population size."""
    rows: list[dict] = []
    # Regions are the primary geographic domain-shift audit.
    for row in _region_error_table(df, y_log, pred_log, groups):
        rows.append({"dimension": "region", "group": row["region"], "n": row["n"],
                     **{k: row[k] for k in ("r2_log", "mae_beds", "rmse_beds", "bias_beds")}})

    for col, dimension, labels in (
        ("gida_share", "gida_share", ("lower or equal to median GIDA share", "above median GIDA share")),
        ("population", "population", ("lower or equal to median population", "above median population")),
    ):
        if col not in df:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() < 4 or vals.nunique(dropna=True) < 2:
            continue
        cut = float(vals.median())
        groups_ = [(labels[0], vals.le(cut).fillna(False).to_numpy()),
                   (labels[1], vals.gt(cut).fillna(False).to_numpy())]
        for group_name, mask in groups_:
            idx = np.flatnonzero(mask)
            if len(idx) < 2:
                continue
            rows.append({"dimension": dimension, "group": group_name, "n": int(len(idx)),
                         "r2_log": _scores(y_log[idx], pred_log[idx])["r2_log"],
                         "mae_beds": _scores(y_log[idx], pred_log[idx])["mae_beds"],
                         "rmse_beds": _scores(y_log[idx], pred_log[idx])["rmse_beds"],
                         "bias_beds": _scores(y_log[idx], pred_log[idx])["bias_beds"],
                         "threshold": cut})
    return rows


def expected_capacity(df: pd.DataFrame, seed: int = 42, model_config: dict | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Evaluate candidate models with region-blocked CV and nested model selection.

    Returns the original three objects: row-level predictions, metrics, and feature
    importance. The CSV/JSON schema is additive so existing dashboard components
    continue to work.
    """
    model_config = model_config or {}
    outer_splits = max(2, int(model_config.get("outer_cv_splits", 5)))
    inner_splits = max(2, int(model_config.get("nested_cv_splits", 3)))
    interval_alpha = float(model_config.get("interval_alpha", ALPHA))
    if not 0 < interval_alpha < 1:
        raise ValueError("ml.interval_alpha must be between 0 and 1")
    n_bootstrap = max(20, int(model_config.get("bootstrap_repeats", N_BOOTSTRAP)))
    permutation_repeats = max(1, int(model_config.get("permutation_repeats", PERMUTATION_REPEATS)))
    if TARGET not in df:
        raise ValueError(f"required model target {TARGET!r} is missing")
    target = pd.to_numeric(df[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = target.notna() & target.ge(0)
    data = df.loc[valid].copy().reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("expected_capacity requires at least five LGUs with non-negative target values")

    X = _features(data)
    if X.shape[1] == 0:
        raise ValueError("No usable features are available for expected-capacity modeling")
    y = np.log1p(target.loc[valid].to_numpy(dtype=float))
    if "region" in data:
        groups = data["region"].fillna("Unknown region").astype(str).to_numpy()
        grouping_field = "region"
    else:
        groups = np.repeat("All data", len(data))
        grouping_field = None
    unique_groups = int(pd.Series(groups).nunique())
    split_scheme = ("GroupKFold grouped by region; entire region groups remain outside each training fold"
                    if unique_groups >= 2 else
                    "KFold row-level fallback; only one region is available, so spatial generalisation cannot be tested")

    model_keys = list(MODEL_NAMES)
    folds = _folds(X, y, groups, max_splits=outer_splits, seed=seed)
    oof = {k: np.full(len(data), np.nan, dtype=float) for k in model_keys}
    fold_number = np.full(len(data), -1, dtype=int)
    fold_region_metrics: list[dict] = []
    importances: list[np.ndarray] = []
    coef_rows: list[np.ndarray] = []

    # Outer grouped folds: model comparisons and out-of-fold permutation importance.
    for fold_id, (tr, te) in enumerate(folds, start=1):
        fold_number[te] = fold_id
        fitted: dict[str, Pipeline] = {}
        for key in model_keys:
            model = _new_model(key, seed + fold_id, model_config)
            model.fit(X.iloc[tr], y[tr])
            p = np.asarray(model.predict(X.iloc[te]), dtype=float)
            oof[key][te] = p
            fitted[key] = model

            for region in pd.unique(groups[te]):
                ridx = te[groups[te] == region]
                vals = _scores(y[ridx], oof[key][ridx])
                fold_region_metrics.append({"model": key, "model_name": MODEL_NAMES[key], "fold": fold_id,
                                            "region": str(region), "n": int(len(ridx)), **vals})

        # Held-out importance, not importance measured on the same rows used for fitting.
        rf = fitted["random_forest"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = permutation_importance(rf, X.iloc[te], y[te], scoring="r2", n_repeats=permutation_repeats,
                                        random_state=seed + fold_id, n_jobs=1)
        importances.append(np.asarray(pi.importances_mean, float))
        lin = fitted["linear"].named_steps["model"]
        coef_rows.append(np.asarray(lin.coef_, float))

    missing_predictions = {k: int(np.isnan(v).sum()) for k, v in oof.items()}
    if any(missing_predictions.values()):
        raise RuntimeError(f"Cross-validation did not generate every out-of-fold prediction: {missing_predictions}")

    model_metrics: dict[str, dict] = {}
    for key in model_keys:
        score = _scores(y, oof[key])
        score["name"] = MODEL_NAMES[key]
        score["validation"] = split_scheme
        score["n"] = int(len(data))
        if unique_groups >= 2:
            score["mae_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, oof[key], groups, "mae_beds", seed + 10, n_bootstrap)
            score["rmse_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, oof[key], groups, "rmse_beds", seed + 20, n_bootstrap)
            score["bias_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, oof[key], groups, "bias_beds", seed + 30, n_bootstrap)
        else:
            score["mae_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, oof[key], groups, "mae_beds", seed + 10, n_bootstrap)
            score["interval_note"] = "Row bootstrap fallback: only one region was available."
        model_metrics[key] = score

    # Keep the existing ``best`` contract for compatibility, but use nested OOF
    # predictions for the headline estimate whenever geographic groups permit it.
    best = max(model_keys, key=lambda k: model_metrics[k]["r2_log"] if np.isfinite(model_metrics[k]["r2_log"]) else -np.inf)
    nested_oof = np.full(len(data), np.nan, dtype=float)
    nested_model = np.full(len(data), "", dtype=object)
    # Each held-out outer fold gets its own uncertainty width calibrated only from
    # inner OOF residuals in that fold's training regions. The outer held-out labels
    # never contribute to their own interval width.
    nested_halfwidth = np.full(len(data), np.nan, dtype=float)
    fold_calibration_rows: list[dict] = []
    nested_available = unique_groups >= 3
    if nested_available:
        for fold_id, (tr, te) in enumerate(folds, start=1):
            Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
            inner = _folds(Xtr, ytr, gtr, max_splits=inner_splits, seed=seed + 1000 + fold_id)
            if len(inner) < 2:
                nested_available = False
                break
            inner_oof_by_model: dict[str, np.ndarray] = {}
            inner_scores: dict[str, float] = {}
            for key in model_keys:
                inner_oof = np.full(len(tr), np.nan, dtype=float)
                for inner_id, (itr, iva) in enumerate(inner, start=1):
                    im = _new_model(key, seed + 10000 + fold_id * 100 + inner_id, model_config)
                    im.fit(Xtr.iloc[itr], ytr[itr])
                    inner_oof[iva] = im.predict(Xtr.iloc[iva])
                inner_oof_by_model[key] = inner_oof
                good = np.isfinite(inner_oof)
                inner_scores[key] = (_scores(ytr[good], inner_oof[good])["r2_log"] if good.sum() > 1 else -np.inf)
            selected = max(inner_scores, key=lambda k: inner_scores[k] if np.isfinite(inner_scores[k]) else -np.inf)
            final_model = _new_model(selected, seed + 20000 + fold_id, model_config)
            final_model.fit(Xtr, ytr)
            nested_oof[te] = final_model.predict(X.iloc[te])
            nested_model[te] = selected

            # Build a conservative empirical calibration score per inner-training row
            # using the largest absolute residual across candidate models. This reduces
            # dependence on which model happened to win the inner selection step. It is
            # still not a distribution-free guarantee for spatially dependent LGUs.
            inner_matrix = np.column_stack([inner_oof_by_model[k] for k in model_keys])
            usable_cal = np.isfinite(ytr) & np.isfinite(inner_matrix).all(axis=1)
            if usable_cal.sum() >= 5:
                max_candidate_error = np.max(np.abs(inner_matrix[usable_cal] - ytr[usable_cal, None]), axis=1)
                q_fold = conformal_halfwidth(max_candidate_error, np.zeros_like(max_candidate_error), alpha=interval_alpha)
                nested_halfwidth[te] = q_fold
                fold_calibration_rows.append({
                    "fold": int(fold_id), "selected_model": selected,
                    "n_inner_calibration_rows": int(usable_cal.sum()),
                    "n_inner_regions": int(pd.Series(gtr[usable_cal]).nunique()),
                    "halfwidth_log": float(q_fold),
                })
            else:
                fold_calibration_rows.append({
                    "fold": int(fold_id), "selected_model": selected,
                    "n_inner_calibration_rows": int(usable_cal.sum()),
                    "n_inner_regions": int(pd.Series(gtr[usable_cal]).nunique()),
                    "halfwidth_log": float("nan"),
                })
    if not nested_available or not np.isfinite(nested_oof).all():
        nested_oof = oof[best].copy()
        nested_model[:] = best
        nested_available = False
        nested_halfwidth[:] = np.nan
        fold_calibration_rows = []

    final_pred = nested_oof
    final_score = _scores(y, final_pred)
    nested_info = {
        "available": bool(nested_available),
        "method": "Nested GroupKFold: model family selected using training-region folds, then evaluated on outer held-out region folds"
                  if nested_available else "Nested model selection unavailable: fewer than three distinct regions or insufficient grouped folds; headline prediction falls back to the top single candidate's OOF predictions and must be treated as exploratory",
        **final_score,
        "selected_model_counts": {str(k): int((nested_model == k).sum()) for k in model_keys if (nested_model == k).any()},
    }
    nested_info["mae_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, final_pred, groups, "mae_beds", seed + 40, n_bootstrap)
    nested_info["rmse_beds_ci95_region_bootstrap"] = _block_bootstrap_ci(y, final_pred, groups, "rmse_beds", seed + 50, n_bootstrap)

    # Compare the baseline with the cross-fitted *selection procedure*, not the
    # best candidate chosen after inspecting pooled outer-fold scores. This avoids
    # giving the selected candidate an unfair advantage in the headline comparison.
    baseline_log = oof["mean_baseline"]
    y_beds = np.maximum(0.0, np.expm1(y))
    baseline_beds = np.maximum(0.0, np.expm1(baseline_log))
    selected_beds = np.maximum(0.0, np.expm1(final_pred))
    baseline_mae = mean_absolute_error(y_beds, baseline_beds)
    selected_mae = mean_absolute_error(y_beds, selected_beds)
    expected_features = [c for c in ("poverty_incidence", "gida_share", "pct_age_0_4", "pct_age_60_plus") if c in data]
    if "pop_density" in data:
        expected_features.append("log_density")
    expected_features.append("log_population")
    dropped_all_missing = [c for c in expected_features if c not in X.columns]
    metrics: dict = {
        "target": "E2SFCA-derived accessible beds per 1,000 (log1p target for fitting)",
        "target_definition": "The target is a modelled spatial-accessibility indicator derived from the project's E2SFCA pipeline; it is not an independent observation of staffed beds, patient utilisation, or clinical outcomes.",
        "n": int(len(data)), "n_regions": unique_groups, "grouping_field": grouping_field,
        "cv": split_scheme, "cv_n_splits": int(len(folds)), "models": model_metrics,
        "best": best, "best_selection_metric": "highest pooled outer-CV R² on log1p(target); nested CV separately estimates selection-procedure performance",
        "nested_selection": nested_info,
        "baseline_comparison": {
            "baseline": "mean_baseline",
            "comparison_model": "nested_model_selection" if nested_available else best,
            "comparison_uses_nested_oof": bool(nested_available),
            "baseline_mae_beds": float(baseline_mae), "selected_model_mae_beds": float(selected_mae),
            "mae_improvement_beds": float(baseline_mae - selected_mae),
            "mae_improvement_ci95_region_bootstrap": _bootstrap_mae_improvement(y, baseline_log, final_pred, groups, seed + 60, n_bootstrap),
            "interpretation": "Positive improvement means lower held-out MAE than the baseline. The primary comparison uses nested cross-fitted predictions when available; the confidence interval resamples whole regions when at least two regions are available.",
        },
        "preprocessing": {
            "imputation": "Median imputation is fitted independently inside each training fold; no full-data median is used during evaluation.",
            "scaling": "StandardScaler is fitted inside each training fold for linear and Ridge regression.",
            "feature_columns": list(X.columns),
            "feature_missing_fraction": {c: float(X[c].isna().mean()) for c in X.columns},
            "features_entirely_missing_dropped": dropped_all_missing,
        },
        "feature_importance_method": f"Random-forest permutation importance on held-out folds (scoring=R² on log1p target; {permutation_repeats} repeats per fold); standardised linear coefficients averaged across folds.",
        "interval_method": (
            "Fold-specific empirical residual band: calibration width is computed inside each outer training fold from inner OOF residuals, using the row-wise maximum absolute error across candidate models; evaluated on outer held-out rows."
            if nested_available else
            "Unavailable: nested spatial validation did not run, so a training-fold-calibrated band cannot be estimated without reusing evaluation residuals."
        ),
        "model_parameters": {
            "outer_cv_splits": outer_splits, "nested_cv_splits": inner_splits,
            "interval_alpha": interval_alpha, "bootstrap_repeats": n_bootstrap,
            "permutation_repeats": permutation_repeats,
            "ridge_alpha": float(model_config.get("ridge_alpha", 1.0)),
            "random_forest": {"n_estimators": int(model_config.get("random_forest", {}).get("n_estimators", 300)),
                              "min_samples_leaf": int(model_config.get("random_forest", {}).get("min_samples_leaf", 5)),
                              "n_jobs": int(model_config.get("random_forest", {}).get("n_jobs", 1))},
            "gradient_boosting": {"max_iter": int(model_config.get("gradient_boosting", {}).get("max_iter", 300)),
                                  "learning_rate": float(model_config.get("gradient_boosting", {}).get("learning_rate", 0.05)),
                                  "min_samples_leaf": int(model_config.get("gradient_boosting", {}).get("min_samples_leaf", 20)),
                                  "l2_regularization": float(model_config.get("gradient_boosting", {}).get("l2_regularization", 1.0))},
        },
        "interval_formal_conformal_guarantee": False,
        "interval_limitation": (
            "Bands use training-fold inner out-of-fold residuals and are evaluated only on the corresponding outer held-out rows, so each row's own outcome is not used to set its width. They are empirical spatial-validation bands, not a formal distribution-free conformal guarantee; geographic dependence, region heterogeneity, model selection, and limited regional sample size may affect coverage."
            if nested_available else
            "No interval is emitted because nested spatial validation was unavailable. Reusing pooled OOF residuals from the same rows to form and assess an interval would make the coverage diagnostic optimistic."
        ),
    }
    metrics["conformal"] = {}  # preserved legacy namespace for the existing dashboard
    if nested_available:
        lo = np.maximum(0.0, np.expm1(final_pred - nested_halfwidth))
        hi = np.maximum(0.0, np.expm1(final_pred + nested_halfwidth))
        actual = np.expm1(y)
        interval_ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(final_pred)
        covered = interval_ok & (actual >= lo) & (actual <= hi)
        if interval_ok.any():
            region_coverage = pd.Series(covered[interval_ok]).groupby(pd.Series(groups[interval_ok])).mean()
            coverage_value = float(covered[interval_ok].mean())
            below = int(((actual < lo) & interval_ok).sum())
            above = int(((actual > hi) & interval_ok).sum())
            widths = hi[interval_ok] - lo[interval_ok]
            band_info = {
                "level": 1 - interval_alpha,
                "empirical_coverage_on_outer_heldout_rows": coverage_value,
                "empirical_coverage": coverage_value,  # backward-compatible alias; now outer-held-out coverage
                "n_outer_heldout_rows_with_interval": int(interval_ok.sum()),
                "below_interval": below, "above_interval": above,
                "region_coverage_min": float(region_coverage.min()),
                "region_coverage_max": float(region_coverage.max()),
                "median_width_beds_per_1000": float(np.median(widths)),
                "fold_calibration": fold_calibration_rows,
                "calibration_scope": "inner OOF residuals restricted to each outer training fold; held-out outer rows used only for coverage audit",
                "method": metrics["interval_method"], "formal_guarantee": False,
                "limitation": metrics["interval_limitation"],
            }
        else:
            band_info = {"level": 1 - interval_alpha, "available": False,
                         "method": metrics["interval_method"], "formal_guarantee": False,
                         "limitation": "No folds had enough inner OOF calibration rows to estimate a band."}
        lo_out, hi_out = lo, hi
    else:
        lo_out = np.full(len(data), np.nan)
        hi_out = np.full(len(data), np.nan)
        band_info = {"level": 1 - interval_alpha, "available": False,
                     "method": metrics["interval_method"], "formal_guarantee": False,
                     "limitation": metrics["interval_limitation"]}
    metrics["prediction_band"] = band_info
    metrics["conformal"] = band_info  # backwards-compatible key; wording is corrected in the UI

    # Test whether held-out errors retain spatial structure. Failure is reported,
    # not silently turned into a non-spatial model claim.
    if {"lat", "lon"}.issubset(data.columns) and len(data) >= 7:
        try:
            metrics["residual_moran"] = residual_moran(data, y - final_pred, seed=seed)
        except Exception as exc:  # retain model output if optional spatial diagnostic fails
            metrics["residual_moran"] = {"available": False, "reason": f"Moran diagnostic unavailable: {type(exc).__name__}: {exc}"}

    metrics["group_performance"] = _region_error_table(data, y, final_pred, groups, nested_model)
    metrics["subgroup_performance"] = _subgroup_error_table(data, y, final_pred, groups)
    if unique_groups >= 2:
        # Compact region-level error bands and selection-adjusted prediction diagnostics.
        metrics["group_performance_note"] = "Errors are reported by held-out region; between-region heterogeneity should be examined before national-level claims."
    else:
        metrics["group_performance_note"] = "Only one region is present; spatial subgroup validation is not possible."

    out = pd.DataFrame({
        "psgc": data["psgc"].astype(str).to_numpy() if "psgc" in data else data.index.astype(str),
        "expected_beds_per_1000": np.maximum(0.0, np.expm1(final_pred)),
        "expected_lo90": lo_out, "expected_hi90": hi_out,
        "actual_beds_per_1000": actual,
        "model_used": nested_model,
        "validation_fold": fold_number,
        "prediction_residual_log": y - final_pred,
    })
    out["capacity_vs_expected"] = out["actual_beds_per_1000"] - out["expected_beds_per_1000"]
    out["capacity_ratio"] = (out["actual_beds_per_1000"] + 0.05) / (out["expected_beds_per_1000"] + 0.05)
    out["prediction_error_beds"] = out["expected_beds_per_1000"] - out["actual_beds_per_1000"]
    out["abs_error_beds"] = out["prediction_error_beds"].abs()
    out["relative_error_pct"] = 100 * out["prediction_error_beds"] / out["actual_beds_per_1000"].replace(0, np.nan)
    out["below_expected_90"] = out["expected_lo90"].notna() & (out["actual_beds_per_1000"] < out["expected_lo90"])
    # Extra candidate OOF predictions make independent row-level auditing possible.
    for key in model_keys:
        out[f"cv_pred_{key}_beds_per_1000"] = np.maximum(0.0, np.expm1(oof[key]))
        out[f"cv_error_{key}_beds"] = out[f"cv_pred_{key}_beds_per_1000"] - out["actual_beds_per_1000"]

    imp_mean = np.mean(np.vstack(importances), axis=0) if importances else np.zeros(X.shape[1])
    imp_std = np.std(np.vstack(importances), axis=0, ddof=1) if len(importances) > 1 else np.zeros(X.shape[1])
    coef_mean = np.mean(np.vstack(coef_rows), axis=0) if coef_rows else np.zeros(X.shape[1])
    imp = pd.DataFrame({
        "feature": X.columns,
        "label": [FEATURES.get(c, c) for c in X.columns],
        "linear_coef_std": coef_mean,
        "rf_importance": imp_mean,
        "rf_importance_std": imp_std,
        "importance_method": "held-out permutation importance averaged across region-blocked folds",
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    # Compact fold-level evaluation table; the existing return signature stays unchanged.
    metrics["region_metrics_rows"] = fold_region_metrics
    metrics["validation_fold_column"] = "validation_fold"
    metrics["prediction_model_column"] = "model_used"
    # Legacy direct keys are used by the pipeline logger and existing tests.
    for key, value in model_metrics.items():
        metrics[key] = value
    return out, metrics, imp


def choose_k(Z: np.ndarray, ks=range(3, 9), seed: int = 42) -> tuple[int, dict]:
    scores = {}
    choices = [int(k) for k in ks if 2 <= int(k) < len(Z)]
    if not choices:
        raise ValueError("Typology needs at least three LGUs to consider clusters")
    for k in choices:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        if np.unique(lab).size > 1:
            scores[k] = float(silhouette_score(Z, lab, sample_size=min(len(Z), 2000), random_state=seed))
    if not scores:
        raise ValueError("Could not compute a valid silhouette score for the selected LGU typology")
    return max(scores, key=scores.get), scores


def cluster_stability(Z: np.ndarray, k: int, base: np.ndarray, n_boot: int = 30, seed: int = 42) -> float:
    """Mean adjusted Rand index between base clusters and bootstrap-refitted labels."""
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=seed + b).fit(Z[idx])
        aris.append(adjusted_rand_score(base, km.predict(Z)))
    return float(np.mean(aris))


def typology(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    cols = [c for c in df.columns if c.startswith(("hvs_", "hcs_")) and not c.endswith("_pct")]
    if len(df) < 4 or not cols:
        raise ValueError("Typology requires at least four LGUs and one HVS/HCS domain score")
    Zdf = df[cols].apply(pd.to_numeric, errors="coerce")
    Zdf = Zdf.loc[:, Zdf.notna().any()]
    if Zdf.shape[1] == 0:
        raise ValueError("No observed HVS/HCS domain score is available for typology")
    Zdf = Zdf.fillna(Zdf.median()).fillna(0)
    Z = StandardScaler().fit_transform(Zdf)
    k, sil = choose_k(Z, seed=seed)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    lab = pd.Series(km.labels_, index=df.index)
    stab = cluster_stability(Z, k, km.labels_, seed=seed)
    cols = list(Zdf.columns)
    prof = df[cols].groupby(lab).mean()
    need_cols = [c for c in cols if c.startswith("hvs_")]
    cap_cols = [c for c in cols if c.startswith("hcs_")]
    need = prof[need_cols].mean(axis=1) if need_cols else pd.Series(0.0, index=prof.index)
    cap = prof[cap_cols].mean(axis=1) if cap_cols else pd.Series(0.0, index=prof.index)

    def name(i):
        n = "High need" if need[i] >= need.quantile(0.6) else ("Low need" if need[i] <= need.quantile(0.4) else "Moderate need")
        c = "low capacity" if cap[i] <= cap.quantile(0.4) else ("high capacity" if cap[i] >= cap.quantile(0.6) else "moderate capacity")
        weak_cols = [c_ for c_ in cols if c_.startswith("hcs_")]
        weakest = (prof.loc[i, weak_cols].idxmin().replace("hcs_", "").replace("_", " ") if weak_cols else "not available")
        return f"{n} · {c} (weakest: {weakest})"

    names = {i: name(i) for i in prof.index}
    info = {"k": int(k), "silhouette": {str(a): b for a, b in sil.items()}, "stability_ari": stab,
            "features": cols, "feature_missingness_handling": "median imputation before clustering; descriptive typology, not a predictive target"}
    return pd.DataFrame({"psgc": df["psgc"].astype(str).to_numpy(), "cluster": lab.to_numpy(),
                         "cluster_name": lab.map(names).to_numpy()}), info


def run_ml(res: pd.DataFrame, out_dir, seed: int = 42, model_config: dict | None = None) -> dict:
    """Run ML analyses and write backward-compatible plus additive audit artefacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, metrics, imp = expected_capacity(res, seed, model_config)
    typ, tinfo = typology(res, seed=seed)
    metrics["typology"] = tinfo
    pred.merge(typ, on="psgc").to_csv(out_dir / "ml_predictions.csv", index=False)
    imp.to_csv(out_dir / "ml_importance.csv", index=False)

    region_rows = metrics.pop("region_metrics_rows", [])
    pd.DataFrame(region_rows).to_csv(out_dir / "ml_region_metrics.csv", index=False)
    pd.DataFrame(metrics.get("subgroup_performance", [])).to_csv(out_dir / "ml_subgroup_metrics.csv", index=False)
    # The original metric aliases (metrics[model_key]) remain for existing consumers.
    for key, value in metrics["models"].items():
        metrics[key] = value
    (out_dir / "ml_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))

    # A concise interpretation/limitations artefact for audit and hand-off.
    nested = metrics.get("nested_selection", {})
    lines = [
        "# Machine-learning validation and diagnostics", "",
        f"- Target: {metrics.get('target_definition', metrics.get('target', 'not specified'))}",
        f"- Observations: {metrics.get('n', 0):,}; distinct regions: {metrics.get('n_regions', 0)}.",
        f"- Candidate validation: {metrics.get('cv', 'not recorded')}.",
        f"- Top single candidate by pooled outer-fold log-scale R² (exploratory; selection can be optimistic): {metrics.get('best', 'not recorded')}.",
        f"- Nested model-selection evaluation available: {nested.get('available', False)}.",
        f"- Nested model-selection procedure: {nested.get('method', 'not recorded')}.",
        f"- Nested-selection MAE: {nested.get('mae_beds', float('nan')):.4f} beds per 1,000.",
        f"- Nested-selection RMSE: {nested.get('rmse_beds', float('nan')):.4f} beds per 1,000.",
        f"- Baseline MAE: {metrics.get('baseline_comparison', {}).get('baseline_mae_beds', float('nan')):.4f}; comparison MAE: {metrics.get('baseline_comparison', {}).get('selected_model_mae_beds', float('nan')):.4f} beds per 1,000.",
        f"- Empirical coverage on outer held-out rows: {metrics.get('prediction_band', {}).get('empirical_coverage_on_outer_heldout_rows', float('nan')):.1%}.",
        f"- Prediction interval status: {metrics.get('prediction_band', {}).get('limitation', 'not recorded')}",
        "- Interval warning: intervals are training-fold calibrated empirical spatial-validation bands, not formal distribution-free conformal intervals.",
        "- Preprocessing: feature imputation and scaling are fitted inside validation training folds.",
        "- Feature importance: random-forest permutation importance is calculated on held-out fold rows.",
        "- Audit files: `ml_predictions.csv`, `ml_importance.csv`, `ml_region_metrics.csv`, and `ml_subgroup_metrics.csv`.",
        "", "## Interpretation limits", "",
        "The target is produced by the application's E2SFCA accessibility pipeline, so this model predicts a derived spatial-accessibility indicator. It does not independently validate staffed beds, patient utilisation, health outcomes, or causality. Region-blocked CV assesses transfer to held-out regional groups, but should not be mistaken for prospective validation. Group-specific errors and travel-time-source differences should be reviewed before policy use.",
        "",
    ]
    (out_dir / "ml_validation.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics
