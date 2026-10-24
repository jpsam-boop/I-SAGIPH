"""Empirical Bayes rate smoothing (Marshall, 1991).

Raw incidence rates in small LGUs are unstable (one case in 2,000 people looks
like an outbreak). Global or grouped Empirical Bayes shrinks each rate toward the
prior mean in proportion to how little population supports it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def eb_rates(cases: pd.Series, pop: pd.Series) -> pd.Series:
    """Method-of-moments Empirical Bayes estimate of y_i / n_i."""
    y = cases.astype(float).to_numpy()
    n = pop.astype(float).to_numpy()
    r = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
    b = y.sum() / n.sum()                          # prior mean
    s2 = np.sum(n * (r - b) ** 2) / n.sum()        # population-weighted variance
    a = max(s2 - b / n.mean(), 0.0)                # prior variance (truncated at 0)
    w = np.divide(a, a + b / n, out=np.zeros_like(n), where=n > 0)
    return pd.Series(w * r + (1 - w) * b, index=cases.index)


def eb_rates_grouped(df: pd.DataFrame, cases_col: str, pop_col: str,
                     group_col: str | None, per: float = 100000) -> pd.Series:
    """EB rates with the prior estimated within each group (e.g. region)."""
    if group_col is None:
        return eb_rates(df[cases_col], df[pop_col]) * per
    parts = [eb_rates(g[cases_col], g[pop_col]) for _, g in df.groupby(group_col)]
    return pd.concat(parts).reindex(df.index) * per
