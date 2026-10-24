"""Equity statistics: how hospital access and the healthcare gap are distributed across
poverty levels, isolation, and age structure (population-weighted)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_quintile(x: pd.Series, w: pd.Series, labels=("Q1 poorest", "Q2", "Q3", "Q4", "Q5 least poor"),
                      descending: bool = True) -> pd.Series:
    """Population-weighted quintiles (Q1 = highest x when descending, e.g. highest poverty)."""
    order = x.sort_values(ascending=not descending).index
    cw = w.loc[order].cumsum() / w.sum()
    q = np.minimum((cw * 5).apply(np.ceil).astype(int), 5).clip(lower=1)
    return pd.Series(pd.Categorical([labels[i - 1] for i in q], categories=labels, ordered=True), index=order).reindex(x.index)


def concentration_curve(x_rank: pd.Series, h: pd.Series, w: pd.Series) -> pd.DataFrame:
    """Cumulative share of population (poorest first) vs cumulative share of a good (e.g. accessible beds)."""
    order = x_rank.sort_values(ascending=False).index          # highest poverty first
    pop = w.loc[order]
    good = (h.loc[order] * pop)
    return pd.DataFrame({"cum_pop": np.r_[0, pop.cumsum() / pop.sum()],
                         "cum_good": np.r_[0, good.cumsum() / good.sum()]})


def concentration_index(x_rank: pd.Series, h: pd.Series, w: pd.Series) -> float:
    """Population-weighted concentration index of h by socioeconomic rank (poorest = lowest rank).

    Negative = the good is concentrated among poorer LGUs; positive = among less-poor LGUs.
    """
    order = x_rank.sort_values(ascending=False).index          # poorest first -> lowest rank
    wt = w.loc[order] / w.sum()
    hv = h.loc[order].astype(float)
    r = wt.cumsum() - wt / 2                                    # fractional rank
    mu = (wt * hv).sum()
    return float(2 * (wt * (hv - mu) * (r - (wt * r).sum())).sum() / mu)


def group_table(df: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    w = df["population"]
    g = df.assign(_g=group, _w=w)

    def agg(d):
        ww = d["_w"]
        return pd.Series({
            "LGUs": len(d), "people (M)": ww.sum() / 1e6,
            "accessible beds per 1,000": 1000 * d["accessible_beds"].sum() / ww.sum(),
            "% ≤30 min to a hospital": 100 * (d["cov_any_30"] * ww).sum() / ww.sum(),
            "% ≤60 min to Malasakit": 100 * (d["cov_malasakit_60"] * ww).sum() / ww.sum(),
            "mean HGI (pop-weighted)": (d["gap_index"] * ww).sum() / ww.sum(),
            "% of people in priority LGUs": 100 * (d["priority"] * ww).sum() / ww.sum(),
        })
    return g.groupby("_g", observed=True).apply(agg, include_groups=False)
