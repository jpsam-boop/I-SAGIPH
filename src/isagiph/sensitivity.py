"""Monte Carlo uncertainty analysis of the Healthcare Gap Index.

Each run perturbs four modelling choices at once: weighting scheme (factor-analysis
vs. equal), domain weights (Dirichlet around equal), normalisation method, and catchment size (d0). We then
report each LGU's median rank, 90% rank interval, and how often it lands in the
priority class. LGUs whose priority status flips in more than the configured
share of runs are flagged as unstable rather than presented as firm findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indices import available_domains, build_indices


def run_sensitivity(base: pd.DataFrame, access_by_mult: dict[float, pd.DataFrame],
                    cfg: dict, n_runs: int | None = None) -> pd.DataFrame:
    sc = cfg["sensitivity"]
    n_runs = n_runs or sc["n_runs"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    mults = list(access_by_mult)
    merged = {m: base.merge(a, on="psgc", how="left") for m, a in access_by_mult.items()}
    sample = merged[mults[0]]
    hvs_domains = list(available_domains(sample, cfg["indicators"]["hvs"]))
    hcs_domains = list(available_domains(sample, cfg["indicators"]["hcs"]))

    ranks = np.zeros((n_runs, len(base)))
    prio = np.zeros((n_runs, len(base)), bool)
    for r in range(n_runs):
        m = mults[rng.integers(len(mults))]
        method = sc["methods"][rng.integers(len(sc["methods"]))]
        hw = dict(zip(hvs_domains, rng.dirichlet([sc["dirichlet_alpha"]] * len(hvs_domains))))
        cw = dict(zip(hcs_domains, rng.dirichlet([sc["dirichlet_alpha"]] * len(hcs_domains))))
        weighting = sc.get("weightings", ["equal"])[rng.integers(len(sc.get("weightings", ["equal"])))]
        res = build_indices(merged[m], cfg, method=method, hvs_weights=hw, hcs_weights=cw, weighting=weighting)
        ranks[r] = res["gap_index"].rank(ascending=False, method="min").to_numpy()
        prio[r] = res["priority"].to_numpy()

    out = pd.DataFrame({"psgc": base["psgc"].to_numpy()})
    out["rank_median"] = np.median(ranks, axis=0)
    out["rank_p05"] = np.percentile(ranks, 5, axis=0)
    out["rank_p95"] = np.percentile(ranks, 95, axis=0)
    out["priority_freq"] = prio.mean(axis=0)
    flip = np.minimum(out["priority_freq"], 1 - out["priority_freq"])
    out["unstable"] = flip > sc["instability_threshold"]
    return out
