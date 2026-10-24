"""Write a Markdown findings report from pipeline outputs (reports/findings.md)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    d = df.copy()
    for c in d.select_dtypes("number"):
        is_int = d[c].dropna().apply(lambda v: float(v).is_integer()).all()
        fmt = "{:,.0f}" if is_int else floatfmt
        d[c] = d[c].map(lambda v: fmt.format(v) if pd.notna(v) else "")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.to_numpy())


ACTIONS = {
    "Distance": "Bring care closer: a new Level 1 hospital or infirmary, referral transport, or telemedicine",
    "Bed supply": "Add beds where people already travel: expand or upgrade an existing hospital (Level 1 to Level 2)",
}


def bottlenecks(res: pd.DataFrame, bed_target: float) -> pd.DataFrame:
    """Main binding constraint for each priority LGU, plus flags for financial protection and primary care.

    Almost every priority area is below the DOH bed minimum, so what distinguishes them is distance.
    Distance: under half the residents are within 30 min of a hospital (new nearby capacity would help most).
    Bed supply: most residents are within reach, so the hospitals they can reach are too small for them.
    """
    pri = res[res["priority"]].copy()
    pri["bottleneck"] = np.where(pri["cov_any_30"] < 0.5, "Distance", "Bed supply")
    pri["no_malasakit_within_60"] = pri["cov_malasakit_60"] < 0.5
    pri["weak_primary_care"] = pri.get("hcs_primary_care", pd.Series(100, index=pri.index)) < 20
    pri["action"] = pri["bottleneck"].map(ACTIONS).fillna("Verify locally: no single binding constraint")
    return pri


def concentration_ci(res: pd.DataFrame, col: str = "beds_per_1000_all", n_boot: int = 1000, seed: int = 0):
    """Concentration index of `col` by poverty rank, with a bootstrap 95% CI (LGUs resampled with replacement)."""
    from .equity import concentration_index
    rank = res["poverty_incidence"]                        # concentration_index ranks the poorest first itself
    est = concentration_index(rank, res[col], res["population"])
    rng = np.random.default_rng(seed)
    n, boots = len(res), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(concentration_index(rank.iloc[i].reset_index(drop=True), res[col].iloc[i].reset_index(drop=True),
                                         res["population"].iloc[i].reset_index(drop=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def write_policy_brief(res: pd.DataFrame, meta: dict, path) -> None:
    """One-page brief for regional and provincial health offices: where to act first, and how."""
    pri = bottlenecks(res, meta["beds_per_1000"])
    stable = pri[~pri["unstable"]]
    P = res["population"].sum()
    out = ["# I-SAGIPH policy brief: where to act first\n"]
    if meta["mode"] == "demo":
        out.append("> **SYNTHETIC DEMO DATA.** Not findings about the Philippines.\n")
    out += [
        f"**{len(stable):,} cities and municipalities** are priority areas (high health need, low hospital capacity) "
        f"in at least {100 * (1 - meta['instability_threshold']):.0f}% of 1,000 model variations, home to "
        f"**{stable['population'].sum() / 1e6:,.1f} million people**. Nationally, "
        f"{100 * (1 - (res['population'] * res['cov_any_30']).sum() / P):.0f}% of residents are modelled as more than "
        f"30 minutes from a hospital, and beds within reach average "
        f"{1000 * res['accessible_beds'].sum() / P:.2f} per 1,000 against a DOH minimum of {meta['beds_per_1000']}.\n",
        "## What holds each priority area back\n",
        _table(stable.groupby("bottleneck").agg(areas=("psgc", "count"), people=("population", "sum"),
                                                 beds_short=("bed_gap_shortfall", "sum"))
               .reindex(["Distance", "Bed supply"]).dropna().reset_index()
               .assign(action=lambda d: d["bottleneck"].map(ACTIONS).fillna("Verify locally")), "{:,.0f}"),
        "",
        f"Also: {int(stable['no_malasakit_within_60'].sum()):,} of these areas have most residents more than 60 minutes "
        f"from a Malasakit Center (candidates under R.A. 11463), and {int(stable['weak_primary_care'].sum()):,} have weak "
        "licensed primary care.\n",
        "## Top three per region (ranked by beds needed to reach the DOH minimum)\n",
    ]
    rows = []
    for reg, g in stable.sort_values("bed_gap_shortfall", ascending=False).groupby("region", sort=True):
        for _, r in g.head(3).iterrows():
            rows.append({"region": reg, "city/municipality": f"{r['name']}, {r['province']}",
                         "people": r["population"], "main bottleneck": r["bottleneck"],
                         "beds needed": r["bed_gap_shortfall"],
                         "% within 30 min": 100 * r["cov_any_30"]})
    out.append(_table(pd.DataFrame(rows), "{:,.0f}") if rows else "No stable priority areas.")
    out += ["",
            "**How to read this.** Scores rank areas against each other; beds needed are against the DOH minimum of "
            "1 bed per 800 people. Travel times are modelled under stated assumptions. Treat this list as where to "
            "look first and verify with provincial health offices before committing resources."]
    path.write_text("\n".join(out), encoding="utf-8")


def write_report(res: pd.DataFrame, glob: dict, valid: pd.DataFrame, meta: dict, path) -> None:
    demo = meta["mode"] == "demo"
    pri = res[res["priority"]]
    stable_pri = pri[~pri["unstable"]]
    lines = [f"# I-SAGIPH findings report\n",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}, mode: **{meta['mode']}**, "
             f"travel times: {meta['travel_time_source']}, sensitivity runs: {meta['n_runs']}.\n"]
    if demo:
        lines.append("> **SYNTHETIC DEMO DATA.** These numbers test the pipeline only and are not findings "
                     "about the Philippines.\n")
    lines += [
        "## 1. Headline numbers\n",
        f"- LGUs analysed: {len(res):,}; population covered: {res['population'].sum():,.0f}.",
        f"- Priority LGUs (top Health Need Index tertile and bottom Hospital Capacity Index tertile): {len(pri):,}, "
        f"home to {pri['population'].sum():,.0f} people; {len(stable_pri):,} remain priority in at least "
        f"{100 * (1 - meta['instability_threshold']):.0f}% of sensitivity runs.",
        f"- Accessible-bed shortfall against the DOH target of {meta['beds_per_1000']} beds per 1,000 (1 per 800): "
        f"{res['bed_gap_shortfall'].sum():,.0f} beds nationally, "
        f"{pri['bed_gap_shortfall'].sum():,.0f} of them in priority LGUs.",
        f"- Beds within reach nationally: {1000 * res['accessible_beds'].sum() / res['population'].sum():.2f} per 1,000, "
        f"below the DOH target of {meta['beds_per_1000']} and far below the 2040 goal of {meta.get('goal_2040', 4.0)}. "
        f"No region reaches the 2040 goal; reaching it everywhere would need about "
        f"{res['bed_gap_2040'].sum():,.0f} more beds within reach." if "bed_gap_2040" in res else "",
        f"- **Modelled** residents with no hospital within 30 minutes: "
        f"{(res['population'] * (1 - res['cov_any_30'])).sum():,.0f} "
        f"({100 * (1 - (res['population'] * res['cov_any_30']).sum() / res['population'].sum()):.1f}%). This is an estimate under the configured transport assumptions, not an observed count of people unable to obtain care.\n",
        "**Travel-time interpretation:** road-network estimates use configured road-class speeds plus a short access/egress speed assumption. Where a route is unavailable, the configured generalized effective-speed fallback is used to approximate mixed local travel. The reported number of residents beyond 30 minutes from a hospital is a model estimate; where a DOH reference figure is supplied, compare the values after estimation and check that their definitions and geographic scope align.\n",
        "HNI = Health Need Index, HCI = Hospital Capacity Index, HGI = Healthcare Gap Index.\n",
        "## 2. Is the gap spatially clustered?\n",
        f"Global Moran's I = {glob['I']:.3f} (E[I] = {glob['EI']:.4f}, pseudo-p = {glob['p_sim']:.3f}, "
        f"{meta['permutations']} permutations, KNN k = {meta['knn_k']}). "
        + ("Gaps cluster geographically, so responses can be organised regionally rather than LGU by LGU."
           if glob["p_sim"] < 0.05 else "No significant clustering was detected."),
        "",
        _table(res["lisa_cluster"].value_counts().rename_axis("LISA class (FDR 5%)").reset_index(name="LGUs"), "{:,.0f}"),
        "",
        _table(res["hotspot"].value_counts().rename_axis("Gi* class (FDR 5%)").reset_index(name="LGUs"), "{:,.0f}"),
        "\n## 3. Regional picture\n",
        _table(res.groupby("region")[["priority", "gap_index", "population", "bed_gap_shortfall", "cov_any_30"]].apply(lambda g: pd.Series({
            "LGUs": len(g), "priority LGUs": g["priority"].sum(),
            "pop-weighted HGI": (g["gap_index"] * g["population"]).sum() / g["population"].sum(),
            "bed shortfall": g["bed_gap_shortfall"].sum(),
            "% within 30 min": 100 * (g["cov_any_30"] * g["population"]).sum() / g["population"].sum(),
        })).sort_values("pop-weighted HGI", ascending=False).reset_index(), "{:,.1f}"),
        "\n## 4. Highest-need LGUs (stable priority, ranked by bed shortfall)\n",
        _table(stable_pri.sort_values("bed_gap_shortfall", ascending=False)
               .head(15)[["name", "province", "population", "HVS", "HCS", "gap_index",
                          "bed_gap_shortfall", "cov_any_30", "rank_p05", "rank_p95"]]
               .rename(columns={"HVS": "HNI", "HCS": "HCI", "gap_index": "HGI"}), "{:,.2f}"),
        "\n## 5. External validation\n",
        _table(valid, "{:.3f}") if len(valid) else "No outcome indicators supplied.",
        "\nA consistent result means the gap index moves in the expected direction with an outcome "
        "that was not used to build it.\n",
        "## 6. Equity: who gets the beds?\n",
        (lambda c: f"Concentration index of beds within reach by poverty rank: **{c[0]:+.3f}** (bootstrap 95% CI "
                   f"{c[1]:+.3f} to {c[2]:+.3f}, 1,000 resamples of LGUs). "
                   + ("The interval excludes 0: beds within reach are significantly concentrated in less-poor areas."
                      if c[1] > 0 else "The interval includes 0: no significant inequality by poverty was detected."
                      if c[1] <= 0 <= c[2] else "The interval is below 0: beds favour poorer areas."))(concentration_ci(res))
        if "poverty_incidence" in res and "beds_per_1000_all" in res else "",
        "",
        "## 7. Bottlenecks and suggested actions\n",
        _table(bottlenecks(res, meta["beds_per_1000"]).query("~unstable").groupby("bottleneck")
               .agg(areas=("psgc", "count"), people=("population", "sum"), beds_short=("bed_gap_shortfall", "sum"))
               .reset_index().assign(action=lambda d: d["bottleneck"].map(ACTIONS).fillna("Verify locally")), "{:,.0f}"),
        "",
        "Stable priority areas only. Distance = under half of residents within 30 min of a hospital; Bed supply = most "
        "residents within reach, but too few beds for them. See `reports/policy_brief.md` for the areas in each region.",
        "",
        "- **Relative scores are not adequacy**: a low gap or high capacity score means better than other LGUs, not "
        "enough. Compare beds per 1,000 with the DOH target and the 2040 goal.",
        "- **High vulnerability without Malasakit reach**: candidates for new Malasakit Centers (R.A. 11463).",
        "- **Unstable LGUs**: results depend on modelling choices; verify with provincial health offices "
        "before acting.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    write_policy_brief(res, meta, path.parent / "policy_brief.md")
