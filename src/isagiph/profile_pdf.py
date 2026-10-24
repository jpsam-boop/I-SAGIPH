"""One-page PDF profile of a city or municipality (downloaded from the LGU Profile page)."""
from __future__ import annotations

import datetime as dt
import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAVY, RED, BLUE, YELLOW, GREY = "#002157", "#CE3550", "#4F7FBE", "#F5C400", "#5A6B85"
DOMAINS = {"hvs_demographic": "Age dependency", "hvs_socioeconomic": "Poverty", "hvs_geographic": "Isolation (GIDA)",
           "hvs_disease": "Disease burden", "hcs_beds": "Beds within reach", "hcs_coverage": "Hospitals within 30–60 min",
           "hcs_primary_care": "Primary care", "hcs_social_protection": "Malasakit reach"}
AGES = [("pop_children", "Children 0–14", RED), ("pop_youth", "Youth 15–24", YELLOW),
        ("pop_young_adults", "Young adults 25–44", BLUE), ("pop_middle_age", "Middle age 45–59", "#9AA6B8"),
        ("pop_elderly", "Elderly 60+", NAVY)]


def _lc(label: str) -> str:
    """Lower-case a label inside a sentence, keeping proper nouns (Malasakit) and acronyms intact."""
    return label if label.startswith("Malasakit") else label[0].lower() + label[1:]


def _fmt_km(v) -> str:
    return "—" if v is None or pd.isna(v) else ("<1 km" if v < 1 else f"{v:,.0f} km")


def _pct(series: pd.Series, value: float) -> float:
    return float((series < value).mean() * 100)


def _box(fig, x, y, w, h, title, value, note, color):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor="#E3E8EF", lw=1))
    ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, facecolor=color, edgecolor="none"))
    ax.text(0.07, 0.78, title.upper(), fontsize=6.8, color=GREY, fontweight="bold", transform=ax.transAxes)
    ax.text(0.07, 0.36, value, fontsize=19, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.07, 0.1, note, fontsize=6.6, color=GREY, transform=ax.transAxes)


def lgu_profile_pdf(r: pd.Series, res: pd.DataFrame, logo: Path | None = None, benchmark: float = 1.25) -> bytes:
    """Return a one-page A4 PDF for LGU row ``r`` (``res`` is the national table, for percentiles)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("#FBF8F1")

    # ---- header
    if logo is not None and Path(logo).exists():
        lax = fig.add_axes([0.06, 0.915, 0.07, 0.06]); lax.imshow(plt.imread(logo)); lax.axis("off")
    x = 0.15                                       # I- (black) SAGI (blue) PH (red), placed edge to edge
    rend = fig.canvas.get_renderer()
    for part, col in (("I-", "#111111"), ("SAGI", BLUE), ("PH", RED)):
        t = fig.text(x, 0.955, part, fontsize=15, fontweight="bold", color=col)
        x += t.get_window_extent(renderer=rend).width / fig.bbox.width
    fig.text(0.15, 0.935, "LGU health access profile", fontsize=8.5, color=GREY)
    fig.text(0.94, 0.955, dt.date.today().strftime("%d %B %Y"), fontsize=8, color=GREY, ha="right")
    fig.text(0.06, 0.885, str(r["name"]), fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.06, 0.862, f"{r['province']} · {r['region']} · {r['population']:,.0f} people (2020 census)",
             fontsize=9.5, color=GREY)

    # ---- index boxes
    pri = f"Priority area · {100 * r.get('priority_freq', 0):.0f}% of model runs" if r["priority"] else \
        f"Priority in {100 * r.get('priority_freq', 0):.0f}% of model runs"
    _box(fig, 0.06, 0.765, 0.27, 0.075, "Healthcare Gap Index", f"{r['gap_index']:+.0f}", pri, "#7A4BB0")
    _box(fig, 0.365, 0.765, 0.27, 0.075, "Health Need Index", f"{r['HVS']:.0f}",
         f"Higher need than {r['hvs_pct']:.0f}% of LGUs", "#d6604d")
    _box(fig, 0.67, 0.765, 0.27, 0.075, "Hospital Capacity Index", f"{r['HCS']:.0f}",
         f"More than {r['hcs_pct']:.0f}% of LGUs (relative, not adequacy)", BLUE)

    # ---- plain-language reading
    tier = {3: "top third", 2: "middle third", 1: "bottom third"}
    doms = {k: r[k] for k in DOMAINS if k in r.index and pd.notna(r[k])}
    need = sorted([k for k in doms if k.startswith("hvs_")], key=lambda k: -doms[k])[:2]
    cap = sorted([k for k in doms if k.startswith("hcs_")], key=lambda k: doms[k])[:2]
    text = (f"{r['name']} is in the {tier.get(r['v_tertile'], '')} of Philippine cities and municipalities for health "
            f"need and the {tier.get(r['c_tertile'], '')} for hospital capacity"
            + (", which makes it a priority area (high need, low capacity)" if r["priority"] else "")
            + f". Need is driven most by {' and '.join(_lc(DOMAINS[k]) for k in need)}; the weakest parts of "
            f"capacity are {' and '.join(_lc(DOMAINS[k]) for k in cap)}. "
            + ("This result held across most of 1,000 model variations." if not r.get("unstable", False)
               else "This result depends on modelling choices, so it should be verified locally."))
    fig.text(0.06, 0.745, "\n".join(textwrap.wrap(text, 118)), fontsize=8.6, color="#16213A", va="top", linespacing=1.5)

    # ---- domain scores
    ax = fig.add_axes([0.24, 0.455, 0.27, 0.2])
    ax.set_facecolor("none")                      # no white box behind the bars
    keys = [k for k in DOMAINS if k in doms]
    vals = [doms[k] for k in keys]
    colors = [("#d6604d" if k.startswith("hvs_") else BLUE) for k in keys]
    ax.barh(range(len(keys)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([DOMAINS[k] for k in keys], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.tick_params(axis="x", labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("What drives the scores (0–100)", fontsize=9, color=NAVY, loc="left", fontweight="bold")
    ax.text(0, -0.17, "Red = need factors (higher is worse) · Blue = capacity factors (higher is better)",
            fontsize=6.5, color=GREY, transform=ax.transAxes)

    # ---- age pie
    pax = fig.add_axes([0.58, 0.455, 0.2, 0.2])
    pax.set_facecolor("none")
    if all(c in r.index for c, _, _ in AGES):
        sizes = [max(r[c], 0) for c, _, _ in AGES]
        pax.pie(sizes, colors=[c for _, _, c in AGES], startangle=90, counterclock=False,
                wedgeprops={"width": 0.38, "edgecolor": "white"})
        for i, (c, lab, col) in enumerate(AGES):
            fig.text(0.79, 0.62 - i * 0.025, f"■ {lab}  {100 * sizes[i] / max(sum(sizes), 1):.0f}%", fontsize=7.3, color=col)
    pax.set_title("Who lives here", fontsize=9, color=NAVY, loc="left", fontweight="bold")

    # ---- key indicators table
    rows = [
        ("Poverty incidence (2023)", f"{100 * r['poverty_incidence']:.1f}%" + (" (imputed)" if r.get("poverty_imputed", 0) else "")),
        ("Share of GIDA barangays", f"{100 * r.get('gida_share', 0):.0f}%"),
        ("Tuberculosis per 100,000", f"{r.get('rate_tb', np.nan):,.1f}"),
        ("Pneumonia per 100,000", f"{r.get('rate_pneumonia', np.nan):,.1f}"),
        ("Hypertension per 100,000", f"{r.get('rate_hypertension', np.nan):,.1f}"),
        (f"Beds within reach per 1,000 (DOH target {benchmark:g}; 2040 goal 4)", f"{r['beds_per_1000_all']:.2f}"),
        ("Residents within 30 min of a hospital (" + ("by road" if float(r.get("road_share", 0) or 0) >= 0.5
                                                      else "straight-line estimate") + ")", f"{100 * r['cov_any_30']:.0f}%"),
        ("Residents within 60 min of a Malasakit Center", f"{100 * r['cov_malasakit_60']:.0f}%"),
        ("Nearest hospital (straight line)", _fmt_km(r.get("km_hospital"))),
        ("Nearest dialysis (straight line)", _fmt_km(r.get("km_dialysis"))),
        ("Ambulances in the LGU", f"{r.get('ambulances', 0):.0f}"),
        ("Licensed hospital beds in the LGU", f"{r.get('beds_public', 0) + r.get('beds_private', 0):,.0f}"),
        (f"Bed shortfall vs DOH target ({benchmark:g} per 1,000)", f"{r['bed_gap_shortfall']:,.0f} beds"),
        ("Bed shortfall vs 2040 goal (4 per 1,000)", f"{r.get('bed_gap_2040', np.nan):,.0f} beds"),
    ]
    tax = fig.add_axes([0.06, 0.1, 0.88, 0.31]); tax.axis("off")
    tax.text(0, 1.0, "Key indicators", fontsize=9.5, color=NAVY, fontweight="bold", transform=tax.transAxes, va="top")
    for i, (k, v) in enumerate(rows):
        y = 0.92 - i * 0.07
        if i % 2 == 0:
            tax.add_patch(plt.Rectangle((0, y - 0.045), 1, 0.07, transform=tax.transAxes, facecolor="#EEF1F6", edgecolor="none"))
        tax.text(0.01, y - 0.01, k, fontsize=8.3, color="#16213A", transform=tax.transAxes, va="center")
        tax.text(0.99, y - 0.01, v, fontsize=8.3, color="#16213A", transform=tax.transAxes, va="center", ha="right",
                 fontweight="bold")

    # ---- footer
    fig.text(0.06, 0.06, "\n".join(textwrap.wrap(
        "Sources: PSA 2020 Census and 2023 poverty estimates; DOH licensed facilities (June 2026), GIDA list (2023) and "
        "FHSIS morbidity (2022); PCSO Malasakit Centers (2022); OCHA COD-AB boundaries; OpenStreetMap. Indices are "
        "relative rankings among 1,634 cities and municipalities; distances are straight-line. Findings show where to "
        "look, not what to decide.", 140)), fontsize=6.6, color=GREY, va="top", linespacing=1.4)
    fig.text(0.94, 0.025, "I-SAGIPH", fontsize=7, color=GREY, ha="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
