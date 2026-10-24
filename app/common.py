"""Shared data, styling and helpers for every I-SAGIPH page."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import branca.colormap as bcm
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "views"))   # pages can import shared panels (e.g. equity_panel)
PROC = ROOT / "data" / "processed"
ASSETS = APP / "assets"

from isagiph.config import load_config  # noqa: E402
from isagiph.maplayers import hex_grid  # noqa: E402

APP_NAME = "I-SAGIPH"
BRAND_HTML = ("<span style='color:#111111'>I-</span><span style='color:#4F7FBE'>SAGI</span>"
              "<span style='color:#CE3550'>PH</span>")
FULL_NAME = "Interactive Software for Analyzing Geographic Inadequacy in Philippine Healthcare"
CFG = load_config()
NEED_INDICATORS = [c for cols in CFG["indicators"]["hvs"].values() for c in cols]

# Palette taken from the logo: navy hexagon, blue magnifier, red / yellow / blue bars.
NAVY, BLUE, RED, YELLOW = "#002157", "#4F7FBE", "#CE3550", "#F5C400"
GAP_STOPS = ["#F6F1FA", "#D9C6EC", "#B08AD6", "#7A4BB0", "#45197A"]      # purple: distinct from need (orange-red); pale = smaller gap, never "enough"
ADEQ_STOPS = ["#A50026", "#E8613C", "#F7B65A", "#D9EAA6", "#1E9E8C"]     # share of the 2040 bed goal reached
for _nm, _stops in (("Gap", GAP_STOPS), ("Adequacy", ADEQ_STOPS)):
    if _nm not in matplotlib.colormaps:
        matplotlib.colormaps.register(matplotlib.colors.LinearSegmentedColormap.from_list(_nm, _stops))
GOAL_2040 = CFG["benchmarks"].get("goal_2040_per_1000", 4.0)
INK, CREAM, BEIGE, MUTED, SOFT = "#16213A", "#FBF8F1", "#F1E6CF", "#5A6B85", "#E8EEF7"

HNI, HCI, HGI = "Health Need Index (HNI)", "Hospital Capacity Index (HCI)", "Healthcare Gap Index (HGI)"
LAYERS = {  # map layers: only the three consolidated indices
    "Healthcare Gap (HGI)": ("gap_index", "Gap",
                             "−100 to +100. Darker purple = need is furthest ahead of capacity compared with other areas. "
                             "Pale areas have a smaller gap than elsewhere, not enough care."),
    "Health Need (HNI)": ("HVS", "OrRd", "0–100, higher = greater health need."),
    "Hospital Capacity (HCI)": ("HCS", "YlGnBu", "0–100, higher = more capacity than other areas (not necessarily enough)."),
    "Bed need met": ("pct_goal", "Adequacy", "Share of the 2040 goal of 4 hospital beds per 1,000 people that is within "
                     "reach. 31% = the DOH minimum of 1.25. Almost every area is far below 100%."),
}
LAYER_HELP = {
    "Bed need met": "Beds within realistic travel time per 1,000 people, as a share of the Philippine Health Facility "
        "Development Plan's 2040 goal of 4 per 1,000. Unlike the three indices, this is not a ranking: it shows how far "
        "each area is from having enough beds, including areas that rank well against others.",
    "Healthcare Gap (HGI)": "Each LGU's need percentile minus its capacity percentile. +60 means its need ranks 60 "
        "percentile points above its capacity. Red areas are where services lag furthest behind need.",
    "Health Need (HNI)": "Combines the share of young children and older adults, poverty incidence, the share of "
        "geographically isolated (GIDA) barangays, and tuberculosis, pneumonia and hypertension rates. Weights come from factor analysis.",
    "Hospital Capacity (HCI)": "Combines hospital beds reachable within realistic travel times (E2SFCA), the share of "
        "people within 30/60 minutes of a hospital, licensed primary care, and Malasakit Center reach.",
}
BIVARIATE = {"V3-C1": "#b2182b", "V3-C2": "#d6604d", "V3-C3": "#f4a582", "V2-C1": "#e08f6c", "V2-C2": "#bdbdbd",
             "V2-C3": "#92c5de", "V1-C1": "#f7d9c4", "V1-C2": "#9ecae1", "V1-C3": "#2166ac"}
INDICATOR_LABELS = {
    "pct_age_0_4": "% aged 0–4", "pct_age_60_plus": "% aged 60+", "poverty_incidence": "Poverty incidence",
    "gida_share": "GIDA share of barangays", "eb_rate_tb": "TB incidence (EB-smoothed)",
    "eb_rate_dengue": "Dengue incidence (EB-smoothed)", "beds_per_1000_all": "Accessible beds per 1,000",
    "beds_per_1000_l2plus": "Accessible Level 2+ beds per 1,000", "cov_any_30": "% ≤30 min to any hospital",
    "cov_l2plus_60": "% ≤60 min to Level 2+ hospital", "primary_care_per_10k": "Licensed primary care per 10,000",
    "cov_malasakit_60": "% ≤60 min to a Malasakit Center",
    "rate_tb": "Tuberculosis per 100,000", "rate_pneumonia": "Pneumonia per 100,000",
    "rate_hypertension": "Hypertension per 100,000",
}
PH_BOUNDS = [[4.4, 116.0], [21.3, 127.0]]


def img_b64(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode()


LOGO_B64 = img_b64("logo_icon.png")
HERO_SVG = (ASSETS / "hero_map.svg").read_text()


# --------------------------------------------------------------------------- data
@st.cache_data
def load():
    res = pd.read_csv(PROC / "lgu_results.csv", dtype={"psgc": str})
    fac = pd.read_csv(PROC / "facilities.csv", dtype={"psgc": str})
    meta = json.loads((PROC / "run_meta.json").read_text())
    try:
        valid = pd.read_csv(PROC / "validation.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        valid = pd.DataFrame()
    try:
        weights = pd.read_csv(PROC / "index_weights.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        weights = pd.DataFrame()
    geo = PROC / "lgu_boundaries.geojson"
    gj = json.loads(geo.read_text()) if geo.exists() else None
    pts = pd.read_csv(PROC / "points.csv", dtype={"psgc": str}) if (PROC / "points.csv").exists() \
        else res[["psgc", "lat", "lon"]]
    res = enrich_access(res, fac)
    res["pct_goal"] = (100 * res["beds_per_1000_all"] / GOAL_2040).clip(0, 100)
    return res, fac, meta, valid, weights, gj, pts


ENRICH_COLS = ["km_hospital", "km_primary", "km_malasakit", "km_dialysis", "ambulances", "ambulances_per_100k",
               "n_hospitals", "beds_public", "beds_private"]


def enrich_access(res: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    """Nearest-facility distances (straight line, from the LGU's population-weighted centre), ambulances,
    and licensed beds by ownership, for hover cards and profiles."""
    from scipy.spatial import cKDTree
    from isagiph.accessibility import to_local_km

    res = res.copy()
    xy = to_local_km(res["lat"].to_numpy(), res["lon"].to_numpy())

    def nearest(df):
        if df is None or len(df) == 0:
            return np.full(len(res), np.nan)
        d, _ = cKDTree(to_local_km(df["lat"].to_numpy(), df["lon"].to_numpy())).query(xy)
        return d

    res["km_hospital"] = nearest(fac)
    dpath = PROC / "directory.csv"
    if dpath.exists():
        d = pd.read_csv(dpath, dtype={"psgc": str})
        d["services"] = d["services"].fillna("")
        res["km_primary"] = nearest(d[d["kind"] == "Primary care"])
        res["km_malasakit"] = nearest(d[d["malasakit"] == 1])
        res["km_dialysis"] = nearest(d[d["services"].str.contains("Dialysis", regex=False)])
        amb = d.groupby("psgc")["ambulances"].sum()
        res["n_hospitals"] = res["psgc"].map(d[d["kind"] == "Hospital"].groupby("psgc").size()).fillna(0).astype(int)
        bedded = d[d["kind"].isin(["Hospital", "Infirmary"])]
        by = bedded.pivot_table(index="psgc", columns="ownership", values="beds", aggfunc="sum").fillna(0)
        res["beds_public"] = res["psgc"].map(by["Government"] if "Government" in by else pd.Series(dtype=float)).fillna(0)
        res["beds_private"] = res["psgc"].map(by["Private"] if "Private" in by else pd.Series(dtype=float)).fillna(0)
    else:
        amb = pd.Series(dtype=float)
        for c in ("km_primary", "km_malasakit", "km_dialysis"):
            res[c] = np.nan
        res["n_hospitals"] = res["psgc"].map(fac.groupby("psgc").size()).fillna(0).astype(int)
        res["beds_public"] = res["psgc"].map(fac.groupby("psgc")["beds"].sum()).fillna(0)
        res["beds_private"] = 0.0
    free = res["n_ambulance_units"] if "n_ambulance_units" in res else 0
    res["ambulances"] = res["psgc"].map(amb).fillna(0) + free
    res["ambulances_per_100k"] = res["ambulances"] / res["population"] * 1e5
    return res


def require_data():
    if not (PROC / "lgu_results.csv").exists():
        st.error("Results are not available.")
        st.stop()
    data = load()
    if data[2].get("mode") == "demo":
        st.warning("Synthetic demo data: numbers illustrate the method only and are not findings about the Philippines.")
    return data


@st.cache_data
def cached_hexes(lat: tuple, lon: tuple, radius: float):
    return hex_grid(np.array(lat), np.array(lon), radius)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def scale_for(series: pd.Series, cmap: str, caption: str):
    lo, hi = series.quantile([0.02, 0.98])
    if cmap in ("RdBu_r", "RdYlBu_r", "Gap"):   # gap scales are symmetric around 0
        m = max(abs(lo), abs(hi)); lo, hi = -m, m
    if cmap == "Adequacy":                       # absolute scale: 0–100% of the 2040 goal, never stretched
        lo, hi = 0.0, 100.0
    norm = matplotlib.colors.Normalize(lo, hi, clip=True)
    cm = matplotlib.colormaps[cmap]
    legend = bcm.LinearColormap([matplotlib.colors.to_hex(cm(x)) for x in np.linspace(0, 1, 9)],
                                vmin=float(lo), vmax=float(hi), caption=caption)
    return (lambda v: matplotlib.colors.to_hex(cm(norm(v))) if pd.notna(v) else "#bdbdbd"), legend


def pct_of(series: pd.Series, value: float) -> float:
    return float((series < value).mean() * 100)


def headline_numbers(res: pd.DataFrame) -> dict:
    pri = res[res["priority"]]
    beyond30 = 100 * (1 - (res["cov_any_30"] * res["population"]).sum() / res["population"].sum())
    return {"pri_pop_m": pri["population"].sum() / 1e6,
            "stable": int((res["priority"] & ~res["unstable"]).sum()),
            "beds_short": res["bed_gap_shortfall"].sum(),
            "beyond30": beyond30,
            "beyond30_m": (res["population"] * (1 - res["cov_any_30"])).sum() / 1e6}


# --------------------------------------------------------------------------- page chrome
def page_header(kicker: str, title: str, text: str = "") -> None:
    st.markdown(f"<div class='sec first'><div><span class='kicker'>{kicker}</span><h1 class='ptitle'>{title}</h1></div>"
                + (f"<p>{text}</p>" if text else "<div></div>") + "</div>", unsafe_allow_html=True)


def section(kicker: str, title: str, text: str = "") -> None:
    st.markdown(f"<div class='sec'><div><span class='kicker'>{kicker}</span><h2>{title}</h2></div>"
                + (f"<p>{text}</p>" if text else "<div></div>") + "</div>", unsafe_allow_html=True)


def footer(meta: dict | None = None) -> None:
    st.markdown(f"""<div class="footer">
      <span class="brand"><img src="data:image/png;base64,{LOGO_B64}" alt=""><b>{BRAND_HTML}</b></span>
      <span>© 2026 {APP_NAME}</span>
    </div>""", unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@700;800;900&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
[data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: rgba(251,248,241,.92); backdrop-filter: blur(6px); border-bottom: 1px solid #ECE5D6; }}
.block-container {{ padding-top: 4.6rem; padding-bottom: 1rem; max-width: 1220px; }}
h1, h2, h3 {{ font-family: 'Inter Tight', 'Inter', system-ui, sans-serif !important; letter-spacing: -0.02em; color: {NAVY}; }}

.gl {{ position: relative; cursor: help; border-bottom: 1px dotted #9AA6B8; outline: none; }}
.gl::after {{ content: attr(data-tip); position: absolute; left: 0; top: calc(100% + 6px); width: 260px; max-width: min(260px, calc(100vw - 36px)); z-index: 1000;
  background: #FFFDF8; color: #16213A; border: 1px solid #D8CCB5; font: 400 12.5px/1.5 Inter, system-ui, sans-serif;
  padding: 10px 12px; border-radius: 10px; box-shadow: 0 8px 22px rgba(0,33,87,.16);
  opacity: 0; visibility: hidden; transition: opacity .12s ease; pointer-events: none;
  white-space: normal; overflow-wrap: anywhere; word-break: normal; max-height: 220px; overflow-y: auto; text-transform: none; letter-spacing: 0; }}
/* Pointer clicks must not pin an old tooltip. Keyboard users retain :focus-visible support. */
.gl:hover::after, .gl:focus-visible::after {{ opacity: 1; visibility: visible; }}
.gl:focus:not(:focus-visible)::after {{ opacity: 0; visibility: hidden; }}
/* Keep the current page/navigation item visually obvious in the top navigation. */
header nav a[aria-current="page"], header nav [aria-current="page"],
header [data-testid="stTopNav"] a[aria-current="page"],
header [data-testid="stTopNav"] button[aria-current="page"],
header [role="tab"][aria-selected="true"] {{ font-weight: 800 !important; color: #002157 !important; }}
header nav a[aria-current="page"] *, header [data-testid="stTopNav"] [aria-current="page"] * {{ font-weight: 800 !important; }}
/* Streamlit-native help popovers use the same light tooltip palette as glossary tips. */
[data-testid="stTooltipContent"] {{ background: #FFFDF8 !important; color: #16213A !important;
  border: 1px solid #D8CCB5 !important; border-radius: 10px !important; box-shadow: 0 8px 22px rgba(0,33,87,.16) !important; }}
[data-testid="stTooltipContent"] * {{ color: #16213A !important; }}
.kicker {{ display: inline-flex; align-items: center; gap: 8px; color: {BLUE}; font-weight: 700; font-size: .78rem;
           letter-spacing: .08em; text-transform: uppercase; }}
.kicker::before {{ content: ""; width: 8px; height: 8px; border-radius: 50%; background: {RED}; }}

/* hero */
.hero {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: clamp(20px, 4vw, 48px); align-items: center;
         background: #fff; border: 1px solid #EFE8DA; border-radius: 32px; padding: clamp(22px, 4.5vw, 54px); }}
.hero h1 {{ font-size: clamp(2.2rem, 5.4vw, 4.1rem) !important; font-weight: 900 !important; line-height: 1.02 !important;
            margin: 14px 0 0 !important; padding: 0 !important; letter-spacing: -0.035em; }}
.hero h1 em {{ font-style: normal; color: {RED}; }}
.acro {{ color: {MUTED}; font-size: .86rem; margin-top: 12px; line-height: 1.5; }}
.acro b {{ color: {NAVY}; }}
.lead {{ color: #3E4A60; margin: 16px 0 22px; font-size: clamp(.95rem, 1.6vw, 1.05rem); line-height: 1.6; max-width: 540px; }}
.cta {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
.btn {{ border-radius: 14px; padding: 12px 20px; font-weight: 700; font-size: .88rem; text-decoration: none !important; }}
.btn.solid {{ background: {NAVY}; color: #fff !important; }}
.btn.solid:hover {{ background: {BLUE}; }}
.btn.line {{ border: 1px solid #D6DCE8; color: {NAVY} !important; background: {CREAM}; }}
.credits {{ margin-top: 22px; color: {MUTED}; font-size: .78rem; }}
.credits b {{ color: #3E4A60; font-weight: 600; }}
.art {{ position: relative; }}
.art svg {{ width: 100%; height: auto; display: block; }}
.chip {{ position: absolute; background: #fff; border-radius: 16px; padding: 10px 14px; box-shadow: 0 8px 24px rgba(0,33,87,.12);
         font-size: .78rem; color: {MUTED}; line-height: 1.25; }}
.chip b {{ display: block; font-family: 'Inter Tight', sans-serif; font-size: 1.35rem; color: {NAVY}; }}
.chip.c1 {{ left: -14px; bottom: 18%; }} .chip.c2 {{ right: -10px; top: 10%; }}
.chip .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.hero-headlines {{ display: grid; align-items: start; margin-top: 14px; min-height: clamp(150px, 17vw, 220px); }}
.hero-headlines .headline {{ grid-area: 1 / 1; margin: 0 !important; opacity: 0; transform: translateY(9px); animation: headlineCycle 15s infinite; }}
.hero-headlines .headline:nth-child(2) {{ animation-delay: 5s; }}
.hero-headlines .headline:nth-child(3) {{ animation-delay: 10s; }}
@keyframes headlineCycle {{ 0%, 2% {{ opacity: 0; transform: translateY(9px); }} 6%, 29% {{ opacity: 1; transform: translateY(0); }} 34%, 100% {{ opacity: 0; transform: translateY(-6px); }} }}
.chip.pair-a {{ animation: statPairA 14s steps(1,end) infinite; }}
.chip.pair-b {{ opacity: 0; animation: statPairB 14s steps(1,end) infinite; }}
.chip.c3 {{ left: -12px; top: 10%; }} .chip.c4 {{ right: -8px; bottom: 14%; }}
@keyframes statPairA {{ 0%, 49.99% {{ opacity: 1; visibility: visible; }} 50%, 100% {{ opacity: 0; visibility: hidden; }} }}
@keyframes statPairB {{ 0%, 49.99% {{ opacity: 0; visibility: hidden; }} 50%, 100% {{ opacity: 1; visibility: visible; }} }}
.stat-rail {{ display: flex; gap: 12px; overflow-x: auto; padding: 4px 2px 14px; scroll-snap-type: x mandatory; scrollbar-width: thin; }}
.stat-rail-card {{ flex: 0 0 210px; min-height: 118px; scroll-snap-align: start; border: 1px solid #ECE5D6; border-radius: 18px; padding: 15px 16px; background: #fff; color: {NAVY}; transition: background .18s ease, color .18s ease; box-sizing: border-box; }}
.stat-rail-card .value {{ font-family: 'Inter Tight', sans-serif; font-size: 1.8rem; font-weight: 900; letter-spacing: -.03em; line-height: 1.05; }}
.stat-rail-card .label {{ font-size: .82rem; font-weight: 750; margin-top: 9px; line-height: 1.25; }}
.stat-rail-card .note {{ font-size: .72rem; color: {MUTED}; margin-top: 4px; }}
.stat-rail-card.is-active {{ background: {NAVY}; border-color: {NAVY}; color: #fff; }}
.stat-rail-card.is-active .note {{ color: #C8D4E9; }}
.nav-hub {{ height: 100%; min-height: 180px; display: flex; flex-direction: column; gap: 8px; }}
.nav-hub h3 {{ margin-top: 4px !important; }}
.nav-hub p {{ flex: 1; }}
.analysis-note {{ border-left: 4px solid {BLUE}; background: #F0F5FC; border-radius: 0 12px 12px 0; padding: 10px 14px; color: {INK}; margin: 8px 0 12px; line-height: 1.5; }}
.process-step {{ border: 1px solid #DCE4F0; background: #fff; border-radius: 12px; padding: 10px 8px; min-height: 50px; text-align: center; font-size: .84rem; color: {NAVY}; }}
@media (prefers-reduced-motion: reduce) {{ .hero-headlines .headline {{ animation: none; }} .hero-headlines .headline:first-child {{ opacity: 1; transform: none; }} .hero-headlines .headline:not(:first-child) {{ display: none; }} .chip.pair-a {{ animation: none; opacity: 1; visibility: visible; }} .chip.pair-b {{ display: none; }} }}

/* section / page headings */
.sec {{ margin: clamp(40px, 6vw, 70px) 0 clamp(14px, 2.5vw, 22px); display: grid; grid-template-columns: 1fr 1fr;
        gap: 10px 40px; align-items: end; }}
.sec.first {{ margin-top: 6px; }}
.sec h2, .sec h1.ptitle {{ font-size: clamp(1.7rem, 4vw, 2.6rem) !important; font-weight: 800 !important; line-height: 1.1 !important;
           margin: 8px 0 0 !important; padding: 0 !important; }}
.sec p {{ color: {MUTED}; margin: 0; font-size: .95rem; line-height: 1.55; }}

/* findings tiles */
.tiles {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
.tile {{ background: #fff; border: 1px solid #ECE5D6; border-radius: 24px; padding: 20px; position: relative; overflow: hidden; }}
.tile::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c, {BLUE}); }}
.tile.accent {{ background: {NAVY}; border-color: {NAVY}; }}
.tile.accent .num, .tile.accent .lbl {{ color: #fff; }} .tile.accent .note {{ color: #B8C4DA; }}
.num {{ font-family: 'Inter Tight', sans-serif; font-weight: 900; font-size: clamp(2rem, 4vw, 2.9rem); color: {NAVY};
        letter-spacing: -0.03em; line-height: 1; }}
.lbl {{ font-weight: 700; color: {INK}; margin-top: 10px; font-size: .92rem; }}
.note {{ color: {MUTED}; font-size: .78rem; margin-top: 3px; }}

/* index cards */
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
.cards.four {{ grid-template-columns: repeat(4, 1fr); margin-top: 14px; }}
.cards.three {{ grid-template-columns: repeat(3, 1fr); margin-top: 14px; }}
@media (max-width: 900px) {{ .cards.three {{ grid-template-columns: 1fr; }} }}
@media (max-width: 1100px) {{ .cards.four {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 640px) {{ .cards.four {{ grid-template-columns: 1fr; }} }}
.card {{ background: #fff; border: 1px solid #ECE5D6; border-radius: 24px; padding: 22px; color: {INK}; }}
.card .no {{ font-family: 'Inter Tight', sans-serif; font-weight: 800; color: #B8C4DA; font-size: .95rem; }}
.card h3 {{ font-size: 1.15rem !important; font-weight: 800 !important; margin: 6px 0 2px !important; padding: 0 !important; }}
.card .aka {{ color: {BLUE}; font-size: .78rem; font-weight: 700; }}
.card p {{ color: #3E4A60; font-size: .9rem; line-height: 1.5; margin: 10px 0 14px; }}
.scale {{ height: 10px; border-radius: 999px; }}
.scale-l {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .72rem; margin-top: 5px; }}

/* widgets */
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 24px !important; background: #fff; border-color: #ECE5D6 !important; }}
.stButton button, .stDownloadButton button, div[data-testid="stPopover"] button {{ border-radius: 14px !important; }}
div[data-testid="stPageLink"] a {{ background: #fff; border: 1px solid #ECE5D6; border-radius: 14px; padding: 8px 14px; }}
iframe {{ border-radius: 22px; }}
.idx-card {{ background: #fff; border: 1px solid #ECE5D6; border-radius: 22px; padding: 16px 18px; color: {INK}; height: 100%; }}
.idx-card .t {{ font-size: .72rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }}
.idx-card .v {{ font-family: 'Inter Tight', sans-serif; font-size: 2.2rem; font-weight: 900; color: var(--c); letter-spacing: -0.03em; }}
.bar {{ background: #EEF1F6; border-radius: 999px; height: 8px; overflow: hidden; margin-top: 6px; }}
.bar > div {{ height: 100%; background: var(--c); border-radius: 999px; }}
.biv {{ display: grid; grid-template-columns: auto repeat(3, minmax(52px, 88px)); gap: 5px; align-items: center;
        font-size: .8rem; width: fit-content; }}
.biv div.c {{ height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
             color: #fff; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,.35); }}

/* about + footer */
.about {{ background: {SOFT}; border-radius: 28px; padding: clamp(22px, 4vw, 40px); display: grid;
          grid-template-columns: 1.2fr 1fr; gap: 24px; color: {INK}; }}
.about h3 {{ font-size: 1.05rem !important; font-weight: 800 !important; margin: 0 0 6px !important; padding: 0 !important; }}
.about p {{ color: #3E4A60; font-size: .92rem; line-height: 1.6; margin: 0 0 12px; }}
.footer {{ margin-top: 48px; border-top: 1px solid #E6DECB; padding: 18px 2px 6px; display: flex; justify-content: space-between;
           align-items: center; flex-wrap: wrap; gap: 14px; color: {MUTED}; font-size: .76rem; line-height: 1.5; }}
.brand {{ display: inline-flex; align-items: center; gap: 10px; }}
.brand img {{ width: 30px; height: auto; }}
.brand b {{ font-family: 'Inter Tight', sans-serif; font-weight: 900; font-size: 1.05rem; color: {NAVY}; letter-spacing: .02em; }}
.brand .ph {{ color: {RED}; }}

@media (max-width: 900px) {{
  .hero {{ grid-template-columns: 1fr; }}
  .art {{ max-width: 420px; margin: 0 auto; }}
  .tiles {{ grid-template-columns: repeat(2, 1fr); }}
  .sec {{ grid-template-columns: 1fr; }}
  .about {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 640px) {{
  .block-container {{ padding: 4.2rem .7rem 1rem; }}
  .hero {{ border-radius: 24px; }}
  .chip.c1 {{ left: 6px; }} .chip.c2 {{ right: 6px; }}
  .tiles {{ gap: 10px; }}
  .tile {{ padding: 16px; border-radius: 20px; }}
  .biv div.c {{ height: 36px; }}
}}
</style>
""", unsafe_allow_html=True)


# =========================================================================== map helpers (legend, hover cards)
DOMAIN_LABELS = {"hvs_demographic": "Age dependency (young & old)", "hvs_socioeconomic": "Poverty",
                 "hvs_geographic": "Geographic isolation (GIDA)", "hvs_disease": "Disease burden",
                 "hcs_beds": "Hospital beds within reach", "hcs_coverage": "Hospitals within 30–60 min",
                 "hcs_primary_care": "Primary care", "hcs_social_protection": "Malasakit reach"}
LEGEND_TEXT = {
    "gap_index": ("Healthcare Gap (HGI)", "−100 · smaller gap", "+100 · larger gap",
                  "<b style='color:#45197A'>Darker purple</b> = health need is furthest ahead of the care people can reach, "
                  "compared with other areas. <b>Pale</b> = a smaller gap than elsewhere, <b>not</b> enough care: almost "
                  "every area is below the national bed targets (see <i>Bed need met</i>)."),
    "pct_goal": ("Bed need met (2040 goal)", "0%", "100% · goal met",
                 "Share of the 2040 goal of <b>4 hospital beds per 1,000 people</b> that people here can reach. "
                 "31% equals the DOH minimum of 1 bed per 800 people."),
    "HVS": ("Health Need (HNI)", "0 · low need", "100 · high need",
            "<b style='color:#b30000'>Darker red</b> = more people likely to need care (young children, older adults, "
            "poverty, isolation)."),
    "HCS": ("Hospital Capacity (HCI)", "0 · least", "100 · most (relative)",
            "<b style='color:#253494'>Darker blue</b> = more beds and services within reach than other areas. "
            "Even the darkest areas are below the 2040 goal of 4 beds per 1,000."),
    "dHGI": ("Change in gap after your facilities", "gap closes", "gap widens",
             "<b style='color:#1a9850'>Green</b> = the gap gets smaller after the facilities you placed. "
             "Grey = no change."),
}
CMAP_STOPS = {"RdBu_r": ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
              "RdYlBu_r": ["#313695", "#74add1", "#fee090", "#f46d43", "#a50026"],
              "Gap": GAP_STOPS, "Adequacy": ADEQ_STOPS,
              "OrRd": ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"],
              "YlGnBu": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
              "RdYlGn": ["#1a9850", "#91cf60", "#f7f7f7", "#fc8d59", "#d73027"]}

TOOLTIP_CSS = """<style>
.leaflet-tooltip{padding:0;border:0;border-radius:16px;box-shadow:0 10px 28px rgba(0,33,87,.18)}
.tt{font-family:Inter,system-ui,sans-serif;width:292px;padding:14px 16px;color:#16213A;white-space:normal}
.tt .top{display:flex;align-items:center;gap:8px;font-size:11px;color:#5A6B85}
.tt .bd{display:inline-block;min-width:34px;text-align:center;color:#fff;font-weight:700;font-size:12px;border-radius:7px;padding:2px 6px}
.tt .nm{font-size:17px;font-weight:700;margin:6px 0 0;line-height:1.15}
.tt .sb{font-size:11.5px;color:#5A6B85;margin-bottom:8px}
.tt .hd{font-size:12.5px;font-weight:700;margin:10px 0 6px;color:#002157}
.tt .gr{display:grid;grid-template-columns:1fr 1fr;gap:6px 10px}
.tt .it{display:flex;align-items:center;gap:6px;font-size:10.5px;line-height:1.15;color:#3E4A60}
.tt .it .bd{min-width:28px;font-size:11px;padding:1px 4px}
.tt .age{display:flex;height:7px;border-radius:4px;overflow:hidden;margin:6px 0 3px}
.tt .lg{display:flex;gap:10px;font-size:10px;color:#5A6B85}
.tt .lg i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:3px}
.tt .ft{margin-top:8px;padding-top:7px;border-top:1px solid #ECE5D6;font-size:10.5px;color:#5A6B85}
.tt .ac{display:grid;grid-template-columns:22px 1fr auto;gap:4px 6px;align-items:center;font-size:11px;color:#3E4A60}
.tt .ac .ic{font-size:14px;line-height:1;text-align:center}
.tt .ac b{color:#16213A;text-align:right;white-space:nowrap}
.tt .pies{display:grid;grid-template-columns:1fr;gap:10px;margin-top:2px}
.tt .pl div{white-space:nowrap}
.tt .pie{display:flex;align-items:center;gap:8px}
.tt .dn{width:46px;height:46px;border-radius:50%;flex-shrink:0;position:relative}
.tt .dn::after{content:"";position:absolute;inset:11px;border-radius:50%;background:#fff}
.tt .pl{font-size:10px;line-height:1.45;color:#5A6B85}
.tt .pl i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px}
.tt .pl b{color:#16213A}
.tt .need{margin:8px 0 2px;padding:7px 9px;border-radius:10px;background:#FFF6EE;border:1px solid #F3DCC8}
.tt .need .nt{font-size:11.5px;color:#16213A} .tt .need .nt b{color:#A50026}
.tt .need .bar{position:relative;height:7px;border-radius:5px;background:#EADFD3;margin:5px 0 4px;overflow:visible}
.tt .need .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px}
.tt .need .bar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:#16213A}
.tt .need .ns{font-size:10.5px;color:#5A6B85}
.tt .tip{position:relative;cursor:help;border-bottom:1px dotted #9AA6B8;outline:none}
.tt .tip::after{content:attr(data-tip);position:absolute;left:0;top:calc(100% + 6px);width:205px;max-width:calc(100vw - 36px);z-index:1000;
  background:#FFFDF8;color:#16213A;border:1px solid #D8CCB5;font:400 11px/1.5 Inter,system-ui,sans-serif;letter-spacing:0;text-transform:none;
  padding:9px 11px;border-radius:10px;box-shadow:0 8px 22px rgba(0,33,87,.16);white-space:normal;text-align:left;
  opacity:0;visibility:hidden;transition:opacity .12s ease;pointer-events:none}
.tt .tip.r::after{left:auto;right:0}
.tt .tip:hover::after,.tt .tip:focus-visible::after{opacity:1;visibility:visible}
.tt .tip:focus:not(:focus-visible)::after{opacity:0;visibility:hidden}
.folium-map,.leaflet-container{width:calc(100% - 330px)!important}
.isg-panel{position:absolute;top:0;right:0;width:330px;height:100vh;overflow-y:auto;box-sizing:border-box;z-index:1000;
  background:#F6F3EC;border-left:1px solid #E6DECB;padding:10px;font-family:Inter,system-ui,sans-serif}
.isg-panel .tt{width:auto;background:#fff;border-radius:16px;box-shadow:0 4px 16px rgba(0,33,87,.08)}
.isg-panel .ph{color:#5A6B85;font-size:13px;line-height:1.5;padding:18px 14px;background:#fff;border-radius:16px;
  border:1px dashed #D6DCE8}
.isg-panel .ph b{color:#002157;font-size:15px;display:block;margin-bottom:6px}
.isg-tip{font:600 12px Inter,system-ui,sans-serif;color:#16213A;padding:5px 9px!important}
@media (max-width:700px){
  .folium-map,.leaflet-container{width:100%!important;height:58vh!important}
  .isg-panel{top:58vh;width:100%;height:42vh;border-left:0;border-top:1px solid #E6DECB}
  .isg-legend{bottom:calc(42vh + 10px)!important}
}
</style>"""


def badge_color(v: float, kind: str) -> str:
    """need: high = bad (red); cap: high = good (teal); gap: positive = bad."""
    if kind == "gap":
        return "#45197A" if v >= 33 else ("#7A4BB0" if v >= 0 else "#9C86B8")
    bad = v >= 66 if kind == "need" else v <= 33
    mid = 33 <= v < 66
    return "#CE3550" if bad else ("#E8913A" if mid else "#1E9E8C")


TIPS = {
    "Healthcare Gap Index": "How far this area's health need ranks above (+) or below (−) its hospital capacity compared with other areas, from −100 to +100. A negative gap means a smaller gap than elsewhere, not enough care.",
    "Bed need met": "How many hospital beds people here can reach, as a share of the 2040 goal of 4 beds per 1,000 people. The DOH minimum of 1 bed per 800 people is 31% of that goal.",
    "Travel times": "Travel times use explicit transport assumptions: road-class speeds plus an access/egress leg for mapped road routes, and a configurable generalized effective-speed fallback where routes are unavailable. The transport assumptions reflect configured road speeds and generalized local/mixed travel. The often-cited DOH 30-minute statistic refers to RHUs/health centres (primary care), not hospitals, so it is contextual rather than a direct benchmark for the 30-minute hospital estimate.",
    "Priority area": "An area in the top third for health need and the bottom third for hospital capacity: where care lags furthest behind need.",
    "Health Need": "How much health care the people here are likely to need, from 0 (low) to 100 (high).",
    "Hospital Capacity": "How much hospital care people here can reach compared with other areas, from 0 (least) to 100 (most). A high score is not the same as enough.",
    "What drives it": "Scores from 0 to 100 for each part of the two indices; red marks the parts that make the gap worse.",
    "hvs_demographic": "How many residents are young children (0–4) or older adults (60+), the groups who need care most often.",
    "hvs_socioeconomic": "How many residents live below the poverty line and may struggle to pay for or travel to care.",
    "hvs_geographic": "How many barangays here the DOH lists as geographically isolated and disadvantaged (GIDA), where reaching care is hard.",
    "hvs_disease": "How common tuberculosis, pneumonia and high blood pressure are here compared with the rest of the country.",
    "hcs_beds": "How many hospital beds people can reach within 30–60 minutes, shared fairly among everyone who uses the same hospitals.",
    "hcs_coverage": "How many residents live within 30 minutes of any hospital or 60 minutes of a larger Level 2 or 3 hospital.",
    "hcs_primary_care": "How many DOH-licensed primary care facilities, such as rural health units, there are for every 10,000 residents.",
    "hcs_social_protection": "How many residents live within 60 minutes of a Malasakit Center, which helps patients pay hospital bills.",
    "Access & services": "Distances (straight line) from where most people here live to the nearest facilities, travel time to hospitals, and services inside the area.",
    "Nearest hospital": "Straight-line distance to the closest licensed hospital or infirmary, and the average travel time to a hospital (by road where road data exist, otherwise estimated).",
    "Nearest primary care facility": "Distance to the closest DOH-licensed primary care facility, the first stop for check-ups and common illness.",
    "Nearest Malasakit Center": "Distance to the closest hospital with a Malasakit Center, a one-stop desk for help with hospital bills.",
    "Nearest dialysis": "Distance to the closest facility that offers dialysis for people with kidney failure.",
    "Ambulances in LGU": "Ambulances listed by hospitals and ambulance providers in this area, also shown per 100,000 residents.",
    "Hospitals · licensed beds here": "How many licensed hospitals are inside this area and how many beds the DOH allows them in total.",
    "Disease per 100,000": "Cases reported to the DOH in 2022 for every 100,000 people; * means no DOH data, so the regional median is shown.",
    "Tuberculosis": "A lung infection that spreads through the air and needs months of treatment.",
    "Pneumonia": "A lung infection and a leading cause of hospital admission, especially for young children and older adults.",
    "Hypertension": "High blood pressure, a long-term condition that raises the risk of stroke and heart disease.",
    "Influenza": "The flu, a seasonal viral infection; shown for reference but not scored because many areas lack data.",
    "Leptospirosis": "An infection caught from floodwater contaminated by animal urine; shown for reference but not scored.",
    "People & beds": "Who lives here by age group, and how the licensed beds inside this area split between public and private hospitals.",
    "Age": "Share of residents who are children (0–14), youth (15–24), young adults (25–44), middle-aged (45–59) or elderly (60+), from the 2020 census.",
    "Beds here": "Licensed hospital beds inside this area, split by public and private ownership.",
    "Poverty incidence": "Share of people living below the poverty line in 2023, from PSA estimates.",
    "priority": "How often this area stayed a priority (high need, low capacity) when the model was re-run 1,000 times with different reasonable settings. 80% or more is stable; less is unstable.",
}



def _svg(body: str, color: str) -> str:
    return (f"<svg viewBox='0 0 16 16' width='15' height='15' fill='none' stroke='{color}' stroke-width='1.5' "
            f"stroke-linecap='round' stroke-linejoin='round' style='vertical-align:-3px'>{body}</svg>")


ICONS = {  # simple outline icons drawn in-page, so they look the same on every device
    "hospital": _svg("<rect x='2.5' y='3' width='11' height='11' rx='1.5'/><path d='M8 5.5v6M5 8.5h6'/>", "#002157"),
    "primary": _svg("<path d='M2.5 7.5 8 3l5.5 4.5V14h-11z'/><path d='M8 8v4M6 10h4'/>", "#1E9E8C"),
    "malasakit": _svg("<path d='M8 13.5S2.5 10 2.5 6.3A2.8 2.8 0 0 1 8 5a2.8 2.8 0 0 1 5.5 1.3C13.5 10 8 13.5 8 13.5z'/>", "#CE3550"),
    "dialysis": _svg("<path d='M8 2.5S4 7 4 9.8a4 4 0 0 0 8 0C12 7 8 2.5 8 2.5z'/>", "#CE3550"),
    "ambulance": _svg("<path d='M1.5 11.5V5h8v6.5M9.5 7.5h2.6l2.4 2.4v1.6'/><circle cx='4.5' cy='12' r='1.3'/>"
                      "<circle cx='11.5' cy='12' r='1.3'/><path d='M5.5 6.5v3M4 8h3'/>", "#4F7FBE"),
    "beds": _svg("<path d='M1.5 4v9M1.5 10h13v3M14.5 10V8.5a2 2 0 0 0-2-2H7v3.5'/><circle cx='4.3' cy='8' r='1.3'/>", "#002157"),
    "tb": _svg("<circle cx='8' cy='8' r='3.2'/><path d='M8 1.8v2.4M8 11.8v2.4M1.8 8h2.4M11.8 8h2.4M3.6 3.6l1.7 1.7M10.7 10.7l1.7 1.7"
               "M3.6 12.4l1.7-1.7M10.7 5.3l1.7-1.7'/>", "#3E8E41"),
    "pneumonia": _svg("<path d='M8 2.5v5M8 7.5c-1 0-2-1-2.5-.5C3.5 8 2.5 10.5 2.5 12.5c0 1 .8 1.3 1.6 1L7 12.3c.6-.3 1-.9 1-1.6z"
                      "M8 7.5c1 0 2-1 2.5-.5 2 1 3 3.5 3 5.5 0 1-.8 1.3-1.6 1L9 12.3c-.6-.3-1-.9-1-1.6z'/>", "#D9767E"),
    "hypertension": _svg("<path d='M8 13.5S2.5 10 2.5 6.3A2.8 2.8 0 0 1 8 5a2.8 2.8 0 0 1 5.5 1.3C13.5 10 8 13.5 8 13.5z'/>"
                         "<path d='M3.5 8.5h2l1-1.5 1.5 3 1-1.5h3.5'/>", "#B2182B"),
    "influenza": _svg("<path d='M7 2.5a1.5 1.5 0 0 1 3 0v7a2.8 2.8 0 1 1-3 0z'/><path d='M8.5 6v5'/>", "#E8913A"),
    "road": _svg("<path d='M5 1.5 3 14.5M11 1.5l2 13M8 2v2.2M8 6.9v2.2M8 11.8V14'/>", "#5A6B85"),
    "lepto": _svg("<path d='M1.5 6c1.5-1.3 2.8-1.3 4.3 0s2.8 1.3 4.3 0 2.8-1.3 4.3 0M1.5 10c1.5-1.3 2.8-1.3 4.3 0s2.8 1.3 4.3 0"
                  " 2.8-1.3 4.3 0'/>", "#5A6B85"),
}


TIP_KEYS = list(TIPS)
BED_TARGET = CFG["benchmarks"]["beds_per_1000"]


def tip(label: str, key: str | None = None, right: bool = False) -> str:
    """Label with a one-sentence description shown on hover (or tap, on phones)."""
    import html as _html
    text = TIPS.get(key or label)
    if not text:
        return label
    cls = "tip r" if right else "tip"
    return f"<span class='{cls}' tabindex='0' onmousedown='this.blur()' data-k='{TIP_KEYS.index(key or label)}'>{label}</span>"


AGE_PIE = [("pop_children", "#CE3550", "Children 0–14"), ("pop_youth", "#F5C400", "Youth 15–24"),
           ("pop_young_adults", "#4F7FBE", "Young adults 25–44"), ("pop_middle_age", "#9AA6B8", "Middle age 45–59"),
           ("pop_elderly", "#002157", "Elderly 60+")]
PRIORITY_TAG = None   # set after tip() is defined
MALASAKIT_TAG = " · <b style='color:#CE3550'>Malasakit Center</b>"


def lgu_card(r) -> str:
    """Rich hover card for one LGU (built from the processed indices and demographics)."""
    items = []
    for c, lab in DOMAIN_LABELS.items():
        if c in r.index and pd.notna(r[c]):
            kind = "need" if c.startswith("hvs_") else "cap"
            items.append(f"<div class='it'><span class='bd' style='background:{badge_color(r[c], kind)}'>"
                         f"{r[c]:.0f}</span>{tip(lab, c, right=len(items) % 2 == 1)}</div>")
    if "pop_children" in r.index and pd.notna(r.get("pop_children")) and r["population"] > 0:
        ages = [(100 * r[c] / r["population"], col, lab) for c, col, lab in AGE_PIE]
        age_pie = donut([(pct, col) for pct, col, _ in ages], [(col, lab, f"{pct:.0f}%") for pct, col, lab in ages], tip("Age"))
    else:                                       # older data without the five age groups
        p04 = 100 * r.get("pct_age_0_4", np.nan)
        p60 = 100 * r.get("pct_age_60_plus", np.nan)
        mid = max(0.0, 100 - p04 - p60)
        age_pie = donut([(p04, "#CE3550"), (mid, "#B8C4DA"), (p60, "#002157")],
                        [("#CE3550", "0–4", f"{p04:.0f}%"), ("#B8C4DA", "5–59", f"{mid:.0f}%"), ("#002157", "60+", f"{p60:.0f}%")],
                        tip("Age"))
    bp, bv = float(r.get("beds_public", 0) or 0), float(r.get("beds_private", 0) or 0)
    if bp + bv > 0:
        bed_pie = donut([(100 * bp / (bp + bv), "#002157"), (100 * bv / (bp + bv), "#F5C400")],
                        [("#002157", "Public", f"{bp:,.0f}"), ("#F5C400", "Private", f"{bv:,.0f}")], tip("Beds here", right=True))
    else:
        bed_pie = donut([(100, "#E3E8EF")], [("#E3E8EF", "No licensed", ""), ("#E3E8EF", "beds in LGU", "")], tip("Beds here", right=True))
    mins = r.get("mean_minutes_any", np.nan)
    by_road = float(r.get("road_share", 0) or 0) >= 0.5
    tsrc = "by road" if by_road else "straight-line estimate"
    rows = [
        (ICONS["hospital"], "Nearest hospital", km(r.get("km_hospital"))
         + (f" · ~{mins:.0f} min {'by road' if by_road else 'est.'}" if pd.notna(mins)
            else (" · none within 90 min by road" if by_road else ""))),
        (ICONS["primary"], "Nearest primary care facility", km(r.get("km_primary"))),
        (ICONS["malasakit"], "Nearest Malasakit Center", km(r.get("km_malasakit"))),
        (ICONS["dialysis"], "Nearest dialysis", km(r.get("km_dialysis"))),
        (ICONS["ambulance"], "Ambulances in LGU", count(r.get("ambulances"), r.get("ambulances_per_100k"))),
        (ICONS["beds"], "Hospitals · licensed beds here", f"{int(r.get('n_hospitals', 0) or 0)} · {bp + bv:,.0f}"),
        (ICONS["road"], "Travel times", "Road network" if by_road else "Straight-line estimate"),
    ]
    access = "".join(f"<span class='ic'>{i}</span><span>{tip(lab)}</span><b>{v}</b>" for i, lab, v in rows)
    drows = [(ICONS["tb"], "Tuberculosis", "rate_tb"), (ICONS["pneumonia"], "Pneumonia", "rate_pneumonia"),
             (ICONS["hypertension"], "Hypertension", "rate_hypertension"), (ICONS["influenza"], "Influenza", "rate_influenza"),
             (ICONS["lepto"], "Leptospirosis", "rate_leptospirosis")]
    dis = "".join(f"<span class='ic'>{i}</span><span>{tip(lab)}</span><b>{rate(r.get(c), r.get(c + '_imputed', 0))}</b>"
                  for i, lab, c in drows if c in r.index)
    lvl = {"city": "city rate", "province": "province rate"}.get(r.get("disease_level"), "regional median")
    disease_block = (f"<div class='hd'>{tip('Disease per 100,000')} <span style='font-weight:400;color:#5A6B85;"
                     f"font-size:10.5px'>(DOH 2022 · {lvl})</span></div><div class='ac'>{dis}</div>") if dis else ""
    return (f"<div class='tt'><div class='top'><span class='bd' style='background:{badge_color(r['gap_index'], 'gap')}'>"
            f"{r['gap_index']:+.0f}</span>{tip('Healthcare Gap Index')}</div>"
            f"<div class='nm'>{r['name']}</div><div class='sb'>{r['province']} · {r['population']:,.0f} people</div>"
            f"<div class='gr'><div class='it'><span class='bd' style='background:{badge_color(r['HVS'], 'need')}'>{r['HVS']:.0f}</span>"
            f"<b>{tip('Health Need')}</b></div><div class='it'><span class='bd' style='background:{badge_color(r['HCS'], 'cap')}'>"
            f"{r['HCS']:.0f}</span><b>{tip('Hospital Capacity', right=True)}</b></div></div>"
            f"{need_met_bar(r)}"
            f"<div class='hd'>{tip('What drives it')}</div><div class='gr'>{''.join(items)}</div>"
            f"<div class='hd'>{tip('Access &amp; services', 'Access & services')}</div><div class='ac'>{access}</div>"
            f"{disease_block}"
            f"<div class='hd'>{tip('People &amp; beds', 'People & beds')}</div><div class='pies'>{age_pie}{bed_pie}</div>"
            f"<div class='it' style='margin-top:6px'>{tip('Poverty incidence')}: <b>&nbsp;{100 * r['poverty_incidence']:.1f}%</b>"
            f"{' (imputed)' if r.get('poverty_imputed', 0) else ''}</div>"
            f"<div class='ft'>{priority_tag() if r['priority'] else ''}"
            f"{tip('priority in ' + format(100 * r.get('priority_freq', np.nan), '.0f') + '% of 1,000 model runs', 'priority')} · "
            f"{100 * r['cov_any_30']:.0f}% within 30 min of a hospital ({tsrc}) · distances in km are straight-line · "
            f"* = no DOH data, regional median</div></div>")


def priority_tag() -> str:
    return f"<b style='color:#A50026'>{tip('Priority area')}</b> · "


def need_met_bar(r) -> str:
    """Absolute adequacy: how much of the 2040 bed goal people here can reach (shown even when the area ranks well)."""
    b = float(r.get("beds_per_1000_all", 0) or 0)
    pct = max(0.0, min(100.0, 100 * b / GOAL_2040))
    col = matplotlib.colors.to_hex(matplotlib.colormaps["Adequacy"](pct / 100))
    dmin = 100 * BED_TARGET / GOAL_2040
    return (f"<div class='need'><div class='nt'>{tip('Bed need met')} <b>{pct:.0f}%</b> of the 2040 goal</div>"
            f"<div class='bar'><i style='width:{pct:.0f}%;background:{col}'></i><u style='left:{dmin:.0f}%'></u></div>"
            f"<div class='ns'>{b:.2f} of {GOAL_2040:g} beds per 1,000 within reach · DOH minimum {BED_TARGET:g}"
            f"{' (not met)' if b < BED_TARGET else ''}</div></div>")


def rate(v, imputed=0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.1f}" + ("*" if imputed else "")


def km(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return "&lt;1 km" if v < 1 else f"{v:,.0f} km"


def count(n, per100k) -> str:
    if n is None or pd.isna(n):
        return "—"
    return f"{int(n)}" + (f" · {per100k:.1f}/100k" if per100k is not None and pd.notna(per100k) and n else "")


def donut(parts, legend, title) -> str:
    """Small donut chart (CSS conic-gradient) with a legend; parts = [(percent, color), ...]."""
    stops, acc = [], 0.0
    for pct, color in parts:
        pct = 0.0 if pd.isna(pct) else float(pct)
        stops.append(f"{color} {acc:.1f}% {acc + pct:.1f}%")
        acc += pct
    leg = "".join(f"<div><i style='background:{c}'></i>{lab} <b>{val}</b></div>" for c, lab, val in legend)
    return (f"<div class='pie'><div class='dn' style='background:conic-gradient({', '.join(stops)})'></div>"
            f"<div class='pl'><div style='font-weight:700;color:#002157'>{title}</div>{leg}</div></div>")


def add_legend(m, key: str, cmap: str, extra: str = "") -> None:
    """Plain-language legend drawn inside the map (bottom-left)."""
    from branca.element import Element, MacroElement, Template

    title, lo, hi, text = LEGEND_TEXT[key]
    stops = ", ".join(CMAP_STOPS[cmap])
    html = f"""
    <div class="isg-legend" style="position:absolute;left:12px;bottom:22px;z-index:9999;background:rgba(255,255,255,.95);
        border-radius:14px;padding:10px 12px;width:min(290px,70vw);font:12px Inter,system-ui,sans-serif;color:#16213A;
        box-shadow:0 6px 20px rgba(0,33,87,.15)">
      <div style="font-weight:700;margin-bottom:6px">{title}</div>
      <div style="height:10px;border-radius:6px;background:linear-gradient(90deg,{stops})"></div>
      <div style="display:flex;justify-content:space-between;color:#5A6B85;font-size:10.5px;margin:3px 0 6px">
        <span>{lo}</span><span>{hi}</span></div>
      <div style="font-size:11px;line-height:1.4;color:#3E4A60">{text}</div>{extra}
    </div>"""
    macro = MacroElement()
    macro._template = Template("{% macro html(this, kwargs) %}" + html + "{% endmacro %}")
    m.get_root().add_child(macro)
    m.get_root().header.add_child(Element(TOOLTIP_CSS))


FACILITY_STYLE = {  # legend key for facility markers
    "Public hospital": ("#002157", 6), "Private hospital": ("#8A94A6", 5), "Infirmary": ("#4F7FBE", 4),
    "Primary care": ("#1E9E8C", 3), "Malasakit Center": ("#CE3550", 7),
}


def facility_legend(kinds) -> str:
    if not kinds:
        return ""
    rows = "".join(f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                   f"background:{FACILITY_STYLE[k][0]};border:2px solid #fff;box-shadow:0 0 0 1px #999;margin-right:6px'></span>{k}</div>"
                   for k in kinds)
    return f"<div style='margin-top:8px;padding-top:6px;border-top:1px solid #eee;font-size:11px;line-height:1.7'>{rows}</div>"


@st.cache_data
def lgu_hex_shapes(psgcs: tuple, hex_km: float) -> dict:
    """Hexagon cells grouped by LGU: {psgc: [ring, ring, ...]} for MultiPolygon features.

    Each hexagon belongs to the LGU of its nearest *barangay*. A hexagon is drawn only if it lies within
    that barangay's reach, which adapts to local spacing: 0.75 x the distance to the barangay's
    3rd-nearest neighbour, kept between one hexagon width and 8 km. Large rural barangays therefore
    fill their surroundings, while closely spaced coastal ones do not spill far into the sea.
    """
    from scipy.spatial import cKDTree
    from isagiph.accessibility import to_local_km

    res, *_rest, pts = load()
    keep = set(psgcs)
    allpts = pts if len(pts) else res[["psgc", "lat", "lon"]]
    axy = to_local_km(allpts["lat"].to_numpy(), allpts["lon"].to_numpy())
    atree = cKDTree(axy)
    d, _ = atree.query(axy, k=min(4, len(allpts)))
    reach = np.clip(0.75 * d[:, -1], hex_km, max(hex_km, 8.0))          # spacing measured among *all* barangays
    sel = allpts["psgc"].astype(str).isin(keep).to_numpy()
    mask = allpts[sel]
    if len(mask) == 0:
        return {}
    mreach = reach[sel]
    hexes = hex_grid(mask["lat"].to_numpy(), mask["lon"].to_numpy(), float(hex_km), mask_km=float(mreach.max()))
    tree = cKDTree(to_local_km(mask["lat"].to_numpy(), mask["lon"].to_numpy()))
    k = min(8, len(mask))
    dd, ii = tree.query(to_local_km(hexes["lat"].to_numpy(), hexes["lon"].to_numpy()), k=k)
    dd, ii = dd.reshape(len(hexes), k), ii.reshape(len(hexes), k)
    ok = (dd <= mreach[ii]).any(axis=1)
    owner = mask["psgc"].astype(str).to_numpy()[ii[:, 0]]
    shapes: dict = {}
    for ring, pc, good in zip(hexes["ring"], owner, ok):
        if good:
            shapes.setdefault(pc, []).append(ring)
    return shapes


def lgu_feature_collection(view: pd.DataFrame, value_col: str, to_color, hex_km: float, gj=None,
                           values: pd.Series | None = None) -> dict:
    """One GeoJSON feature per LGU (boundaries if available, else grouped hexagons) with color + hover card."""
    vals = values if values is not None else view.set_index("psgc")[value_col]
    feats = []
    if gj is not None:
        geo = {str(f["properties"].get("psgc")): f["geometry"] for f in gj["features"]}
        for _, r in view.iterrows():
            if r["psgc"] in geo:
                feats.append({"type": "Feature", "geometry": geo[r["psgc"]],
                              "properties": {"color": to_color(vals[r["psgc"]]), "card": lgu_card(r), "name": r["name"],
                                             "psgc": r["psgc"]}})
    else:
        shapes = lgu_hex_shapes(tuple(view["psgc"]), hex_km)
        for _, r in view.iterrows():
            rings = shapes.get(r["psgc"])
            if rings:
                feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[x] for x in rings]},
                              "properties": {"color": to_color(vals[r["psgc"]]), "card": lgu_card(r), "name": r["name"],
                                             "psgc": r["psgc"]}})
    return {"type": "FeatureCollection", "features": feats}


def add_lgu_layer(m, fc: dict, opacity: float, placeholder: str | None = None) -> None:
    """LGU layer whose full profile card appears in a side panel (not over the map) on hover or tap.

    To keep the page light, the profile cards and the label descriptions are not stored on every map
    feature: they are sent once, gzip-compressed, and unpacked by the browser.
    """
    import base64
    import gzip
    import folium
    from branca.element import Element, MacroElement, Template

    cards = {}
    feats = []
    for f in fc["features"]:
        props = dict(f["properties"])
        cards[props.get("psgc")] = props.pop("card", "")
        props.setdefault("name", "")
        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
    light = {"type": "FeatureCollection", "features": feats}
    blob = base64.b64encode(gzip.compress(json.dumps({"cards": cards, "tips": [TIPS[k] for k in TIP_KEYS]},
                                                     separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    geo = folium.GeoJson(
        light,
        style_function=lambda f: {"fillColor": f["properties"]["color"], "color": f["properties"]["color"],
                                  "weight": 0.6, "opacity": 0.5, "fillOpacity": opacity},
        highlight_function=lambda f: {"weight": 3, "color": "#16213A", "opacity": 1,
                                      "fillOpacity": min(opacity + 0.2, 0.85)},
        tooltip=folium.GeoJsonTooltip(fields=["name"], labels=False, sticky=True, class_name="isg-tip"),
    )
    geo.add_to(m)
    ph = placeholder or ("<div class='ph'><b>Click or tap an area</b>Its health need, hospital capacity, access to "
                         "services and demographics appear here.</div>")
    panel = MacroElement()
    panel._template = Template(
        "{% macro html(this, kwargs) %}" + TOOLTIP_CSS + "<div id='isg-panel' class='isg-panel'>" + ph
        + "</div>{% endmacro %}"
        "{% macro script(this, kwargs) %}"
        "(function(){"
        "  window.ISG = {cards: null, tips: []};"
        "  window.isgFill = function(el){ if (!el) return; el.querySelectorAll('[data-k]').forEach(function(t){"
        "    var v = ISG.tips[+t.getAttribute('data-k')]; if (v) t.setAttribute('data-tip', v); }); };"
        "  var bin = atob('" + blob + "'); var u = new Uint8Array(bin.length);"
        "  for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);"
        "  new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).text()"
        "    .then(function(t){ var d = JSON.parse(t); ISG.cards = d.cards; ISG.tips = d.tips;"
        "      isgFill(document.getElementById('isg-panel')); });"
        "})();"
        "(function bindPanel(tries){"
        "  var g = window['" + geo.get_name() + "'];"
        "  var n = 0; if (g) { g.eachLayer(function(){ n++; }); }"
        "  if (!g || n === 0) { if (tries < 50) { setTimeout(function(){ bindPanel(tries + 1); }, 200); } return; }"
        "  g.eachLayer(function(layer){"
        "    var show = function(){ var p = document.getElementById('isg-panel');"
        "      var pc = layer.feature && layer.feature.properties.psgc;"
        "      var html = (window.ISG && ISG.cards) ? ISG.cards[pc] : null;"
        "      if (p && html) { p.innerHTML = html; p.scrollTop = 0; isgFill(p); } };"
        "    layer.on('click', show); });"
        "})(0);"
        "setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 250);"
        "{% endmacro %}")
    m.add_child(panel)
    m.get_root().header.add_child(Element(TOOLTIP_CSS))


@st.cache_resource(max_entries=12, show_spinner=False)
def cached_lgu_layer(psgcs: tuple, col: str, cmap: str, hex_km: float, lite: bool = False) -> dict:
    """Feature collection for the given LGUs and index, built once and reused across reruns.
    ``lite`` uses coarser outlines (national view), which keeps the page light enough for phones."""
    res, _fac, _meta, _valid, _w, gj, _pts = load()
    if lite and load_lite_boundaries() is not None:
        gj = load_lite_boundaries()
    view = res.set_index("psgc").loc[list(psgcs)].reset_index()
    to_color, _ = scale_for(res[col], cmap, col)
    return lgu_feature_collection(view, col, to_color, hex_km, gj)



# =========================================================================== plain-language definitions
GLOSSARY = {
    "priority": "Top third of all cities and municipalities for health need and bottom third for hospital capacity: "
                "where care lags furthest behind need.",
    "stable": "Stayed a priority in at least 80% of 1,000 re-runs of the model with different reasonable settings, "
              "so the result does not hinge on one modelling choice.",
    "unstable": "Was a priority in fewer than 80% of 1,000 model re-runs: the result depends on modelling choices, "
                "so check it locally before acting on it.",
    "Hospital": "A licensed facility with beds that admits patients for treatment.",
    "Infirmary": "A small facility with a few beds for first aid, emergencies and short stays, often in remote areas.",
    "Primary care": "Clinics such as rural health units and health centers for check-ups, vaccines and common "
                    "illnesses; most have no beds.",
    "Level 1": "A basic hospital: emergency care, general wards, normal deliveries and common lab tests.",
    "Level 2": "Adds specialist departments (surgery, obstetrics, pediatrics, internal medicine), intensive care, "
               "and more lab and imaging services.",
    "Level 3": "A teaching and training hospital with the widest range of specialists and advanced care.",
    "Government": "Publicly run by the national, provincial, or city/municipal government; fees and assistance depend on the facility and applicable programs.",
    "Public": "Run by the national, provincial, or city/municipal government; fees and assistance depend on the facility and applicable programs.",
    "Private": "Privately run; services are usually paid by the patient or PhilHealth.",
    "Malasakit Center": "A one-stop desk inside a hospital where patients can ask for help paying hospital bills "
                        "from the DOH, PhilHealth, DSWD and PCSO.",
    "Dialysis": "Treatment that cleans the blood for people whose kidneys have failed, usually several times a week.",
    "Laboratory": "Tests on blood, urine and other samples.",
    "X-ray": "Imaging of bones and the chest.",
    "Pharmacy": "Dispenses medicines.",
    "Ambulance": "The facility lists at least one ambulance.",
    "Region": "One of the country's 17 administrative regions.",
    "Province": "A province (or an independent city counted like one) within the chosen regions.",
    "City or municipality": "A local government unit (LGU): the level at which every index is calculated.",
}


def term(word: str, key: str | None = None) -> str:
    """A word with a dotted underline that shows its plain-language definition on hover (or tap)."""
    import html as _html
    text = GLOSSARY.get(key or word) or GLOSSARY.get((key or word).lower())
    if not text:
        return word
    return f"<span class='gl' tabindex='0' onmousedown='this.blur()' data-tip=\"{_html.escape(text, quote=True)}\">{word}</span>"


def options_help(title: str, options) -> str:
    """Help text for a dropdown: one line per option that has a definition."""
    lines = [f"**{o}**: {GLOSSARY[o]}" for o in options if o in GLOSSARY]
    return (title + ("\n\n" + "\n\n".join(lines) if lines else "")).strip()


def cascade(container, regions_all, frame: pd.DataFrame, key: str, with_lgu: bool = False, multi: bool = True):
    """Region -> province (-> city/municipality) dropdowns where each list only shows places inside the one above.
    ``container`` may be one container or a list of containers (one per dropdown, e.g. st.columns(3))."""
    cs = list(container) if isinstance(container, (list, tuple)) else [container] * 3
    cs += [cs[-1]] * (3 - len(cs))
    if multi:
        regions = cs[0].multiselect("Region", regions_all, key=f"{key}_reg", placeholder="All regions",
                                        help=GLOSSARY["Region"])
        provs_all = sorted(frame.loc[frame["region"].isin(regions) if regions else slice(None), "province"].dropna().unique())
        provinces = cs[1].multiselect("Province", provs_all, key=f"{key}_prov", placeholder="All provinces",
                                          help=GLOSSARY["Province"])
        return regions, provinces
    region = cs[0].selectbox("Region", regions_all, key=f"{key}_reg", help=GLOSSARY["Region"])
    provs = sorted(frame.loc[frame["region"] == region, "province"].dropna().unique())
    province = cs[1].selectbox("Province", provs, key=f"{key}_prov", help=GLOSSARY["Province"])
    if not with_lgu:
        return region, province
    lgus = sorted(frame.loc[(frame["region"] == region) & (frame["province"] == province), "name"].unique())
    lgu = cs[2].selectbox("City or municipality", lgus, key=f"{key}_lgu", help=GLOSSARY["City or municipality"])
    return region, province, lgu



HEADERS = {
    "name": "City/municipality", "lgu": "City/municipality", "province": "Province", "lgu_province": "Province",
    "region": "Region", "population": "Population", "kind": "Type", "level": "Level", "ownership": "Ownership",
    "beds": "Beds", "services": "Services", "malasakit": "Malasakit Center", "HVS": "Health Need (HNI)",
    "HCS": "Hospital Capacity (HCI)", "gap_index": "Gap (HGI)", "HNI": "Health Need (HNI)", "HCI": "Hospital Capacity (HCI)",
    "HGI": "Gap (HGI)", "priority": "Priority", "unstable": "Unstable", "stable": "Stable",
    "priority_freq": "Priority in % of model runs", "actual_beds_per_1000": "Actual beds per 1,000",
    "expected_beds_per_1000": "Expected beds per 1,000", "LGUs": "Areas", "people (M)": "People (millions)",
    "accessible beds per 1,000": "Beds within reach per 1,000", "% ≤30 min to a hospital": "% within 30 min of a hospital",
    "% ≤60 min to Malasakit": "% within 60 min of a Malasakit Center", "mean HGI (pop-weighted)": "Average gap (population-weighted)",
    "% of people in priority LGUs": "% living in priority areas", "fifth": "Poverty fifth", "cluster_name": "Type of area",
    "domain": "Domain", "score": "Score", "bed_gap_shortfall": "Bed shortfall (DOH minimum)",
    "gida_share": "Share of GIDA barangays", "rate_tb": "Tuberculosis per 100,000", "rate_pneumonia": "Pneumonia per 100,000",
    "rate_hypertension": "Hypertension per 100,000", "rank_p05": "Best rank (5th percentile)",
    "rank_p95": "Worst rank (95th percentile)", "approx": "Approximate location", "lgu_province": "Province",
    "bed_gap_2040": "Bed shortfall (2040 goal)", "road_share": "Share with road travel times",
    "cov_any_30_straight": "Share within 30 min (straight-line estimate)", "pct_goal": "Bed need met (% of 2040 goal)",
    "bed_gap_2040": "Bed shortfall (2040 goal)", "beds_per_1000_all": "Beds within reach per 1,000",
    "cov_any_30": "Share within 30 min of a hospital", "poverty_incidence": "Poverty incidence", "rank": "Rank",
}
HEADER_HELP = {"Priority": GLOSSARY["priority"], "Unstable": GLOSSARY["unstable"], "Stable": GLOSSARY["stable"],
               "Gap (HGI)": "Need percentile minus capacity percentile, −100 to +100. Compares areas with each other.",
               "Health Need (HNI)": "How much care people here are likely to need, 0–100 (relative).",
               "Hospital Capacity (HCI)": "How much hospital care people here can reach, 0–100 (relative; not adequacy).",
               "Level": "Level 1: basic hospital. Level 2: adds specialist departments and intensive care. "
                        "Level 3: teaching hospital with the widest range of care.",
               "Type": "Hospital: admits patients. Infirmary: a few beds for first aid and short stays. "
                       "Primary care: clinics such as rural health units."}


def nice_header(c) -> str:
    if c in HEADERS:
        return HEADERS[c]
    t = str(c).replace("_", " ").strip()
    return t[:1].upper() + t[1:] if t else t


def show_table(df: pd.DataFrame, **kw):
    """Render only populated rows and size fixed-height tables to the available records."""
    out = df.copy()
    # Some source tables contain fully blank trailing records. Remove only rows that carry no information.
    out = out.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    out = out.rename(columns={c: nice_header(c) for c in out.columns})
    if out.index.name:
        out.index.name = nice_header(out.index.name)
    cfg = {c: st.column_config.Column(help=HEADER_HELP[c]) for c in out.columns if c in HEADER_HELP}
    cfg.update(kw.pop("column_config", {}) or {})
    requested_height = kw.get("height")
    if isinstance(requested_height, (int, float)) and len(out) < 1000:
        # Avoid showing blank placeholder rows when only a few LGUs/facilities match a filter.
        kw["height"] = min(int(requested_height), max(92, 35 + 35 * len(out)))
    return st.dataframe(out, column_config=cfg or None, **kw)



def add_map_control(m, html: str, position: str = "bottomleft") -> None:
    """Put an HTML box (e.g. a legend) inside the map as a Leaflet control, so it never ends up below the map."""
    from branca.element import MacroElement, Template
    ctl = MacroElement()
    ctl._template = Template(
        "{% macro script(this, kwargs) %}"
        "(function(){ var c = L.control({position: '" + position + "'});"
        " c.onAdd = function(){ var d = L.DomUtil.create('div'); d.innerHTML = " + json.dumps(html) + ";"
        " L.DomEvent.disableClickPropagation(d); return d; };"
        " c.addTo({{ this._parent.get_name() }}); })();"
        "{% endmacro %}")
    m.add_child(ctl)


SERVICE_HELP = {"Dialysis": GLOSSARY["Dialysis"], "X-ray / imaging": "X-ray, ultrasound, CT scan or other imaging.",
                "Pharmacy": GLOSSARY["Pharmacy"], "Birthing / maternity": "Care during pregnancy and childbirth.",
                "Laboratory": GLOSSARY["Laboratory"], "Dental": "Dental check-ups and treatment.",
                "Blood service": "Blood collection, storage, or transfusion.", "Ambulance": GLOSSARY["Ambulance"],
                "HIV testing": "Testing and counselling for HIV."}


# =========================================================================== maps
PH_MAX_BOUNDS = dict(min_lat=3.0, max_lat=22.5, min_lon=113.0, max_lon=129.5)


def make_map(**kw):
    """OpenStreetMap base map kept on the Philippines; surrounding countries are greyed out when an outline exists."""
    import folium

    m = folium.Map(tiles="OpenStreetMap", control_scale=True, prefer_canvas=True, max_bounds=True, min_zoom=5,
                   **PH_MAX_BOUNDS, **kw)
    mask = PROC / "ph_mask.geojson"
    if mask.exists():
        folium.GeoJson(json.loads(mask.read_text(encoding="utf-8")), interactive=False,
                       style_function=lambda f: {"fillColor": "#F6F3EC", "fillOpacity": 0.88, "color": "#9AA6B8",
                                                 "weight": 0.8}).add_to(m)
    return m


@st.cache_data(show_spinner=False)
def load_lite_boundaries():
    """Coarser outlines for the national view (much lighter for phones); None if not built."""
    f = PROC / "lgu_boundaries_lite.geojson"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


# =========================================================================== facilities directory + community
@st.cache_data
def load_directory() -> pd.DataFrame:
    p = PROC / "directory.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, dtype={"psgc": str})
    d["services"] = d["services"].fillna("")
    res = load()[0][["psgc", "name", "province", "region"]].rename(columns={"name": "lgu", "province": "lgu_province",
                                                                           "region": "lgu_region"})
    d = d.drop(columns=[c for c in ("lgu",) if c in d.columns]).merge(res, on="psgc", how="left")
    return d


def _canonical_services(value: object) -> str:
    """Normalize known service names for public-facing labels without changing source records."""
    import re
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    if not text or text.casefold() == "nan":
        return "Not listed"
    replacements = [
        (r"(?i)\bx\s*[- ]?\s*ray(?:\s*/\s*imaging)?\b", "X-ray / imaging"),
        (r"(?i)\bbirthing(?:\s*/\s*maternity)?\b", "Birthing / maternity"),
        (r"(?i)\bmalasakit\s+cent(?:er|re)\b", "Malasakit Center"),
        (r"(?i)\bprimary\s+care\b", "Primary care"),
        (r"(?i)\bblood\s+service\b", "Blood service"),
        (r"(?i)\bhiv\s+testing\b", "HIV testing"),
        (r"(?i)\bdialysis\b", "Dialysis"),
        (r"(?i)\blaboratory\b", "Laboratory"),
        (r"(?i)\bpharmacy\b", "Pharmacy"),
        (r"(?i)\bambulance\b", "Ambulance"),
        (r"(?i)\bdental\b", "Dental"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def add_facility_markers(m, d: pd.DataFrame, kinds: list[str], *, clickable: bool = False) -> None:
    """Add facility markers. In citizen finder mode, each marker opens a wrapped detail popup with its address."""
    import html
    import folium

    def details(f):
        name = html.escape(str(f.get("name", "Unnamed facility")))
        kind = html.escape(str(f.get("kind", "Facility")))
        level = html.escape(str(f.get("level", "Level not listed")))
        ownership = str(f.get("ownership", "Not listed"))
        ownership = "Public" if ownership == "Government" else ownership
        ownership = html.escape(ownership)
        beds_value = pd.to_numeric(pd.Series([f.get("beds")]), errors="coerce").iloc[0]
        beds = f"{beds_value:,.0f} licensed beds" if pd.notna(beds_value) and beds_value > 0 else "Beds not listed"
        address_raw = str(f.get("address", "")).strip()
        address = "Address not listed" if not address_raw or address_raw.casefold() == "nan" else html.escape(address_raw)
        services = html.escape(_canonical_services(f.get("services", "")))
        mal = bool(pd.to_numeric(pd.Series([f.get("malasakit", 0)]), errors="coerce").fillna(0).iloc[0])
        mal_line = "<div class='ft'><b>Malasakit Center:</b> Listed in source data; confirm current status.</div>" if mal else ""
        return ("<div class='tt' style='width:250px;max-width:260px;white-space:normal;overflow-wrap:anywhere;line-height:1.35'>"
                f"<div class='nm' style='font-size:14px;font-weight:800;line-height:1.2;margin-bottom:6px'>{name}</div>"
                f"<div class='sb'>{kind} · {level} · {ownership}</div>"
                f"<div class='it'>{html.escape(beds)}</div>"
                f"<div class='ft'><b>Address:</b> {address}</div>"
                f"<div class='ft'><b>Services:</b> {services}</div>{mal_line}</div>")

    groups = {
        "Public hospital": d[(d["kind"] == "Hospital") & (d["ownership"] == "Government")],
        "Private hospital": d[(d["kind"] == "Hospital") & (d["ownership"] == "Private")],
        "Infirmary": d[d["kind"] == "Infirmary"], "Primary care": d[d["kind"] == "Primary care"],
        "Malasakit Center": d[d["malasakit"] == 1],
    }
    for k in kinds:
        if k not in groups:
            continue
        color, rad = FACILITY_STYLE[k]
        fg = folium.FeatureGroup(k)
        for _, f in groups[k].iterrows():
            if pd.isna(f.get("lat")) or pd.isna(f.get("lon")):
                continue
            popup = folium.Popup(details(f), max_width=300, min_width=220, parse_html=False) if clickable else None
            marker_kwargs = dict(location=[float(f["lat"]), float(f["lon"])], radius=rad, color="#ffffff", weight=1.5,
                                 fill=True, fill_color=color, fill_opacity=0.95, tooltip=html.escape(str(f.get("name", "Facility"))))
            if popup is not None:
                marker_kwargs["popup"] = popup
            folium.CircleMarker(**marker_kwargs).add_to(fg)
        fg.add_to(m)


@st.cache_resource
def community_store():
    from isagiph.community import Store
    return Store(ROOT / "data" / "community")
