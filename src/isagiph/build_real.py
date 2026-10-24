"""Build the standard tables (lgus, points, facilities) from the team's real files.

Usage:  python -m isagiph.build_real
Put the original files in data/manual/ (names may vary; patterns below):

    phl_admin_boundaries.xlsx                 COD-AB admin attributes with centroids (HDX)
    4_Household*2020*CPH*.xlsx                2020 CPH household population by age group
    2_2023_SAE_with_PSGC*.xlsx                PSA 2023 city/municipal poverty estimates
    *gida*barangay*.pdf                       DOH 2023 GIDA barangay list
    Hospital_list_as_of*.xlsx                 DOH licensed hospitals (sheets Government, Private)
    List_of_Infirmary*.xlsx                   DOH licensed infirmaries
    List_of_Primary_Care_Facilities*.xlsx     DOH licensed primary care facilities
    Free-standing_Ambulance*.xlsx             DOH free-standing ambulance providers
    PCSO_Malasakit_Centers*.pdf               PCSO Malasakit Center list
    health_facilities.gpkg                    HOT OSM health facilities (for coordinates)

Outputs data/interim/{lgus,points,facilities}.csv and reports/data_quality.md,
which records every match rate and every record that could not be linked.
"""
from __future__ import annotations

import json
import re
import sqlite3
import struct
import subprocess
import unicodedata
import warnings
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

from .config import ROOT, load_config
from .schema import validate

MANUAL = load_config()["paths"]["manual"]
QUALITY: list[str] = []


def note(msg: str) -> None:
    print(msg)
    QUALITY.append(msg)


def find(pattern: str) -> Path:
    hits = sorted(MANUAL.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {MANUAL}")
    return hits[0]


# --------------------------------------------------------------------------- names
ABBREV = [(r"\bSTA\.?\s", "SANTA "), (r"\bSTO\.?\s", "SANTO "), (r"\bGEN\.?\s", "GENERAL "),
          (r"\bPRES\.?\s", "PRESIDENT "), (r"\bSN\.?\s", "SAN "), (r"\bMT\.?\s", "MOUNT ")]


def key(s, strip_city: bool = False) -> str:
    """Comparable key for place names: ASCII, upper case, no brackets/punctuation,
    'CITY OF X' -> 'X CITY'. With strip_city, the word CITY is removed entirely."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\(.*?\)", " ", s)
    for pat, rep in ABBREV:
        s = re.sub(pat, rep, s + " ")
    s = re.sub(r"^\s*CITY OF\s+(.*)$", r"\1 CITY", s.strip())
    s = re.sub(r"\bMUNICIPALITY OF\b", " ", s)
    if strip_city:
        s = re.sub(r"\bCITY\b", " ", s)
    return re.sub(r"[^A-Z0-9]", "", s)


REGION_PATTERNS = [  # (regex on any region label, COD adm1 pcodes)
    (r"\bNCR\b|NATIONAL CAPITAL", ["PH13"]), (r"\bCAR\b|CORDILLERA", ["PH14"]),
    (r"BARMM|BANGSAMORO|ARMM", ["PH19"]), (r"CARAGA|REGION 13\b|REGION XIII\b", ["PH16"]),
    (r"MIMAROPA|4-?B\b|IV-?B\b", ["PH17"]), (r"CALABARZON|4-?A\b|IV-?A\b", ["PH04"]),
    (r"\bNIR\b|NEGROS ISLAND", ["PH06", "PH07"]),
    (r"ILOCOS|REGION 1\b|REGION I\b|REG I\b", ["PH01"]), (r"CAGAYAN VALLEY|REGION 2\b|REGION II\b|REG II\b", ["PH02"]),
    (r"CENTRAL LUZON|REGION 3\b|REGION III\b|REG III\b", ["PH03"]), (r"BICOL|REGION 5\b|REGION V\b|REG V\b", ["PH05"]),
    (r"WESTERN VISAYAS|REGION 6\b|REGION VI\b|REG VI\b", ["PH06"]),
    (r"CENTRAL VISAYAS|REGION 7\b|REGION VII\b|REG VII\b", ["PH07"]),
    (r"EASTERN VISAYAS|REGION 8\b|REGION VIII\b|REG VIII\b", ["PH08"]),
    (r"ZAMBOANGA|REGION 9\b|REGION IX\b|REG IX\b", ["PH09"]),
    (r"NORTHERN MINDANAO|REGION 10\b|REGION X\b|REG X\b", ["PH10"]),
    (r"DAVAO|REGION 11\b|REGION XI\b|REG XI\b", ["PH11"]),
    (r"SOCCSKSARGEN|REGION 12\b|REGION XII\b|REG XII\b", ["PH12"]),
]


def region_codes(label) -> list[str]:
    s = str(label).upper().strip()
    for pat, codes in REGION_PATTERNS:
        if re.search(pat, s):
            return codes
    return []


class LguMatcher:
    """Resolve (region, province, city/municipality) text to a COD adm3 pcode."""

    def __init__(self, adm3: pd.DataFrame):
        self.a = adm3.assign(k=adm3["adm3_name"].map(key), ks=adm3["adm3_name"].map(lambda s: key(s, True)),
                             pk=adm3["adm2_name"].map(key))

    def province_keys(self, region: list[str], prov) -> list[str]:
        """Province keys for a label: exact, then prefix (MAGUINDANAO -> del Norte + del Sur), then fuzzy."""
        if prov is None or str(prov).strip() in ("", "nan"):
            return []
        pk = key(prov)
        sub = self.a if not region else self.a[self.a["adm1_pcode"].isin(region)]
        pks = sorted(set(sub["pk"]))
        if pk in pks:
            return [pk]
        pref = [k for k in pks if len(pk) >= 5 and k.startswith(pk)]
        if pref:
            return pref
        close = get_close_matches(pk, pks, n=1, cutoff=0.85)
        return close

    def province_key(self, region: list[str], prov) -> str | None:
        ks = self.province_keys(region, prov)
        return ks[0] if ks else None

    def _match_in(self, sub: pd.DataFrame, pks: list[str], name) -> str | None:
        scopes = ([sub[sub["pk"].isin(pks)]] if pks else []) + [sub]
        for scope in scopes:
            for col, k in (("k", key(name)), ("ks", key(name, True))):
                hit = scope[scope[col] == k]
                if len(hit) == 1:
                    return hit["adm3_pcode"].iloc[0]
        if pks:  # fuzzy, only inside the province
            scope = sub[sub["pk"].isin(pks)]
            close = get_close_matches(key(name, True), scope["ks"].tolist(), n=1, cutoff=0.85)
            if close:
                return scope.loc[scope["ks"] == close[0], "adm3_pcode"].iloc[0]
        return None

    def match(self, region: list[str], prov, name) -> str | None:
        sub = self.a if not region else self.a[self.a["adm1_pcode"].isin(region)]
        hit = self._match_in(sub, self.province_keys(region, prov), name)
        if hit is None and region:   # region labels change (e.g. Sulu left BARMM in 2024)
            hit = self._match_in(self.a, self.province_keys([], prov), name)
        return hit


def similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        return SequenceMatcher(None, a, b).ratio()


HOSP_STOP = r"\b(HOSPITAL|MEDICAL|CENTER|CENTRE|MEMORIAL|GENERAL|INC|CORP|THE|OF|AND|DISTRICT|PROVINCIAL|CITY)\b"


def hosp_key(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(HOSP_STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- loaders
def load_admin() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = find("phl_admin_boundaries*.xlsx")
    a3 = pd.read_excel(f, sheet_name="phl_admin3",
                       usecols=["adm3_name", "adm3_pcode", "adm2_name", "adm2_pcode", "adm1_name",
                                "adm1_pcode", "area_sqkm", "center_lat", "center_lon"])
    a4 = pd.read_excel(f, sheet_name="phl_admin4",
                       usecols=["adm4_name", "adm4_pcode", "adm3_pcode", "center_lat", "center_lon"])
    note(f"- Boundaries: {len(a3):,} cities/municipalities, {len(a4):,} barangays (COD-AB attributes).")
    return a3, a4


SHEET_REGION = {"NCR": "NCR", "CAR": "CAR", "Reg I": "REGION I", "Reg II": "REGION II", "Reg III": "REGION III",
                "CALABARZON": "CALABARZON", "MIMAROPA": "MIMAROPA", "Reg V": "REGION V", "Reg VI": "REGION VI",
                "Reg VII": "REGION VII", "Reg VIII": "REGION VIII", "Reg IX": "REGION IX", "Reg X": "REGION X",
                "Reg XI": "REGION XI", "Reg XII": "REGION XII", "CARAGA": "CARAGA", "BARMM": "BARMM"}
AGE_0_4 = {"0 - 4"}
AGE_GROUPS = {  # dashboard age pie chart: (lowest age, highest age) per group
    "pop_children": (0, 14), "pop_youth": (15, 24), "pop_young_adults": (25, 44),
    "pop_middle_age": (45, 59), "pop_elderly": (60, 200)}


def age_lower(label: str) -> int | None:
    m_ = re.match(r"^(\d+)", label)
    return int(m_.group(1)) if m_ else None
AGE_60P = {"60 - 64", "65 - 69", "70 - 74", "75 - 79", "80 years and over"}


def load_census(m: LguMatcher) -> pd.DataFrame:
    """Parse name -> Total -> age-group blocks and keep city/municipality blocks."""
    f = find("4_Household*.xlsx")
    rows, unmatched = [], []
    for sheet, rlabel in SHEET_REGION.items():
        reg = region_codes(rlabel)
        df = pd.read_excel(f, sheet_name=sheet, header=None, usecols=[0, 1])
        col0 = [str(v).strip() for v in df[0]]
        col1 = pd.to_numeric(df[1], errors="coerce").tolist()
        current_prov: list[str] = []
        sub = m.a[m.a["adm1_pcode"].isin(reg)]
        for i in range(len(col0) - 1):
            if col0[i + 1] != "Total" or col0[i] in ("nan", "Total"):
                continue
            name = col0[i]
            block = {}
            j = i + 1
            while j < len(col0) and (col0[j] == "Total" or re.match(r"^\d+ - \d+$|^80 years", col0[j])):
                block[col0[j]] = col1[j]
                j += 1
            k_full = key(name)
            in_prov = bool(current_prov) and (sub["pk"].isin(current_prov) &
                                              ((sub["k"] == k_full) | (sub["ks"] == key(name, True)))).any()
            pkeys = m.province_keys(reg, name)
            is_prov = bool(pkeys) and not in_prov and sheet != "NCR" and i > 4
            if is_prov:
                current_prov = pkeys
                # an independent city coded as its own "province" (e.g. City of Isabela)
                own = sub[sub["pk"].isin(pkeys)]
                if not (len(own) == 1 and own["k"].iloc[0] == k_full):
                    continue
            if i <= 4:   # region header block
                continue
            pc = m._match_in(sub, current_prov or [], name)
            if pc is None:
                unmatched.append(f"{sheet}: {name}")
                continue
            row = {"psgc": pc, "population": block.get("Total"),
                   "pop_0_4": sum(v for a, v in block.items() if a in AGE_0_4),
                   "pop_60_plus": sum(v for a, v in block.items() if a in AGE_60P)}
            for grp, (lo, hi) in AGE_GROUPS.items():
                row[grp] = sum(v for a, v in block.items()
                               if a != "Total" and age_lower(a) is not None and lo <= age_lower(a) <= hi)
            rows.append(row)
    out = pd.DataFrame(rows).groupby("psgc", as_index=False).sum()
    note(f"- Census 2020: {len(out):,} cities/municipalities matched; {len(unmatched)} blocks unmatched"
         + (f" (e.g. {', '.join(unmatched[:8])})" if unmatched else "") + ".")
    return out


def load_poverty(m: LguMatcher, a3: pd.DataFrame) -> pd.DataFrame:
    f = find("2_2023_SAE*.xlsx")
    s = pd.read_excel(f, header=None, skiprows=5).iloc[:, :6]
    s.columns = ["psgc6", "prov", "mun", "p2018", "p2021", "p2023"]
    s["psgc6"] = pd.to_numeric(s["psgc6"], errors="coerce")
    s["region_label"] = s["prov"].where(s["psgc6"].isna()).ffill()
    s["prov"] = s["prov"].where(s["psgc6"].notna()).ffill()
    s = s.dropna(subset=["psgc6", "p2023"]).copy()
    code = s["psgc6"].astype(int).astype(str).str.zfill(6)
    s["cod"] = "PH" + code.str[:2] + "0" + code.str[2:4] + code.str[4:6]
    ok = s["cod"].isin(set(a3["adm3_pcode"]))
    # Manila's 14 districts -> City of Manila (unweighted mean; district populations not in this file)
    manila = s["psgc6"].between(133901, 133914)
    s.loc[manila, "cod"] = "PH1303901"
    ok |= manila
    fixed = 0
    for idx in s.index[~ok]:
        pc = m.match(region_codes(s.at[idx, "region_label"]), s.at[idx, "prov"], s.at[idx, "mun"])
        if pc:
            s.at[idx, "cod"] = pc
            fixed += 1
    s = s[s["cod"].isin(set(a3["adm3_pcode"]))]
    out = s.groupby("cod", as_index=False)["p2023"].mean().rename(columns={"cod": "psgc"})
    out["poverty_incidence"] = out["p2023"] / 100.0
    note(f"- Poverty 2023 SAE: {len(out):,} LGUs ({fixed} matched by name after code mismatch). "
         "Highly urbanized cities are not in this file and are imputed (see below).")
    return out[["psgc", "poverty_incidence"]]


def load_gida(m: LguMatcher, a4: pd.DataFrame) -> pd.DataFrame:
    f = find("*gida*.pdf")
    txt = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(f), "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    recs = []
    for line in txt.splitlines():
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) == 4 and parts[0] not in ("REGION",) and region_codes(parts[0]):
            recs.append(parts)
    g = pd.DataFrame(recs, columns=["region", "province", "mun", "barangay"])
    cache: dict = {}
    g["psgc"] = [cache.setdefault((r, p, mu), m.match(region_codes(r), p, mu))
                 for r, p, mu in zip(g["region"], g["province"], g["mun"])]
    miss = g["psgc"].isna().sum()
    n_brgy = a4.groupby("adm3_pcode").size()
    cnt = g.dropna(subset=["psgc"]).groupby("psgc").size()
    out = pd.DataFrame({"psgc": n_brgy.index})
    out["gida_barangays"] = out["psgc"].map(cnt).fillna(0).astype(int)
    out["gida_share"] = (out["gida_barangays"] / out["psgc"].map(n_brgy)).clip(0, 1)
    note(f"- GIDA 2023: {len(g):,} GIDA barangays read; {len(g) - miss:,} linked to an LGU "
         f"({miss} unmatched); {int((out['gida_barangays'] > 0).sum()):,} LGUs have at least one GIDA barangay.")
    return out


def read_gpkg_points(path: Path) -> pd.DataFrame:
    """Read the OSM GeoPackage without GDAL: geometry blob -> envelope centre or WKB point."""
    con = sqlite3.connect(path)
    table = pd.read_sql("select table_name from gpkg_contents where data_type='features'", con)["table_name"][0]
    gcol = pd.read_sql("select column_name from gpkg_geometry_columns", con)["column_name"][0]
    df = pd.read_sql(f"select name, amenity, healthcare, adm3_pcode, {gcol} as g from '{table}'", con)
    lat, lon = [], []
    for blob in df["g"]:
        x = y = np.nan
        if blob is not None and blob[:2] == b"GP":
            flags = blob[3]
            env = (flags >> 1) & 0b111
            bo = "<" if flags & 1 else ">"
            if env in (1, 2, 3, 4):
                minx, maxx, miny, maxy = struct.unpack(bo + "4d", blob[8:40])
                x, y = (minx + maxx) / 2, (miny + maxy) / 2
            else:
                w = blob[8:]
                wbo = "<" if w[0] == 1 else ">"
                if struct.unpack(wbo + "I", w[1:5])[0] % 1000 == 1:
                    x, y = struct.unpack(wbo + "2d", w[5:21])
        lon.append(x)
        lat.append(y)
    df["lat"], df["lon"] = lat, lon
    return df.drop(columns="g").dropna(subset=["lat", "lon"])


MALASAKIT_REGION = {"1": "REGION 1", "2": "REGION 2", "3": "REGION 3", "4A": "REGION 4-A", "4B": "REGION 4-B",
                    "5": "REGION 5", "6": "REGION 6", "7": "REGION 7", "8": "REGION 8", "9": "REGION 9",
                    "10": "REGION 10", "11": "REGION 11", "12": "REGION 12", "NCR": "NCR", "CAR": "CAR",
                    "CARAGA": "CARAGA", "BARMM": "BARMM"}


def load_malasakit() -> pd.DataFrame:
    """Read the PCSO table by word position: each region label in the left column anchors
    a row, and the hospital-name lines in the right column are assigned to the nearest anchor."""
    import pdfplumber

    recs = []
    with pdfplumber.open(find("PCSO_Malasakit*.pdf")) as pdf:
        for page in pdf.pages:
            W = page.width
            words = page.extract_words()
            anchors = [w for w in words if w["x0"] < 0.22 * W and w["text"] in MALASAKIT_REGION and w["top"] > 150]
            right = [w for w in words if w["x0"] > 0.595 * W and w["top"] > 150]
            lines: dict[int, list] = {}
            for w in right:
                lines.setdefault(round(w["top"] / 3), []).append(w)
            cells: dict[int, list] = {}
            for _, ws in sorted(lines.items()):
                ws = sorted(ws, key=lambda w: w["x0"])
                yc = (ws[0]["top"] + ws[0]["bottom"]) / 2
                if not anchors or ws[0]["text"].startswith("Name"):
                    continue
                i = min(range(len(anchors)), key=lambda a: abs((anchors[a]["top"] + anchors[a]["bottom"]) / 2 - yc))
                if abs((anchors[i]["top"] + anchors[i]["bottom"]) / 2 - yc) < 45:
                    cells.setdefault(i, []).append(" ".join(w["text"] for w in ws))
            for i, txt in sorted(cells.items()):
                recs.append({"region": anchors[i]["text"], "name": " ".join(txt)})
    out = pd.DataFrame(recs)
    out["region_codes"] = out["region"].map(lambda r: region_codes(MALASAKIT_REGION.get(r, r)))
    return out


EMPTY = {"", "NAN", "NONE", "N/A", "NA", "-", "--", "0", "NOT APPLICABLE", "NOT AVAILABLE", "NULL"}


def has(v) -> bool:
    return str(v).strip().upper() not in EMPTY


def norm_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d.columns = [" ".join(str(c).split()).upper() for c in d.columns]
    return d


def col(d: pd.DataFrame, *starts: str) -> pd.Series:
    """First column whose normalised name starts with any of ``starts`` (else an empty series)."""
    for st_ in starts:                       # exact name first, then prefix
        if st_ in d.columns:
            return d[st_]
    for st_ in starts:
        for c in d.columns:
            if c.startswith(st_):
                return d[c]
    return pd.Series([np.nan] * len(d), index=d.index)


SERVICE_KEYWORDS = {"Dialysis": r"DIALYSIS|HEMODIALYSIS", "X-ray / imaging": r"X-?RAY|RADIOLOG|IMAGING|CT[ -]?SCAN|ULTRASOUND",
                    "Pharmacy": r"PHARMAC", "Birthing / maternity": r"BIRTHING|MATERNITY|LYING-?IN",
                    "Laboratory": r"LABORATOR", "Dental": r"DENTAL", "Blood service": r"BLOOD",
                    "Ambulance": r"AMBULANCE", "HIV testing": r"HIV"}


def ambulance_count(total, listing) -> float:
    """Ambulances per hospital: the numeric total if given, else one per plate number in the listing
    (only the count is kept; plate numbers are not stored)."""
    n = pd.to_numeric(total, errors="coerce")
    n = 0.0 if pd.isna(n) else float(n)
    if has(listing):
        txt = str(listing).upper()
        plates = len(re.findall(r"PLATE", txt))
        lead = re.match(r"\s*(\d+)", txt)
        n = max(n, plates if plates else (float(lead.group(1)) if lead and int(lead.group(1)) < 50 else 1.0))
    return n


def services_from_text(txt) -> set[str]:
    t = str(txt).upper() if has(txt) else ""
    return {name for name, pat in SERVICE_KEYWORDS.items() if re.search(pat, t)}


def urban_cores(a4: pd.DataFrame, a3: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
    """Each LGU's urban core: the barangay centre with the most other barangay centres within 2 km.

    Town centres are made of many small, densely packed barangays, so this lands downtown, whereas the
    geometric centre of a large city or municipality can fall in hills or farmland far from any hospital.
    """
    from scipy.spatial import cKDTree
    from .accessibility import to_local_km
    rows = []
    for pc, g in a4.groupby("adm3_pcode"):
        g = g.dropna(subset=["center_lat", "center_lon"])
        if g.empty:
            continue
        xy = to_local_km(g["center_lat"].to_numpy(), g["center_lon"].to_numpy())
        counts = np.array([len(x) for x in cKDTree(xy).query_ball_point(xy, radius_km)])
        order = np.lexsort(((g["area_sqkm"].to_numpy() if "area_sqkm" in g else np.zeros(len(g))), -counts))
        best = g.iloc[order[0]]
        rows.append((pc, float(best["center_lat"]), float(best["center_lon"])))
    core = pd.DataFrame(rows, columns=["adm3_pcode", "core_lat", "core_lon"]).set_index("adm3_pcode")
    return core.reindex(a3["adm3_pcode"]).fillna(
        a3.set_index("adm3_pcode")[["center_lat", "center_lon"]].rename(columns={"center_lat": "core_lat", "center_lon": "core_lon"}))


def ownership_class_label(value) -> str:
    """Normalize published ownership-class casing without changing the underlying ownership enum."""
    if not has(value):
        return ""
    raw = str(value).strip()
    compact = re.sub(r"[^A-Z]", "", raw.upper())
    if "DOHRETAINED" in compact:
        return "DOH-retained"
    if compact == "LGU":
        return "LGU"
    return raw.title()


def build_directory(m: LguMatcher, a3: pd.DataFrame, a4: pd.DataFrame | None = None) -> pd.DataFrame:
    """All licensed hospitals, infirmaries and primary care facilities, with services, for the Facility Finder."""
    recs = []
    hf = find("Hospital_list_as_of*.xlsx")
    for sheet in ("Government", "Private"):
        d = norm_cols(pd.read_excel(hf, sheet_name=sheet))
        if "NAME OF HOSPITAL" not in d.columns:
            d = d.rename(columns={"NAME OF FACILITY": "NAME OF HOSPITAL"})
        c_dial, c_blood, c_hiv = col(d, "NO OF DIALYSIS"), col(d, "BLOOD SERVICE"), col(d, "RHIV")
        c_lab, c_amb_n, c_amb = col(d, "CLINICAL LAB"), col(d, "TOTAL NO. OF AMBULANCE"), col(d, "AMBULANCE (NO.")
        for idx, r in d.iterrows():
            if not has(r.get("NAME OF HOSPITAL")):
                continue
            svc = services_from_text(r.get("ADD-ONS SERVICES"))
            dial = np.nan
            if has(c_dial.loc[idx]):
                svc.add("Dialysis")
                num = re.search(r"\d+", str(c_dial.loc[idx]))
                dial = float(num.group()) if num else np.nan
            if has(c_blood.loc[idx]):
                svc.add("Blood service")
            if has(c_hiv.loc[idx]):
                svc.add("HIV testing")
            if has(c_lab.loc[idx]):
                svc.add("Laboratory")
            amb = ambulance_count(c_amb_n.loc[idx], c_amb.loc[idx])
            if amb > 0:
                svc.add("Ambulance")
            cap = str(r.get("SERVCE CAPABILITY", ""))
            lvl = re.search(r"(\d)", cap)
            recs.append({"name": str(r["NAME OF HOSPITAL"]).strip(), "kind": "Hospital",
                         "level": f"Level {lvl.group(1)}" if lvl else cap.strip().title(),
                         "class": str(r.get("CLASS", "")).strip().title(), "ownership": sheet,
                         "ownership_class": ownership_class_label(r.get("OWNERSHIP CLASS")),
                         "region": r.get("REGION"), "province": r.get("PROVINCE"), "mun": r.get("CITY/ MUNICIPALITY"),
                         "address": r.get("BLDG OR HOUSE NO., STREET, BRGY (AS APPLICABLE)"),
                         "beds": pd.to_numeric(r.get("ABC"), errors="coerce"), "dialysis_stations": dial,
                         "ambulances": amb, "services": sorted(svc),
                         "contact": r.get("CONTACT NOS.") if has(r.get("CONTACT NOS.")) else "",
                         "license_valid_until": r.get("VALIDITY UNTIL")})
    inf = find("List_of_Infirmary*.xlsx")
    for sheet in ("Government", "Private"):
        d = norm_cols(pd.read_excel(inf, sheet_name=sheet))
        c_lab, c_amb = col(d, "CLINICAL LAB"), col(d, "AMBULANCE (NO.")
        for idx, r in d.iterrows():
            if not has(r.get("NAME OF FACILITY")):
                continue
            svc = services_from_text(r.get("ADD-ONS SERVICES"))
            if has(c_lab.loc[idx]):
                svc.add("Laboratory")
            amb_i = ambulance_count(None, c_amb.loc[idx])
            if amb_i > 0:
                svc.add("Ambulance")
            recs.append({"name": str(r["NAME OF FACILITY"]).strip(), "kind": "Infirmary", "level": "Infirmary",
                         "class": "General", "ownership": sheet, "ownership_class": "",
                         "region": r.get("REGION"), "province": r.get("PROVINCE"), "mun": r.get("CITY / MUNICIPALITY"),
                         "address": r.get("BLDG OR HOUSE NO., STREET, BRGY (AS APPLICABLE)"),
                         "beds": pd.to_numeric(r.get("ABC"), errors="coerce"), "dialysis_stations": np.nan,
                         "ambulances": amb_i, "services": sorted(svc),
                         "contact": r.get("CONTACT NOS.") if has(r.get("CONTACT NOS.")) else "",
                         "license_valid_until": r.get("VALIDITY UNTIL")})
    pcf = find("List_of_Primary_Care_Facilities*.xlsx")
    for sheet in ("Government", "Private"):
        d = norm_cols(pd.read_excel(pcf, sheet_name=sheet))
        flags = {"Ambulance": col(d, "AMBULANCE ONE-STOP", "AMBULANCE (NAME"), "Birthing / maternity": col(d, "BIRTHING HOME ONE-STOP", "BIRTHING HOME (NAME"),
                 "Laboratory": col(d, "CLINICAL LABORATORY ONE-STOP", "CLINICAL LABORATORY (SERVICE"), "Dental": col(d, "DENTAL CLINIC ONE-STOP", "DENTAL CLINIC (NAME"),
                 "Pharmacy": col(d, "PHARMACY ONE-STOP", "PHARMACY (NAME"), "X-ray / imaging": col(d, "X-RAY ONE-STOP", "X-RAY (NAME")}
        anc, c_contact, c_valid = col(d, "ANCILLARY"), col(d, "CONTACT NO"), col(d, "VALID UNTIL")
        for idx, r in d.iterrows():
            if not has(r.get("NAME OF HEALTH FACILITY")):
                continue
            svc = {k for k, s_ in flags.items() if has(s_.loc[idx])} | services_from_text(anc.loc[idx])
            recs.append({"name": str(r["NAME OF HEALTH FACILITY"]).strip(), "kind": "Primary care", "level": "Primary care",
                         "class": "", "ownership": sheet, "ownership_class": "",
                         "region": r.get("REGION"), "province": r.get("PROVINCE"), "mun": r.get("CITY/ MUNICIPALITY"),
                         "address": r.get("BRGY/ STREET"), "beds": np.nan, "dialysis_stations": np.nan, "ambulances": np.nan,
                         "services": sorted(svc), "contact": c_contact.loc[idx] if has(c_contact.loc[idx]) else "",
                         "license_valid_until": c_valid.loc[idx]})
    dfx = pd.DataFrame(recs).dropna(subset=["mun"])
    cache: dict = {}
    dfx["psgc"] = [cache.setdefault((r, p, mu), m.match(region_codes(r), p, mu))
                   for r, p, mu in zip(dfx["region"], dfx["province"], dfx["mun"])]
    miss = int(dfx["psgc"].isna().sum())
    dfx = dfx.dropna(subset=["psgc"]).reset_index(drop=True)

    # Coordinates: best OSM health facility match by name inside the same LGU, else the LGU's urban core
    # (densest cluster of barangay centres), else the LGU centroid.
    osm = read_gpkg_points(find("health_facilities.gpkg"))
    osm = osm[osm["name"].notna() & osm["adm3_pcode"].notna()].copy()
    osm["k"] = osm["name"].map(hosp_key)
    by_lgu = {pc: g for pc, g in osm.groupby("adm3_pcode")}
    cen = a3.set_index("adm3_pcode")[["center_lat", "center_lon"]]
    core = urban_cores(a4, a3) if a4 is not None else None
    lat, lon, method = [], [], []
    for _, r in dfx.iterrows():
        cands = by_lgu.get(r["psgc"])
        best = (0.0, None)
        if cands is not None:
            hk = hosp_key(r["name"])
            for _, c in cands.iterrows():
                s_ = similarity(hk, c["k"])
                if s_ > best[0]:
                    best = (s_, c)
        if best[1] is not None and best[0] >= 0.75:
            lat.append(best[1]["lat"]); lon.append(best[1]["lon"]); method.append("osm_name_match")
        else:
            if core is not None and r["psgc"] in core.index:
                lat.append(core.at[r["psgc"], "core_lat"]); lon.append(core.at[r["psgc"], "core_lon"]); method.append("lgu_urban_core")
            else:
                lat.append(cen.at[r["psgc"], "center_lat"]); lon.append(cen.at[r["psgc"], "center_lon"]); method.append("lgu_centroid")
    dfx["lat"], dfx["lon"], dfx["geocode"] = lat, lon, method

    # Malasakit Centers (PCSO list, Nov 2022): fuzzy name match among hospitals in the same region.
    mal = load_malasakit()
    dfx["malasakit"] = 0
    dfx["_k"] = dfx["name"].map(hosp_key)
    dfx["_reg"] = dfx["psgc"].str[:4]
    hosp = dfx["kind"] == "Hospital"
    linked = 0
    for _, r in mal.iterrows():
        pool = dfx[hosp & (dfx["_reg"].isin(r["region_codes"]) if r["region_codes"] else hosp)]
        sims = pool["_k"].map(lambda x: similarity(hosp_key(r["name"]), x))
        if len(sims) and sims.max() >= 0.8:
            dfx.loc[sims.idxmax(), "malasakit"] = 1
            linked += 1
    dfx["services"] = dfx["services"].map(lambda s_: "; ".join(s_))
    dfx["facility_id"] = ["F" + str(n).zfill(5) for n in range(1, len(dfx) + 1)]
    dfx["license_valid_until"] = pd.to_datetime(dfx["license_valid_until"], errors="coerce").dt.date
    note(f"- Facility directory: {len(dfx):,} licensed facilities ({(dfx.kind == 'Hospital').sum():,} hospitals, "
         f"{(dfx.kind == 'Infirmary').sum():,} infirmaries, {(dfx.kind == 'Primary care').sum():,} primary care); "
         f"{miss} not linked to an LGU; {method.count('osm_name_match'):,} located via OSM, {method.count('lgu_urban_core'):,} at their city/municipality's "
         f"urban core (densest cluster of barangays), {method.count('lgu_centroid'):,} at the LGU centroid; "
         f"{linked} of {len(mal)} PCSO Malasakit Centers linked; "
         f"{int(dfx['services'].str.contains('Dialysis').sum())} list dialysis.")
    return dfx.drop(columns=["_k", "_reg", "mun"])


def load_facilities(directory: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bedded facilities used for spatial access, derived from the directory."""
    h = directory[directory["kind"].isin(["Hospital"] + (["Infirmary"] if cfg.get("facilities", {}).get(
        "include_infirmaries", True) else []))].copy()
    specialty = h["class"].astype(str).str.upper().str.startswith("SPECIALTY")
    note(f"- Bedded facilities for access modelling: {int(specialty.sum())} specialty hospitals excluded from general "
         "bed access (e.g. the 4,200-bed National Center for Mental Health); infirmaries count as first-level beds.")
    h["lvl"] = h["level"].str.extract(r"(\d)")[0].astype(float)
    h.loc[h["kind"] == "Infirmary", "lvl"] = 1
    h = h[~specialty & h["beds"].gt(0) & h["lvl"].between(1, 3)].copy()
    h["level"] = h["lvl"].astype(int)
    h["kind"] = h["kind"].str.lower()
    return h[["facility_id", "name", "psgc", "lat", "lon", "level", "beds", "malasakit", "kind", "ownership", "geocode"]]


def out_dir_directory(cfg: dict) -> Path:
    return cfg["paths"]["interim"]


DISEASES = {"Tuberculosis": "tb", "Pneumonia": "pneumonia", "Hypertension": "hypertension",
            "Influenza": "influenza", "Leptospirosis": "leptospirosis"}
MORB_ALIASES = {"MT. PROVINCE": "Mountain Province", "WESTERN SAMAR": "Samar", "NORTHERN LEYTE": "Leyte",
                "MINDORO OCCIDENTAL": "Occidental Mindoro", "MINDORO ORIENTAL": "Oriental Mindoro",
                "NORTH COTABATO": "Cotabato", "PROVINCE OF DINAGAT": "Dinagat Islands",
                "LEGASPI CITY": "Legazpi City", "OZAMIS CITY": "Ozamiz City", "ISABELA CITY": "City of Isabela"}
REGION_ROW = re.compile(r"^(NCR|CAR|BARMM|CARAGA|REGION\s+\w+)$", re.I)


def read_morbidity(path: Path) -> pd.DataFrame:
    """Rows (disease, area, cases, rate per 100,000) from the DOH FHSIS 2022 morbidity tables.

    Only the pages that carry the 'Total Both Sexes - Number, Rate' columns are used; pdftotext's
    layout mode keeps each area name on the same line as its numbers.
    """
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    n_pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    rows = []
    for page in range(1, n_pages + 1):
        txt = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", "-f", str(page), "-l", str(page), str(path), "-"],
                             capture_output=True, check=True).stdout.decode("utf-8", "replace")
        head = re.search(r"Table 2\.H\.2\.\d+\s*-\s*([A-Za-z ,]+)", txt)
        if not head or "Rate" not in txt:
            continue
        disease = head.group(1).split(",")[0].strip()
        if disease not in DISEASES:
            continue
        for line in txt.splitlines():
            mm = re.match(r"^\s*(.+?)\s{2,}((?:[\d,]+\s+)*[\d,]+\s+[\d,]*\.\d+)\s*$", line)
            if mm:
                toks = mm.group(2).split()
                rows.append({"disease": DISEASES[disease], "area": mm.group(1).strip(),
                             "cases": int(toks[-2].replace(",", "")), "rate": float(toks[-1].replace(",", ""))})
    return pd.DataFrame(rows)


def load_morbidity(m: LguMatcher, a3: pd.DataFrame) -> pd.DataFrame:
    """Disease rates per 100,000 for every LGU: the city's own row if it has one, else its province's row.

    The DOH legend states that 0 means 'No Data', so zero totals are treated as missing (and later
    imputed with the regional median), not as zero disease.
    """
    f = find("Morbidity*.pdf")
    raw = read_morbidity(f)
    city, prov = {}, {}
    unmatched = set()
    region: list[str] = []
    for _, r in raw.iterrows():
        name = r["area"]
        if name.upper() == "PHILIPPINES":
            continue
        if REGION_ROW.match(name.strip()):
            region = region_codes(name.replace("Region 4A", "Region 4-A").replace("Region 4B", "Region 4-B"))
            continue
        val = r["rate"] if r["cases"] > 0 else np.nan
        alias = MORB_ALIASES.get(name.upper(), name)
        pks = [] if "CITY" in alias.upper() else m.province_keys(region, alias)
        if pks:
            for pk in pks:
                prov[(pk, r["disease"])] = val
            continue
        pc = m.match(region, None, alias)
        if pc:
            city[(pc, r["disease"])] = val
        else:
            unmatched.add(name)
    a = m.a.set_index("adm3_pcode")
    out = pd.DataFrame({"psgc": a.index})
    level = []
    for d in DISEASES.values():
        vals, lv = [], []
        for pc in out["psgc"]:
            v = city.get((pc, d), np.nan)
            src = "city"
            if pd.isna(v):
                v = prov.get((a.at[pc, "pk"], d), np.nan)
                src = "province"
            vals.append(v)
            lv.append(src if pd.notna(v) else "missing")
        out[f"rate_{d}"] = vals
        level.append(lv)
    out["disease_level"] = level[0]                 # level used for TB (city / province / missing)
    n_city = int((out["disease_level"] == "city").sum())
    miss = {d: int(out[f"rate_{d}"].isna().sum()) for d in DISEASES.values()}
    note(f"- Disease rates (DOH FHSIS 2022, per 100,000): {len(raw) // len(DISEASES)} areas per disease; "
         f"{n_city} LGUs use their own city row, the rest their province's. Areas reported as 0 ('No Data') "
         f"are imputed with the regional median: {', '.join(f'{k} {v}' for k, v in miss.items())} LGUs."
         + (f" Unmatched area names: {', '.join(sorted(unmatched))}." if unmatched else ""))
    return out


def count_by_lgu(m: LguMatcher, pattern: str, label: str, units_col: str | None = None) -> pd.Series:
    f = find(pattern)
    frames = []
    for sheet in ("Government", "Private"):
        d = pd.read_excel(f, sheet_name=sheet)
        d.columns = [" ".join(str(c).split()).upper() for c in d.columns]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    mun_col = next(c for c in d.columns if c.startswith("CITY"))
    d = d.dropna(subset=[mun_col])
    cache: dict = {}
    d["psgc"] = [cache.setdefault((r, p, mu), m.match(region_codes(r), p, mu))
                 for r, p, mu in zip(d["REGION"], d["PROVINCE"], d[mun_col])]
    w = pd.to_numeric(d[units_col], errors="coerce").fillna(1) if units_col else pd.Series(1, index=d.index)
    out = w.groupby(d["psgc"]).sum()
    note(f"- {label}: {len(d):,} records; {int(d['psgc'].notna().sum()):,} linked to an LGU.")
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    QUALITY.clear()
    QUALITY.append("# Data quality report (real data)\n")
    a3, a4 = load_admin()
    m = LguMatcher(a3)

    census = load_census(m)
    pov = load_poverty(m, a3)
    gida = load_gida(m, a4)
    directory = build_directory(m, a3, a4)
    fac = load_facilities(directory, cfg)
    morb = load_morbidity(m, a3)
    pcf = count_by_lgu(m, "List_of_Primary_Care_Facilities*.xlsx", "Primary care facilities (licensed)")
    amb = count_by_lgu(m, "Free-standing_Ambulance*.xlsx", "Free-standing ambulance providers", "NO. OF UNITS")

    lg = a3.rename(columns={"adm3_pcode": "psgc", "adm3_name": "name", "adm2_name": "province",
                            "adm1_name": "region", "center_lat": "lat", "center_lon": "lon"})
    lg = lg[["psgc", "name", "province", "region", "lat", "lon", "area_sqkm"]]
    lg = lg.merge(census, on="psgc", how="left").merge(pov, on="psgc", how="left") \
           .merge(gida[["psgc", "gida_share", "gida_barangays"]], on="psgc", how="left") \
           .merge(morb, on="psgc", how="left")
    no_pop = lg["population"].isna() | (lg["population"] <= 0)
    if no_pop.any():
        note(f"- Excluded {int(no_pop.sum())} LGUs with no 2020 census match (mostly created after 2020, e.g. "
             f"{', '.join(lg.loc[no_pop, 'name'].head(6))}).")
    lg = lg[~no_pop].copy()
    imp = lg["poverty_incidence"].isna()
    reg_med = lg.groupby("region")["poverty_incidence"].transform("median")
    lg["poverty_imputed"] = imp.astype(int)
    lg["poverty_incidence"] = lg["poverty_incidence"].fillna(reg_med).fillna(lg["poverty_incidence"].median())
    note(f"- Poverty imputed with the regional median for {int(imp.sum())} LGUs (mostly HUCs): "
         f"{', '.join(lg.loc[imp, 'name'].head(10))}{'...' if imp.sum() > 10 else ''}. Flag: poverty_imputed.")
    lg["gida_share"] = lg["gida_share"].fillna(0)
    for d in DISEASES.values():                     # 'No Data' areas -> regional median, flagged
        c = f"rate_{d}"
        lg[f"{c}_imputed"] = lg[c].isna().astype(int)
        lg[c] = lg[c].fillna(lg.groupby("region")[c].transform("median")).fillna(lg[c].median())
    from .impute import graph_impute               # replace regional medians where a validated model does better
    lg, imp_report = graph_impute(lg, [f"rate_{d}" for d in ("tb", "pneumonia", "hypertension")])
    (cfg["paths"]["interim"] / "imputation.json").write_text(json.dumps(imp_report, indent=2))
    note("- Missing disease rates (DOH 'No Data'): " + "; ".join(
        f"{c.replace('rate_', '')}: {r['method_used']} ({100 * r['error_reduction_vs_median']:.0f}% lower held-out error "
        f"than the regional median)" for c, r in imp_report.items()) + ".")
    lg["n_primary_care"] = lg["psgc"].map(pcf).fillna(0).astype(int)
    lg["n_ambulance_units"] = lg["psgc"].map(amb).fillna(0).astype(int)
    lg["pop_density"] = lg["population"] / lg["area_sqkm"]
    for c in ("population", "pop_0_4", "pop_60_plus", *AGE_GROUPS):
        lg[c] = lg[c].round().astype(int)

    # Demand points: barangay centroids; LGU population split equally across its barangays.
    pts = a4.merge(lg[["psgc", "population"]], left_on="adm3_pcode", right_on="psgc")
    pts["population"] = pts["population"] / pts.groupby("psgc")["psgc"].transform("size")
    pts = pts.rename(columns={"adm4_pcode": "point_id", "center_lat": "lat", "center_lon": "lon"})
    pts = pts[["point_id", "psgc", "lat", "lon", "population"]]
    note(f"- Demand points: {len(pts):,} barangay centroids; LGU population split equally across barangays "
         "(barangay-level 2020 population would refine this).")

    fac = fac[fac["psgc"].isin(lg["psgc"])]
    names = lg.set_index("psgc")["name"]
    directory["lgu"] = directory["psgc"].map(names)
    directory.to_csv(out_dir_directory(cfg) / "directory.csv", index=False)
    out = cfg["paths"]["interim"]
    validate(pts, "points").to_csv(out / "points.csv", index=False)
    validate(fac, "facilities").to_csv(out / "facilities.csv", index=False)
    validate(lg, "lgus").to_csv(out / "lgus.csv", index=False)
    note(f"\n**Final tables:** {len(lg):,} LGUs, {len(pts):,} demand points, {len(fac):,} bedded facilities "
         f"({int(fac['beds'].sum()):,} beds; Level 2+: {int((fac['level'] >= 2).sum())}; "
         f"Malasakit-linked: {int(fac['malasakit'].sum())}).")
    note("\nDisease rates are reported by DOH at province and city level, so municipalities share their "
         "province's rate.")
    acc = cfg["accessibility"]
    roads_cfg = cfg.get("roads", {})
    note("- Travel-time assumptions: the road-routing module uses explicit road-class speed assumptions plus a "
         f"{float(roads_cfg.get('access_speed_kmh', 15.0)):g} km/h access/egress assumption for short trips to/from mapped roads; "
         f"where route data are unavailable, the fallback uses an effective speed of {float(acc['fallback_speed_kmh']):g} km/h "
         f"and circuity {float(acc['circuity']):g}. These assumptions approximate mixed local transportation conditions; schedules, fares, transfers, waiting, and boat/ferry connections are not individually routed.")
    note("- Hospital access comparison: the pipeline estimates the population beyond a 30-minute hospital travel-time threshold from the facility inventory, population allocation, and configured transport assumptions. Where a DOH reference figure is supplied, it is compared with this estimate after calculation; check that scope and definitions align before interpreting differences.")
    try:                                   # official LGU outlines for the map, if a boundary file is present
        from .boundaries import build as build_boundaries
        build_boundaries()
        note("- Map boundaries: official city/municipality outlines written to data/processed/lgu_boundaries.geojson.")
    except FileNotFoundError:
        note(f"- Map boundaries: no boundary file in {MANUAL}, so the map uses hexagons.")
    (cfg["paths"]["reports"] / "data_quality.md").write_text("\n".join(QUALITY), encoding="utf-8")
    print(f"\nWrote data/interim/*.csv and reports/data_quality.md")


if __name__ == "__main__":
    main()
