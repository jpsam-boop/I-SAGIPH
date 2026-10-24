"""Official city/municipality outlines for the map (replaces the hexagon approximation).

Usage:  python -m isagiph.boundaries
Put an administrative-level-3 boundary file in data/manual/. Any of these work:
  * the COD-AB shapefile / geodatabase / GeoPackage download from HDX (zipped or unzipped),
  * a GeoJSON of level-3 boundaries (e.g. geoBoundaries PHL ADM3).
The script finds the level-3 layer, links every polygon to our LGU codes (by PSGC code when
present, otherwise by name and province), simplifies the outlines so the map stays fast, and
writes data/processed/lgu_boundaries.geojson (and data/interim/ for the pipeline). The dashboard
switches to true boundaries automatically when that file exists.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config

MANUAL = load_config()["paths"]["manual"]
LEVEL3 = re.compile(r"adm(in)?_?3|admbnda_adm3|level_?3|municipal", re.I)
GEO_SUFFIXES = (".zip", ".shp", ".gpkg", ".geojson", ".json", ".gdb")


# --------------------------------------------------------------------------- finding the file
def find_boundary_source() -> tuple[Path, str | None]:
    """(path, member-or-layer) of the first file in data/manual that holds level-3 polygons.

    For a .zip, the second value is the file *inside* the zip (e.g. 'phl_admin3.geojson');
    for a GeoPackage or geodatabase it is the layer name.
    """
    import zipfile

    cands = [p for p in sorted(MANUAL.iterdir())
             if (p.suffix.lower() in GEO_SUFFIXES or p.name.lower().endswith(".gdb.zip"))
             and not p.name.lower().endswith(".xlsx") and "facilit" not in p.name.lower()]
    for p in cands:                                   # single-layer files named for level 3
        if p.suffix.lower() in (".shp", ".geojson", ".json") and LEVEL3.search(p.name):
            return p, None
    for p in cands:                                   # zips: look inside for a level-3 file
        if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
            names = [n for n in zipfile.ZipFile(p).namelist()
                     if n.lower().endswith((".geojson", ".json", ".shp")) and not n.startswith("__MACOSX")]
            hit = [n for n in names if LEVEL3.search(Path(n).name)]
            if hit:
                return p, sorted(hit, key=lambda n: (not n.lower().endswith((".geojson", ".json")), len(n)))[0]
    for p in cands:                                   # GeoPackage / geodatabase: pick the level-3 layer
        if p.suffix.lower() in (".gpkg", ".gdb", ".zip"):
            try:
                import pyogrio
                layers = [str(row[0]) for row in pyogrio.list_layers(p)]
            except Exception:
                layers = []
            hit = [lyr for lyr in layers if LEVEL3.search(lyr)]
            if hit:
                return p, sorted(hit, key=len)[0]
    raise FileNotFoundError(
        f"No level-3 (city/municipality) boundary file found in {MANUAL}. Download the COD-AB "
        "boundaries for the Philippines from HDX (GeoJSON, shapefile, geodatabase, or GeoPackage).")


# --------------------------------------------------------------------------- reading
def read_polygons(path: Path, layer: str | None) -> pd.DataFrame:
    """Return a DataFrame with columns: pcode, name, province, geometry."""
    import zipfile

    p = Path(path)
    if p.suffix.lower() == ".zip" and layer and zipfile.is_zipfile(p):
        import tempfile
        if layer.lower().endswith((".geojson", ".json")):     # GeoJSON inside a zip: extract, then read
            tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(p) as z:
                z.extract(layer, tmp)
            return read_polygons(tmp / layer, None)
                                                              # shapefile inside a zip: extract its parts
        tmp = Path(tempfile.mkdtemp())
        stem = layer.rsplit(".", 1)[0]
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.rsplit(".", 1)[0] == stem:
                    z.extract(n, tmp)
        return _read_gis(tmp / layer, None)
    if p.suffix.lower() in (".geojson", ".json"):
        big = p.stat().st_size > 50e6
        try:
            return _read_gis(p, layer)                         # GDAL streams large files efficiently
        except ImportError:
            if big:
                raise RuntimeError(f"{p.name} is {p.stat().st_size / 1e6:,.0f} MB; install geopandas to read it.")
        return _from_geojson(json.loads(p.read_text(encoding="utf-8")))
    return _read_gis(p, layer)


def _from_geojson(gj: dict) -> pd.DataFrame:
    rows = [{**(f.get("properties") or {}), "geometry": f["geometry"]} for f in gj["features"] if f.get("geometry")]
    return _standardise(pd.DataFrame(rows))


def _read_gis(path: Path, layer: str | None) -> pd.DataFrame:
    import geopandas as gpd

    cols = None
    try:                                                       # read only the code/name columns we need
        import pyogrio
        fields = list(pyogrio.read_info(path, layer=layer)["fields"])
        wanted = [f for f in fields if re.fullmatch(
            r"adm[23]_?(pcode|code|name|en|name_?en)|psgc|shapename|name|province", f, re.I)]
        cols = wanted or None
    except Exception:
        pass
    print(f"  reading {Path(str(path)).name} (this can take a minute for large files)...", flush=True)
    kw = {"layer": layer} if layer else {}
    gdf = gpd.read_file(path, columns=cols, **kw) if cols else gpd.read_file(path, **kw)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["geometry"] = list(gdf.geometry)
    return _standardise(df)


def _pick(cols, *patterns):
    for pat in patterns:
        for c in cols:
            if re.fullmatch(pat, str(c), re.I):
                return c
    return None


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "geometry"]
    pcode = _pick(cols, r"adm3_?pcode", r"adm3_?code", r"psgc", r"code")
    name = _pick(cols, r"adm3_?(name|en)", r"adm3_?name_?en", r"shapename", r"name")
    prov = _pick(cols, r"adm2_?(name|en)", r"adm2_?name_?en", r"province")
    out = pd.DataFrame({"pcode": df[pcode].astype(str) if pcode else None,
                        "name": df[name].astype(str) if name else None,
                        "province": df[prov].astype(str) if prov else None})
    out["geometry"] = df["geometry"].to_numpy()
    return out


# --------------------------------------------------------------------------- linking to our LGUs
def link_to_lgus(polys: pd.DataFrame, a3: pd.DataFrame) -> pd.Series:
    """Our COD-AB adm3 pcode for each polygon: exact code, then same 7 PSGC digits, then name."""
    ours = set(a3["adm3_pcode"])
    by_digits = {re.sub(r"\D", "", c)[:7]: c for c in a3["adm3_pcode"]}
    out = []
    for pc in polys["pcode"].fillna(""):
        if pc in ours:
            out.append(pc)
            continue
        d = re.sub(r"\D", "", pc)
        out.append(by_digits.get(d[:7]) if len(d) >= 7 else None)
    out = pd.Series(out, index=polys.index, dtype=object)
    if out.isna().any() and polys["name"].notna().any():
        from .build_real import LguMatcher
        m = LguMatcher(a3)
        for i in out.index[out.isna()]:
            out[i] = m.match([], polys.at[i, "province"], polys.at[i, "name"])
    return out


# --------------------------------------------------------------------------- simplifying
def _dp(points: list, tol: float) -> list:
    """Douglas-Peucker line simplification (used only when shapely is unavailable)."""
    if len(points) < 4:
        return points
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    pts = np.asarray(points, float)
    ab = b - a
    if not ab.any():
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(np.cross(ab, pts - a)) / np.hypot(*ab)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [points[0], points[-1]]
    return _dp(points[: i + 1], tol)[:-1] + _dp(points[i:], tol)


def _round(coords, nd=5):
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(float(coords[0]), nd), round(float(coords[1]), nd)]
    return [_round(c, nd) for c in coords]


def simplify_geometries(geoms: list, tol: float) -> list[dict]:
    """Simplify all outlines together. With shapely >= 2.1, coverage simplification keeps shared
    borders identical (no gaps or overlaps between neighbouring LGUs)."""
    try:
        import shapely
        from shapely.geometry import mapping, shape

        shp = [g if hasattr(g, "geom_type") else shape(g) for g in geoms]
        shp = [shapely.make_valid(g) for g in shp]
        if hasattr(shapely, "coverage_simplify"):
            simp = shapely.coverage_simplify(np.array(shp, dtype=object), tol)
        else:
            simp = [g.simplify(tol, preserve_topology=True) for g in shp]
        return [_polygonal(mapping(g)) for g in simp]
    except ImportError:
        out = []
        for g in geoms:
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            new = []
            for poly in polys:
                rings = [r if len(r) < 8 else _dp(r[:-1], tol) + [r[0]] for r in poly]
                rings = [r for r in rings if len(r) >= 4]
                if rings:
                    new.append(rings)
            out.append({"type": "MultiPolygon", "coordinates": new})
        return out


def _polygonal(g: dict) -> dict:
    """Keep only polygon parts (make_valid can return collections with stray lines)."""
    if g["type"] in ("Polygon", "MultiPolygon"):
        return g
    if g["type"] == "GeometryCollection":
        parts = []
        for sub in g["geometries"]:
            if sub["type"] == "Polygon":
                parts.append(sub["coordinates"])
            elif sub["type"] == "MultiPolygon":
                parts.extend(sub["coordinates"])
        return {"type": "MultiPolygon", "coordinates": parts}
    return {"type": "MultiPolygon", "coordinates": []}


def _lite_and_mask(fc: dict, tol: float) -> dict:
    """Coarser outlines for the national map view (light enough for phones) and a mask that greys out
    everything outside the Philippines. Needs shapely; skipped without it."""
    try:
        import shapely
        from shapely.geometry import box, mapping, shape
    except ImportError:
        return {}
    shp = [shape(f["geometry"]) for f in fc["features"]]
    lite = simplify_geometries(shp, tol * 4)
    lite_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": f["properties"], "geometry": {"type": "MultiPolygon", "coordinates": _round(
            g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]], 4)}}
        for f, g in zip(fc["features"], lite)]}
    country = shapely.union_all([shapely.make_valid(g).buffer(0.003) for g in shp]).simplify(0.01)
    outside = box(105.0, -2.0, 140.0, 28.0).difference(country)
    mask = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                        "geometry": _polygonal(mapping(outside))}]}
    mask["features"][0]["geometry"]["coordinates"] = _round(mask["features"][0]["geometry"]["coordinates"], 4)
    print(f"Also wrote lighter outlines for the national view and the outside-the-Philippines map mask.")
    return {"lgu_boundaries_lite.geojson": lite_fc, "ph_mask.geojson": mask}


# --------------------------------------------------------------------------- main
def build(tol_deg: float | None = None) -> Path:
    cfg = load_config()
    tol = tol_deg if tol_deg is not None else cfg.get("boundaries", {}).get("simplify_deg", 0.001)
    from .build_real import load_admin
    a3, _ = load_admin()
    src, layer = find_boundary_source()
    print(f"Reading {src.name}" + (f" -> {layer}" if layer else ""))
    polys = read_polygons(src, layer)
    polys["psgc"] = link_to_lgus(polys, a3)
    linked = polys.dropna(subset=["psgc"])
    print(f"Linked {len(linked):,} of {len(polys):,} polygons to {linked['psgc'].nunique():,} LGUs "
          f"({len(set(a3['adm3_pcode']) - set(linked['psgc'])):,} LGUs without an outline).")
    # one feature per LGU (merge multi-part rows)
    print(f"Simplifying {len(linked):,} outlines to ~{tol * 111000:.0f} m (a few minutes for detailed files)...", flush=True)
    geoms = simplify_geometries(list(linked["geometry"]), tol)
    feats: dict[str, list] = {}
    for pc, g in zip(linked["psgc"], geoms):
        parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        feats.setdefault(pc, []).extend(parts)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"psgc": pc},
         "geometry": {"type": "MultiPolygon", "coordinates": _round(parts)}}
        for pc, parts in feats.items() if parts]}
    text = json.dumps(fc, separators=(",", ":"))
    extra = _lite_and_mask(fc, tol)
    for folder in (cfg["paths"]["interim"], cfg["paths"]["processed"]):
        (folder / "lgu_boundaries.geojson").write_text(text, encoding="utf-8")
        for name, obj in extra.items():
            (folder / name).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    out = cfg["paths"]["processed"] / "lgu_boundaries.geojson"
    print(f"Wrote {out} ({len(fc['features']):,} LGUs, {len(text) / 1e6:.1f} MB, simplified to ~{tol * 111000:.0f} m)")
    return out


def build_extras() -> None:
    """Only the light national outlines and the map mask, from an existing lgu_boundaries.geojson (seconds)."""
    cfg = load_config()
    tol = cfg.get("boundaries", {}).get("simplify_deg", 0.001)
    src = cfg["paths"]["processed"] / "lgu_boundaries.geojson"
    if not src.exists():
        raise SystemExit("Run `python -m isagiph.boundaries` (or build_real) first to create lgu_boundaries.geojson.")
    extra = _lite_and_mask(json.loads(src.read_text(encoding="utf-8")), tol)
    if not extra:
        raise SystemExit("shapely is needed for this step: pip install shapely")
    for folder in (cfg["paths"]["interim"], cfg["paths"]["processed"]):
        for name, obj in extra.items():
            (folder / name).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    import sys
    build_extras() if "--extras" in sys.argv else build()
