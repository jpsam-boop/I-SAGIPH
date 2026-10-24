"""Road-network travel times for one region, from a local roads GeoPackage (no download needed).

Usage:
    python -m isagiph.roadtimes --list                        # region names
    python -m isagiph.roadtimes "Region V (Bicol Region)"     # one region
    python -m isagiph.roadtimes --all                         # every region (skips ones already done)
    python -m isagiph.roadtimes --all --redo                  # every region again, even ones already done

Put an OpenStreetMap roads GeoPackage in the source-data folder (paths.manual), e.g. the HOT/HDX
"roads" export (column `highway`) or a Geofabrik roads layer (column `fclass`). The file is found
automatically, or set `roads.file` in config.yaml.

For each region the script reads only the roads inside the region's bounding box (plus a margin so
hospitals just across the border count), builds a road graph in which roads meet wherever they share a
vertex, assigns a driving speed to each road type, and routes from every hospital to every barangay
within 90 minutes (SciPy Dijkstra). Results go to data/interim/road_times_<region>.csv; the pipeline
uses them for that region's barangays and keeps straight-line estimates elsewhere (and for barangays
more than 2 km from any road, such as small islands without road links). A comparison with the
straight-line estimates is written to reports/road_vs_straight_<region>.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .accessibility import fallback_travel_times, to_local_km
from .config import load_config

CUTOFF_MIN = 90          # largest catchment used anywhere (60 min x 1.5 in the sensitivity analysis)
SNAP_KM = 2.0            # barangays/hospitals farther than this from a road keep straight-line estimates
ACCESS_KMH = 15.0        # speed for the short trip between a barangay/hospital point and the nearest road
MIN_COMPONENT = 200      # ignore tiny disconnected road fragments when snapping

DEFAULT_SPEEDS = {       # km/h by OSM road class (HOT `highway` or Geofabrik `fclass` values)
    "motorway": 80, "motorway_link": 50, "trunk": 60, "trunk_link": 40, "primary": 50, "primary_link": 35,
    "secondary": 40, "secondary_link": 30, "tertiary": 30, "tertiary_link": 25, "unclassified": 25,
    "road": 25, "residential": 20, "living_street": 15, "service": 15, "track": 12, "track_grade1": 15,
    "track_grade2": 12, "track_grade3": 10, "track_grade4": 8, "track_grade5": 6,
}
EXCLUDE = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian", "corridor", "construction",
           "proposed", "platform", "elevator", "escalator", "abandoned", "razed", "bus_stop", "rest_area"}


# --------------------------------------------------------------------------- the road graph (pure arrays)
def graph_from_lines(coords: np.ndarray, line_idx: np.ndarray, speed_kmh: np.ndarray, decimals: int = 6):
    """Build an undirected road graph from line vertices.

    coords: (N, 2) lon/lat of all vertices of all lines, in order; line_idx: (N,) which line each vertex
    belongs to; speed_kmh: speed for each line. Roads are joined wherever they share a vertex (OSM ways
    share nodes at junctions). Returns (csr matrix of travel seconds, node lon/lat array).
    """
    key = np.round(coords, decimals)
    nodes, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    same = line_idx[1:] == line_idx[:-1]
    u, v = inv[:-1][same], inv[1:][same]
    keep = u != v
    u, v = u[keep], v[keep]
    li = line_idx[:-1][same][keep]
    xy = to_local_km(nodes[:, 1], nodes[:, 0])
    km = np.hypot(*(xy[u] - xy[v]).T)
    sec = km / speed_kmh[li] * 3600.0
    a, b = np.minimum(u, v), np.maximum(u, v)
    e = pd.DataFrame({"a": a, "b": b, "s": sec}).groupby(["a", "b"], as_index=False)["s"].min()
    n = len(nodes)
    G = sparse.coo_matrix((e["s"].to_numpy() + 1e-6, (e["a"].to_numpy(), e["b"].to_numpy())), shape=(n, n)).tocsr()
    return G, nodes


def route(G: sparse.csr_matrix, nodes: np.ndarray, points: pd.DataFrame, facilities: pd.DataFrame,
          cutoff_min: float = CUTOFF_MIN, access_speed_kmh: float = ACCESS_KMH) -> tuple[pd.DataFrame, pd.Series]:
    """Road-route minutes plus assumed access/egress legs; also return an on-network flag.

    Road-class speed profiles and ``access_speed_kmh`` are explicit transport assumptions
    used to approximate mixed local travel conditions.
    """
    ncomp, label = connected_components(G, directed=False)
    sizes = np.bincount(label)
    usable = np.where(sizes[label] >= MIN_COMPONENT)[0]
    if len(usable) == 0:
        usable = np.arange(len(nodes))
    tree = cKDTree(to_local_km(nodes[usable, 1], nodes[usable, 0]))
    pd_km, pi = tree.query(to_local_km(points["lat"].to_numpy(), points["lon"].to_numpy()))
    fd_km, fi = tree.query(to_local_km(facilities["lat"].to_numpy(), facilities["lon"].to_numpy()))
    pnode, fnode = usable[pi], usable[fi]
    on_net = pd_km <= SNAP_KM
    if access_speed_kmh <= 0:
        raise ValueError("Road access/egress speed must be positive.")
    p_access = pd_km / access_speed_kmh * 60.0
    recs = []
    limit = cutoff_min * 60.0
    for k, (fid, fn, fkm) in enumerate(zip(facilities["facility_id"].to_numpy(), fnode, fd_km)):
        if fkm > SNAP_KM:
            continue
        dist = dijkstra(G, directed=False, indices=int(fn), limit=limit)
        mins = dist[pnode] / 60.0 + p_access + fkm / access_speed_kmh * 60.0
        ok = on_net & np.isfinite(mins) & (mins <= cutoff_min)
        if ok.any():
            recs.append(pd.DataFrame({"point_id": points["point_id"].to_numpy()[ok], "facility_id": fid,
                                      "minutes": mins[ok]}))
        if (k + 1) % 50 == 0:
            print(f"  routed {k + 1}/{len(facilities)} hospitals", flush=True)
    tt = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["point_id", "facility_id", "minutes"])
    return tt, pd.Series(on_net, index=points["point_id"].to_numpy())


# --------------------------------------------------------------------------- reading the GeoPackage
def find_roads_file(cfg: dict) -> tuple[Path, str | None]:
    """Find the roads GeoPackage, never a health-facilities GeoPackage.

    Set ``roads.file`` (and optionally ``roads.layer``) in config/config.yaml for a
    fully explicit selection. Automatic selection only accepts a line/generic layer
    with road-related evidence; if two candidates are equally plausible, fail loudly
    rather than silently routing on the wrong GeoPackage.
    """
    rcfg = cfg.get("roads", {})
    manual = Path(cfg["paths"]["manual"])

    configured = str(rcfg.get("file") or "").strip()
    if configured:
        p = Path(configured).expanduser()
        p = p if p.is_absolute() else manual / p
        if re.search(r"facilit|hospital|clinic", p.name, re.I):
            raise ValueError(
                f"Configured roads.file points to a health-facility-style file ({p.name!r}). "
                "That file is for facility inventory/coordinate matching, not road routing. "
                "Set roads.file to the actual Philippines roads GeoPackage."
            )
        if not p.is_file():
            raise FileNotFoundError(
                f"Configured roads GeoPackage does not exist: {p}. "
                "Set roads.file to the actual roads .gpkg filename in config/config.yaml."
            )
        layer = str(rcfg.get("layer") or "").strip() or None
        if layer is not None:
            available = _gpkg_layers(p)
            match = next((n for n, _ in available if n == layer), None)
            if match is None:
                raise ValueError(
                    f"Configured road layer {layer!r} was not found in {p.name}. "
                    f"Available layers: {', '.join(n for n, _ in available) or '(none)'}"
                )
            geom = next(g for n, g in available if n == match)
            if not _is_line_geometry(geom):
                raise ValueError(
                    f"Configured layer {layer!r} in {p.name} is {geom!r}, not a line/generic geometry layer. "
                    "Choose the roads line layer in roads.layer."
                )
        else:
            layer = _line_layer(p)
            if layer is None:
                raise ValueError(
                    f"No line roads layer could be identified in configured file {p}. "
                    "Check that this is the roads GeoPackage, then set roads.layer explicitly."
                )
        print(f"Selected roads GeoPackage: {p.resolve()} | layer: {layer}", flush=True)
        return p, layer

    candidates = []
    gpkg_files = sorted(manual.glob("*.gpkg"))
    for p in gpkg_files:
        name = p.name.lower()
        # This package is only for matching licensed facility names to OSM coordinates.
        # It must never be used to build a road graph, even if it has other layers.
        if "facilit" in name or "hospital" in name or "clinic" in name:
            continue
        layer = _line_layer(p)
        if layer is None:
            continue
        fields = _layer_fields(p, layer)
        field_set = {f.lower() for f in fields}
        road_class_fields = {"highway", "fclass", "road_type", "road_class", "roadtype", "type"}
        has_road_class = bool(field_set & road_class_fields)
        road_filename = bool(re.search(r"road|roads|highway|transport|street|network", name, re.I))
        road_layer = bool(re.search(r"road|roads|highway|transport|street|network|ways", layer, re.I))
        if not (has_road_class or road_filename or road_layer):
            continue
        score = (4 if has_road_class else 0) + (3 if road_filename else 0) + (2 if road_layer else 0)
        candidates.append((score, p, layer, has_road_class, road_filename, road_layer))

    if not candidates:
        available = ", ".join(p.name for p in gpkg_files) or "(no .gpkg files found)"
        raise FileNotFoundError(
            f"Could not confidently identify a roads GeoPackage in {manual}. Found: {available}. "
            "Put the roads .gpkg in data/manual and set roads.file to its exact filename in "
            "config/config.yaml. Do not point roads.file at health_facilities.gpkg."
        )

    candidates.sort(key=lambda item: (-item[0], item[1].name.lower()))
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    if len(best) > 1:
        names = ", ".join(f"{p.name} (layer {layer!r})" for _, p, layer, *_ in best)
        raise ValueError(
            f"Multiple possible roads GeoPackages were found: {names}. "
            "To prevent routing with the wrong file, set roads.file (and roads.layer if needed) "
            "in config/config.yaml to the exact roads dataset."
        )

    _, p, layer, *_ = best[0]
    print(f"Selected roads GeoPackage: {p.resolve()} | layer: {layer}", flush=True)
    return p, layer


def _gpkg_layers(path: Path) -> list[tuple[str, str]]:
    """List GeoPackage layers and geometry types, raising context-rich errors."""
    try:
        import pyogrio
        return [(str(n), str(g or "").lower()) for n, g in pyogrio.list_layers(path)]
    except Exception as exc:
        raise RuntimeError(f"Could not inspect GeoPackage {path}: {exc}") from exc


def _is_line_geometry(geometry_type: str) -> bool:
    g = str(geometry_type or "").lower()
    return "line" in g or g in ("", "geometry", "unknown", "geometrycollection")


def _layer_fields(path: Path, layer: str) -> list[str]:
    try:
        import pyogrio
        return [str(f) for f in pyogrio.read_info(path, layer=layer)["fields"]]
    except Exception as exc:
        print(f"  could not inspect fields for {path.name}:{layer}: {exc}", flush=True)
        return []


def _line_layer(path: Path) -> str | None:
    """Choose a line/generic layer, preferring road-class fields and road-like names."""
    try:
        layers = _gpkg_layers(path)
    except Exception as exc:
        print(f"  could not list layers in {path.name}: {exc}", flush=True)
        return None

    usable = []
    road_class_fields = {"highway", "fclass", "road_type", "road_class", "roadtype", "type"}
    for name, geom in layers:
        if not _is_line_geometry(geom):
            continue
        fields = {f.lower() for f in _layer_fields(path, name)}
        has_class = bool(fields & road_class_fields)
        named_road = bool(re.search(r"road|roads|highway|transport|street|network|ways", name, re.I))
        usable.append((name, 2 if has_class else 0, 1 if named_road else 0))
    if not usable:
        return None
    usable.sort(key=lambda item: (-item[1], -item[2], item[0].lower()))
    return usable[0][0]

def read_roads(path: Path, layer: str | None, bbox: tuple[float, float, float, float], speeds: dict, exclude: set):
    """(coords, line_idx, speed_per_line) for drivable roads inside bbox = (lon_min, lat_min, lon_max, lat_max)."""
    import geopandas as gpd
    import pyogrio
    import shapely

    fields = pyogrio.read_info(path, layer=layer)["fields"]
    cls = next((f for f in fields if f.lower() in ("highway", "fclass", "road_type", "type")), None)
    gdf = gpd.read_file(path, layer=layer, bbox=bbox, columns=[cls] if cls else None)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    kind = gdf[cls].astype(str).str.lower().str.strip() if cls else pd.Series("road", index=gdf.index)
    gdf = gdf[~kind.isin(exclude)].copy()
    kind = kind.loc[gdf.index]
    gdf = gdf.assign(_speed=kind.map(lambda k: speeds.get(k, speeds.get("default", 20))).astype(float))
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "LineString"]
    coords, idx = shapely.get_coordinates(gdf.geometry.to_numpy(), return_index=True)
    return coords, idx, gdf["_speed"].to_numpy()


# --------------------------------------------------------------------------- comparison and main
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def compare(points: pd.DataFrame, straight: pd.DataFrame, road: pd.DataFrame, lgus: pd.DataFrame) -> str:
    def nearest(tt):
        return tt.groupby("point_id")["minutes"].min().reindex(points["point_id"]).fillna(np.inf)

    s, r = nearest(straight).to_numpy(), nearest(road).to_numpy()
    w = points["population"].to_numpy()
    both = np.isfinite(s) & np.isfinite(r)
    ratio = np.median(r[both] / np.maximum(s[both], 0.5)) if both.any() else np.nan
    cov_s = (w * (s <= 30)).sum() / max(w.sum(), 1)
    cov_r = (w * (r <= 30)).sum() / max(w.sum(), 1)
    lines = [
        f"- Barangays compared: {len(points):,} ({int(both.sum()):,} reach a hospital both ways within {CUTOFF_MIN} min).",
        f"- Median road time is **{ratio:.1f}×** the straight-line estimate.",
        f"- Residents within 30 min of a hospital: **{100 * cov_s:.1f}%** (straight-line estimate) vs "
        f"**{100 * cov_r:.1f}%** (road travel time).",
        "- The map's profile panel shows, for each city or municipality, whether its travel times come from roads "
        "or from straight-line estimates."]
    return "\n".join(lines)


def run_region(region: str, cfg: dict, lg: pd.DataFrame, pts: pd.DataFrame, fac: pd.DataFrame,
               roads: tuple[Path, str | None]) -> None:
    rcfg = cfg.get("roads", {})
    speeds = {**DEFAULT_SPEEDS, **(rcfg.get("speeds_kmh") or {})}
    exclude = set(rcfg.get("exclude", EXCLUDE))
    rp = pts[pts["psgc"].isin(lg.loc[lg["region"] == region, "psgc"])]
    pad = 0.5                                        # include hospitals just across the region's border
    lat0, lat1 = rp["lat"].min() - pad, rp["lat"].max() + pad
    lon0, lon1 = rp["lon"].min() - pad, rp["lon"].max() + pad
    rf = fac[fac["lat"].between(lat0, lat1) & fac["lon"].between(lon0, lon1)]
    print(f"{region}: {len(rp):,} barangays, {len(rf):,} hospitals. Reading roads from {roads[0].name}...", flush=True)
    coords, idx, spd = read_roads(roads[0], roads[1], (lon0, lat0, lon1, lat1), speeds, exclude)
    G, nodes = graph_from_lines(coords, idx, spd)
    print(f"Road graph: {len(nodes):,} junctions/vertices, {G.nnz:,} road segments. Routing...", flush=True)
    road, on_net = route(G, nodes, rp, rf, access_speed_kmh=float(rcfg.get("access_speed_kmh", ACCESS_KMH)))
    out = cfg["paths"]["interim"] / f"road_times_{slug(region)}.csv"
    road.to_csv(out, index=False)
    # every barangay the road network reaches (also those with no hospital within 90 min by road)
    pd.Series(on_net[on_net].index, name="point_id").to_csv(
        cfg["paths"]["interim"] / f"road_points_{slug(region)}.csv", index=False)
    acc = cfg["accessibility"]
    straight = fallback_travel_times(rp, rf, CUTOFF_MIN, acc["fallback_speed_kmh"], acc["circuity"])
    covered = rp[rp["point_id"].isin(on_net[on_net].index)]
    report = (f"# Road vs straight-line travel times: {region}\n\n"
              f"{len(covered):,} of {len(rp):,} barangays are within {SNAP_KM:g} km of a road; the rest keep "
              f"straight-line estimates using the configured {float(acc['fallback_speed_kmh']):g} km/h effective-speed assumption. "
              f"Road-class speeds plus {float(rcfg.get('access_speed_kmh', ACCESS_KMH)):g} km/h access/egress assumptions are used; "
              "These travel times are estimates derived from the road network and the configured transport assumptions.\n\n" + compare(covered, straight, road, lg))
    (cfg["paths"]["reports"] / f"road_vs_straight_{slug(region)}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}.")


def main(argv: list[str]) -> None:
    cfg = load_config()
    lg = pd.read_csv(cfg["paths"]["interim"] / "lgus.csv", dtype={"psgc": str})
    regions = sorted(lg["region"].unique())
    if not argv or argv[0] == "--list":
        print("\n".join(regions))
        return
    pts = pd.read_csv(cfg["paths"]["interim"] / "points.csv", dtype={"psgc": str})
    fac = pd.read_csv(cfg["paths"]["interim"] / "facilities.csv", dtype={"psgc": str})
    roads = find_roads_file(cfg)
    if argv[0] == "--all":
        todo = regions
    else:
        name = argv[0]
        hits = [r for r in regions if r == name] or [r for r in regions if name.lower() in r.lower()]
        if len(hits) != 1:
            raise SystemExit(f"Region not found or ambiguous: {name}. Use --list to see names.")
        todo = hits
    for region in todo:
        out = cfg["paths"]["interim"] / f"road_times_{slug(region)}.csv"
        if argv[0] == "--all" and "--redo" not in argv and out.exists() \
                and (cfg["paths"]["interim"] / f"road_points_{slug(region)}.csv").exists():
            print(f"Already done: {region}")
            continue
        try:
            run_region(region, cfg, lg, pts, fac, roads)
        except MemoryError:
            print(f"FAILED (out of memory): {region}. Try again on its own, or close other programs.")
    print("\nDone. Run `python -m isagiph.pipeline` to use the road times.")


if __name__ == "__main__":
    main(sys.argv[1:])
