"""Spatial accessibility: travel times, E2SFCA, and travel-time coverage.

All functions work on *sparse* origin-destination pairs (point i, facility j,
minutes) so the national problem (~42k barangays x ~1.3k hospitals) fits in memory.
Travel times can come from a road network (OSMnx / r5py, see
``network_travel_times``) or, as a documented fallback, from straight-line
distance scaled by a circuity factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


def to_local_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approximate planar coordinates in km (equirectangular, per-point cosine).

    Distortion is < 1% across the Philippines' latitude range; for production
    runs prefer projecting to UTM 51N (EPSG:32651) with geopandas.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = lon * KM_PER_DEG_LON_EQ * np.cos(np.radians(lat))
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def fallback_travel_times(points: pd.DataFrame, facilities: pd.DataFrame,
                          max_minutes: float, speed_kmh: float = 15.0,
                          circuity: float = 1.4) -> pd.DataFrame:
    """Estimate sparse travel times where no usable road-network route exists.

    The effective speed is an explicit, configurable planning assumption intended
    to represent generalized local travel rather than private-car-only movement.
    ``minutes = straight-line km * circuity / speed_kmh * 60``. Water crossings,
    transit schedules, fares, transfers and waiting time are not represented by
    this fallback, so results require sensitivity analysis.
    """
    pxy = to_local_km(points["lat"].to_numpy(), points["lon"].to_numpy())
    fxy = to_local_km(facilities["lat"].to_numpy(), facilities["lon"].to_numpy())
    max_km = max_minutes / 60.0 * speed_kmh / circuity
    tree = cKDTree(fxy)
    sdm = cKDTree(pxy).sparse_distance_matrix(tree, max_km, output_type="coo_matrix")
    minutes = sdm.data * circuity / speed_kmh * 60.0
    # sparse_distance_matrix drops exact-zero distances; restore co-located pairs.
    zero_i, zero_j = _zero_distance_pairs(pxy, tree)
    rows = np.concatenate([sdm.row, zero_i])
    cols = np.concatenate([sdm.col, zero_j])
    mins = np.concatenate([minutes, np.zeros(len(zero_i))])
    out = pd.DataFrame({
        "point_id": points["point_id"].to_numpy()[rows],
        "facility_id": facilities["facility_id"].to_numpy()[cols],
        "minutes": mins,
    })
    return out.drop_duplicates(["point_id", "facility_id"])


def _zero_distance_pairs(pxy: np.ndarray, tree: cKDTree) -> tuple[np.ndarray, np.ndarray]:
    hits = tree.query_ball_point(pxy, r=1e-9)
    i = np.repeat(np.arange(len(hits)), [len(h) for h in hits])
    j = np.concatenate([np.asarray(h, dtype=int) for h in hits]) if len(i) else np.array([], int)
    return i.astype(int), j.astype(int)


def network_travel_times(points: pd.DataFrame, facilities: pd.DataFrame,
                         graphml_path: str, max_minutes: float) -> pd.DataFrame:
    """Road-network travel times with OSMnx (optional dependency).

    Builds a drive graph from a saved GraphML file (see docs/methodology.md for
    how to create it per region), snaps points and facilities to nodes, and runs
    a cut-off Dijkstra from each facility. Returns the same sparse schema as
    ``fallback_travel_times``.
    """
    import networkx as nx  # noqa: WPS433 (optional import)
    import osmnx as ox

    G = ox.load_graphml(graphml_path)
    routing = getattr(ox, "routing", ox)          # OSMnx 2.x moved these helpers
    G = routing.add_edge_speeds(G)
    G = routing.add_edge_travel_times(G)
    p_nodes = ox.distance.nearest_nodes(G, points["lon"], points["lat"])
    f_nodes = ox.distance.nearest_nodes(G, facilities["lon"], facilities["lat"])
    node_to_points: dict[int, list] = {}
    for pid, n in zip(points["point_id"], p_nodes):
        node_to_points.setdefault(n, []).append(pid)
    records = []
    Gr = G.reverse(copy=False)  # facility -> point distances on a directed graph
    for fid, fn in zip(facilities["facility_id"], f_nodes):
        lengths = nx.single_source_dijkstra_path_length(
            Gr, fn, cutoff=max_minutes * 60, weight="travel_time")
        for node, sec in lengths.items():
            for pid in node_to_points.get(node, []):
                records.append((pid, fid, sec / 60.0))
    return pd.DataFrame(records, columns=["point_id", "facility_id", "minutes"])


def gaussian_decay(minutes: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """Gaussian distance-decay weight: 1 at 0 minutes, 0 at the catchment limit d0."""
    minutes = np.asarray(minutes, float)
    d0 = np.asarray(d0, float)
    base = np.exp(-0.5)
    w = (np.exp(-0.5 * (minutes / d0) ** 2) - base) / (1.0 - base)
    return np.where(minutes <= d0, np.clip(w, 0.0, 1.0), 0.0)


def e2sfca(points: pd.DataFrame, facilities: pd.DataFrame, tt: pd.DataFrame,
           supply_col: str = "beds", d0_col: str = "d0") -> pd.Series:
    """Enhanced two-step floating catchment area (Luo & Qi, 2009).

    Step 1: R_j = S_j / sum_k P_k W(d_kj) over points within facility j's catchment.
    Step 2: A_i = sum_j R_j W(d_ij).
    Returns A_i (supply units per person) indexed by point_id. Each facility
    carries its own catchment ``d0`` (minutes), so hospital levels can differ.
    """
    df = tt.merge(points[["point_id", "population"]], on="point_id") \
           .merge(facilities[["facility_id", supply_col, d0_col]], on="facility_id")
    df["w"] = gaussian_decay(df["minutes"].to_numpy(), df[d0_col].to_numpy())
    df = df[df["w"] > 0]
    demand = (df["population"] * df["w"]).groupby(df["facility_id"]).sum()
    supply = facilities.set_index("facility_id")[supply_col]
    ratio = (supply / demand).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["R"] = df["facility_id"].map(ratio)
    access = (df["R"] * df["w"]).groupby(df["point_id"]).sum()
    return access.reindex(points["point_id"]).fillna(0.0)


def nearest_minutes(points: pd.DataFrame, tt: pd.DataFrame,
                    facility_ids: pd.Series | None = None) -> pd.Series:
    """Minutes to the nearest facility (optionally a subset); inf when none in range."""
    sub = tt if facility_ids is None else tt[tt["facility_id"].isin(set(facility_ids))]
    m = sub.groupby("point_id")["minutes"].min()
    return m.reindex(points["point_id"]).fillna(np.inf)


def assign_d0(facilities: pd.DataFrame, d0_l1: float, d0_l2: float,
              multiplier: float = 1.0) -> pd.Series:
    """Catchment per facility: Level 1 -> d0_l1, Level 2/3 -> d0_l2 (scaled by multiplier)."""
    return np.where(facilities["level"] >= 2, d0_l2, d0_l1) * multiplier


def lgu_accessibility(points: pd.DataFrame, facilities: pd.DataFrame,
                      tt: pd.DataFrame, cfg: dict, d0_multiplier: float = 1.0) -> pd.DataFrame:
    """Aggregate point-level access to LGU indicators used in the Capacity Score."""
    acc = cfg["accessibility"]
    fac = facilities.copy()
    fac["d0"] = assign_d0(fac, acc["d0_minutes"]["level1"],
                          acc["d0_minutes"]["level2plus"], d0_multiplier)
    pts = points.copy()
    pts["A_all"] = e2sfca(pts, fac, tt).to_numpy()
    l2 = fac[fac["level"] >= 2]
    pts["A_l2"] = e2sfca(pts, l2, tt[tt["facility_id"].isin(l2["facility_id"])]).to_numpy()

    cov = acc["coverage_minutes"]
    pts["t_any"] = nearest_minutes(pts, tt).to_numpy()
    pts["t_l2"] = nearest_minutes(pts, tt, l2["facility_id"]).to_numpy()
    mal_ids = fac.loc[fac.get("malasakit", 0) == 1, "facility_id"]
    pts["t_mal"] = nearest_minutes(pts, tt, mal_ids).to_numpy()

    P = pts["population"]
    g = pts.assign(
        beds_all=pts["A_all"] * P,
        beds_l2=pts["A_l2"] * P,
        pop_any=P * (pts["t_any"] <= cov["any_hospital"]),
        pop_l2=P * (pts["t_l2"] <= cov["level2plus"]),
        pop_mal=P * (pts["t_mal"] <= cov["malasakit"]),
        tP=np.where(np.isfinite(pts["t_any"]), pts["t_any"], 0.0) * P,
        pop_reach=np.where(np.isfinite(pts["t_any"]), P, 0.0),
    ).groupby("psgc")[["population", "beds_all", "beds_l2", "pop_any", "pop_l2",
                       "pop_mal", "tP", "pop_reach"]].sum()

    out = pd.DataFrame(index=g.index)
    out["accessible_beds"] = g["beds_all"]
    out["beds_per_1000_all"] = 1000 * g["beds_all"] / g["population"]
    out["beds_per_1000_l2plus"] = 1000 * g["beds_l2"] / g["population"]
    out["cov_any_30"] = g["pop_any"] / g["population"]
    out["cov_l2plus_60"] = g["pop_l2"] / g["population"]
    out["cov_malasakit_60"] = g["pop_mal"] / g["population"]
    # Mean minutes among residents with any hospital inside the search radius.
    out["mean_minutes_any"] = g["tP"] / g["pop_reach"].replace(0, np.nan)
    out["pct_no_hospital_in_range"] = 1 - g["pop_reach"] / g["population"]
    return out.reset_index()
