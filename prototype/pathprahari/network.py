"""
Road network construction.

In deployment this module loads a real OpenStreetMap graph (via OSMnx) and enriches every
edge with public-data features. The prototype builds a synthetic corridor network with the
SAME feature schema so the whole pipeline runs offline and reproducibly.

Edge feature schema (identical for OSM and synthetic graphs):
    length_km      : segment length                         (OSM geometry)
    road_class     : 'NH' | 'SH' | 'MDR'                     (OSM highway tag)
    speed_kmph     : free-flow speed                         (OSM maxspeed / class default)
    curvature      : sum |heading change| per km, degrees    (OSM geometry)
    gradient_pct   : absolute grade                          (SRTM 30 m DEM)
    junction_deg   : max node degree at segment ends         (OSM topology)
    lux_index      : 0-1 night-light radiance (lighting)     (NASA VIIRS Black Marble)
    blackspot      : 1 if within 500 m of a MoRTH blackspot  (MoRTH / state lists)
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

# Prototype corridor bounding box (roughly Varanasi -> Prayagraj belt, used only for
# realistic coordinates; all features below are synthetic).
LAT0, LAT1 = 25.25, 25.55
LON0, LON1 = 81.85, 83.00

TOWNS = [  # (lon, lat, size) -> brightness bumps in the synthetic VIIRS field
    (82.98, 25.32, 1.0),   # origin city (east)
    (81.88, 25.44, 1.0),   # destination city (west)
    (82.40, 25.40, 0.55),  # mid-corridor town
    (82.65, 25.29, 0.35),
    (82.15, 25.50, 0.35),
]


def lux_at(lon: float, lat: float) -> float:
    """Synthetic night-light radiance in [0,1]; bright near towns, dark in between."""
    v = 0.04
    for tlon, tlat, s in TOWNS:
        d = math.hypot((lon - tlon) * 100, (lat - tlat) * 111)  # km
        v += s * math.exp(-(d / 6.0) ** 2)
    return float(min(1.0, v))


def build_corridor(n_cols: int = 34, n_rows: int = 11, seed: int = 7) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    lons = np.linspace(LON0, LON1, n_cols)
    lats = np.linspace(LAT0, LAT1, n_rows)
    mid = n_rows // 2

    # Nodes with jitter
    for i in range(n_cols):
        for j in range(n_rows):
            jl = 0 if j == mid else rng.normal(0, 0.006)
            jt = 0 if j == mid else rng.normal(0, 0.006)
            # NH spine gently meanders
            lat = lats[j] + (0.03 * math.sin(i / 4.0) if j == mid else 0) + jt
            G.add_node((i, j), lon=float(lons[i] + jl), lat=float(lat))

    # "Ghat / hilly" pocket with high curvature + gradient (south-central)
    ghat_center = (82.47, 25.40)  # hilly, dark stretch straddling the NH

    def add(u, v, road_class):
        a, b = G.nodes[u], G.nodes[v]
        dx = (a["lon"] - b["lon"]) * 100.6
        dy = (a["lat"] - b["lat"]) * 111.0
        length = math.hypot(dx, dy) * (1.0 + (0.08 if road_class != "NH" else 0.02))
        mlon, mlat = (a["lon"] + b["lon"]) / 2, (a["lat"] + b["lat"]) / 2
        dg = math.hypot((mlon - ghat_center[0]) * 100, (mlat - ghat_center[1]) * 111)
        ghat = math.exp(-(dg / 6.0) ** 2)
        base_curv = {"NH": 8, "SH": 22, "MDR": 35}[road_class]
        curvature = max(0.0, rng.gamma(2.0, base_curv / 2.0) + 230 * ghat)
        gradient = max(0.0, rng.normal(0.8, 0.4) + 5.5 * ghat)
        speed = {"NH": 70, "SH": 54, "MDR": 36}[road_class] * (1 - 0.35 * ghat)
        lux = lux_at(mlon, mlat)
        G.add_edge(u, v, length_km=length, road_class=road_class, speed_kmph=speed,
                   curvature=curvature, gradient_pct=gradient, lux_index=lux,
                   ghat=ghat, mid_lon=mlon, mid_lat=mlat,
                   u_lon=a["lon"], u_lat=a["lat"], v_lon=b["lon"], v_lat=b["lat"])

    for i in range(n_cols):
        for j in range(n_rows):
            if i + 1 < n_cols:
                rc = "NH" if j == mid else ("SH" if j in (mid - 3, mid + 3) else "MDR")
                if rc == "MDR" and rng.random() < 0.18:
                    continue  # sparser village roads
                add((i, j), (i + 1, j), rc)
            if j + 1 < n_rows:
                rc = "SH" if i % 6 == 0 else "MDR"
                if rc == "MDR" and rng.random() < 0.35:
                    continue
                add((i, j), (i, j + 1), rc)

    # State-highway bypass skirting the hilly NH stretch (plain terrain, longer, slower).
    # This is the classic dispatcher dilemma: straight-through NH vs. bypass.
    by_row = mid - 2
    ia = int(np.argmin(np.abs(lons - (ghat_center[0] + 0.15))))
    ib = int(np.argmin(np.abs(lons - (ghat_center[0] - 0.15))))
    chain = [((ia, mid), (ia, mid - 1)), ((ia, mid - 1), (ia, by_row))]
    chain += [((i, by_row), (i - 1, by_row)) for i in range(ia, ib, -1)]
    chain += [((ib, by_row), (ib, mid - 1)), ((ib, mid - 1), (ib, mid))]
    for u, v in chain:
        add(u, v, "SH")
        d = G.edges[u, v]
        d.update(curvature=float(rng.gamma(2.0, 6.0)), gradient_pct=float(abs(rng.normal(0.7, 0.3))),
                 speed_kmph=60.0, ghat=0.0, lux_index=max(d["lux_index"], 0.30), bypass=True)

    # keep largest component
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    by_nodes = {n for e in chain for n in e}
    for u, v, d in G.edges(data=True):
        d.setdefault("bypass", False)
        d["near_bypass"] = bool(d["bypass"] or u in by_nodes or v in by_nodes)
        d["junction_deg"] = max(G.degree(u), G.degree(v))
        d["travel_min"] = 60.0 * d["length_km"] / d["speed_kmph"]
    for k, (u, v) in enumerate(G.edges()):
        G.edges[u, v]["seg_id"] = k
    return G


def origin_destination(G: nx.Graph):
    """East-most and west-most NH spine nodes."""
    mid = max(j for _, j in G.nodes) // 2
    spine = [n for n in G.nodes if n[1] == mid]
    return max(spine), min(spine)
