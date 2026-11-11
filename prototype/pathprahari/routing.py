"""
Route risk scoring, alternative routes (risk-vs-time Pareto frontier), departure-time
optimisation and fatigue/darkness-aware halt suggestions.

Route risk = expected incidents per 10,000 trips
           = 1e4 * sum_e  rate_e(posterior, per M vkm) * length_e/1e6 * ctx(hour_e, weather)
where hour_e is the clock time the truck actually reaches segment e (departure time +
cumulative travel time), so the SAME road scores differently at 14:00 and at 03:00.
Uncertainty comes from Monte-Carlo sampling of each segment's Gamma posterior.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

SCALE = 1e4 / 1e6  # per 10k trips, rates are per million vehicle-km


@dataclass
class RouteScore:
    name: str
    nodes: list
    seg_ids: list
    travel_min: float
    length_km: float
    risk: float
    risk_lo: float
    risk_hi: float
    confidence: str
    seg_risk: np.ndarray
    seg_hours: np.ndarray


def path_segments(G, path):
    return [G.edges[a, b]["seg_id"] for a, b in zip(path[:-1], path[1:])]


def score_path(G, model, path, depart_hour, weather="clear", name="route", n_mc=400, rng=None):
    rng = rng or np.random.default_rng(0)
    segs = path_segments(G, path)
    e = model._edges.loc[segs]
    t_cum = np.cumsum(np.r_[0, e["travel_min"].to_numpy()[:-1]])
    hours = np.floor(depart_hour + (t_cum + e["travel_min"].to_numpy() / 2) / 60).astype(int) % 24
    ctx = model.context_multiplier(segs, hours, weather)
    s = model.seg.loc[segs]
    L = e["length_km"].to_numpy()
    seg_risk = s["post_rate"].to_numpy() * L * ctx * SCALE
    draws = rng.gamma(s["_a"].to_numpy()[None, :], 1 / s["_b"].to_numpy()[None, :],
                      size=(n_mc, len(segs)))
    tot = (draws * (L * ctx * SCALE)[None, :]).sum(1)
    lo, hi = np.percentile(tot, [5, 95])
    risk = float(seg_risk.sum())
    rel = (hi - lo) / risk
    conf = "High" if rel < 0.6 else ("Medium" if rel < 1.2 else "Low")
    return RouteScore(name, path, segs, float(e["travel_min"].sum()), float(L.sum()), risk,
                      float(lo), float(hi), conf, seg_risk, hours)


def candidate_routes(G, model, src, dst, depart_hour, weather="clear",
                     lambdas=(0, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280)):
    """Sweep lambda in cost = minutes + lambda * risk to trace the Pareto frontier."""
    # approximate clock time at each edge = departure + elapsed time along fastest path
    fastest = nx.shortest_path(G, src, dst, weight="travel_min")
    dist_from_src = nx.single_source_dijkstra_path_length(G, src, weight="travel_min")
    seg_ids, hours = [], []
    for a, b, d in G.edges(data=True):
        seg_ids.append(d["seg_id"])
        hours.append(int(depart_hour + min(dist_from_src[a], dist_from_src[b]) / 60) % 24)
    ctx = pd.Series(model.context_multiplier(seg_ids, hours, weather), index=seg_ids)
    rate = model.seg["post_rate"]
    paths = {tuple(fastest)}
    for lam in lambdas:
        def w(a, b, d, lam=lam):
            sid = d["seg_id"]
            return d["travel_min"] + lam * rate[sid] * d["length_km"] * ctx[sid] * SCALE * 10
        paths.add(tuple(nx.shortest_path(G, src, dst, weight=w)))
    scored = [score_path(G, model, list(p), depart_hour, weather) for p in paths]
    # Pareto filter
    pts = sorted(scored, key=lambda r: (r.travel_min, r.risk))
    front, best = [], np.inf
    for r in pts:
        if r.risk < best - 1e-12:
            front.append(r)
            best = r.risk
    return scored, front


def recommend(front, max_extra_frac=0.30):
    """Fastest / Balanced (knee) / Safest-within-reason."""
    fastest = min(front, key=lambda r: r.travel_min)
    ok = [r for r in front if r.travel_min <= fastest.travel_min * (1 + max_extra_frac)]
    safest = min(ok, key=lambda r: r.risk)
    if len(ok) <= 2:
        balanced = safest
    else:
        t = np.array([r.travel_min for r in ok]); k = np.array([r.risk for r in ok])
        tn = (t - t.min()) / max(np.ptp(t), 1e-9); kn = (k - k.min()) / max(np.ptp(k), 1e-9)
        balanced = ok[int(np.argmin(np.hypot(tn, kn)))]
    fastest.name, balanced.name, safest.name = "Fastest", "Balanced", "Safest"
    out = {"Fastest": fastest, "Balanced": balanced, "Safest": safest}
    return out


def departure_sweep(G, model, path, weather="clear", hours=range(24)):
    return pd.DataFrame([{"depart_hour": h, "risk": score_path(G, model, path, h, weather).risk}
                         for h in hours])


def halt_suggestion(G, model, route: RouteScore, lux_threshold=0.25):
    """If the truck would cross dark segments during 00-05h, suggest halting at the last
    well-lit point before the dark stretch (dhaba/town) until 05:00."""
    e = model._edges.loc[route.seg_ids]
    dark_night = (e["lux_index"].to_numpy() < lux_threshold) & (route.seg_hours <= 4)
    if not dark_night.any():
        return None
    first = int(np.argmax(dark_night))
    lit = np.where(e["lux_index"].to_numpy()[:first] >= lux_threshold)[0]
    if len(lit) == 0:
        return {"type": "delay_departure", "depart_after": "04:00",
                "dark_segments_avoided": int(dark_night.sum())}
    stop_idx = int(lit[-1])
    node = route.nodes[stop_idx + 1]
    t_cum = np.cumsum(model._edges.loc[route.seg_ids, "travel_min"].to_numpy())
    if t_cum[stop_idx] < 45:   # a halt within 45 min of the depot == just leave later
        return {"type": "delay_departure", "depart_after": "04:00",
                "dark_segments_avoided": int(dark_night.sum())}
    return {"type": "halt", "halt_node": node, "lon": G.nodes[node]["lon"],
            "lat": G.nodes[node]["lat"], "resume_at": "05:00",
            "dark_segments_avoided": int(dark_night.sum())}
