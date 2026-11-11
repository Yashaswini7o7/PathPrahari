"""
Simulated fleet telematics.

No public Indian fleet-telematics dataset with GPS-level harsh events exists, so the
prototype simulates 12 months of fleet operation on the network. The generator encodes
patterns documented in public sources (MoRTH "Road Accidents in India" reports):
night-time and fatigue risk, darker roads being riskier at night, winter fog on North
Indian highways, monsoon rain, junction conflicts - plus *hidden hotspots* that road
features cannot explain (median cuts, dhaba stretches with parked trucks, encroachment).
Those hidden hotspots are exactly what fleet behaviour data must reveal.

When a real fleet connects, this module is replaced by `ingest_telematics()` reading the
fleet's GPS/event export; every downstream module stays unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = ["harsh_brake", "harsh_turn", "harsh_accel", "near_miss"]
EVENT_SHARE = [0.48, 0.22, 0.18, 0.12]
SEVERITY = {"harsh_brake": 1.0, "harsh_turn": 1.0, "harsh_accel": 0.6, "near_miss": 3.0,
            "incident": 10.0}

# Hour-of-day traffic profile for freight (trucks run heavily at night in India)
HOUR_TRAFFIC = np.array([5, 5, 4.5, 4, 4, 4, 4.5, 5, 5, 4.5, 4, 4, 4, 4, 4, 4.5, 5, 5.5,
                         6, 6, 6, 6, 5.5, 5.5], dtype=float)
HOUR_TRAFFIC /= HOUR_TRAFFIC.sum()


def weather_for(month: int, rng) -> str:
    if month in (12, 1) and rng.random() < 0.45:
        return "fog"
    if month in (7, 8, 9) and rng.random() < 0.40:
        return "rain"
    return "clear"


def time_multiplier(hour, lux, weather, curvature_norm=0.0):
    """Ground-truth risk multiplier for time/lighting/weather (used ONLY to simulate)."""
    hour = np.asarray(hour)
    night = ((hour >= 22) | (hour <= 5)).astype(float)
    fatigue = ((hour >= 2) & (hour <= 5)).astype(float)
    m = 1.0 + night * (0.4 + 1.8 * (1 - lux)) + fatigue * 0.7
    if weather == "fog":
        m = m * (1.0 + night * 2.2 * (1 - lux) + 0.4)
    elif weather == "rain":
        m = m * (1.35 + 0.6 * curvature_norm)
    return m


def true_base_rate(edges: pd.DataFrame, rng, forced_hidden=None) -> np.ndarray:
    """Incidents per million vehicle-km (daytime, clear weather)."""
    curv = edges["curvature"].to_numpy() / 100.0
    grad = edges["gradient_pct"].to_numpy() / 5.0
    junc = (edges["junction_deg"].to_numpy() >= 4).astype(float)
    dark = 1 - edges["lux_index"].to_numpy()
    cls = edges["road_class"].map({"NH": 0.15, "SH": 0.0, "MDR": 0.25}).to_numpy()
    log_rate = (np.log(0.35) + 0.9 * curv + 0.45 * grad + 0.35 * junc + 0.30 * dark + cls
                + rng.normal(0, 0.18, len(edges)))
    hidden = np.zeros(len(edges))
    allowed = np.where(~edges["near_bypass"].to_numpy())[0]
    idx = rng.choice(allowed, size=max(3, int(0.06 * len(edges))), replace=False)
    if forced_hidden is not None:
        idx = np.unique(np.r_[idx, forced_hidden])
    hidden[idx] = rng.uniform(np.log(3), np.log(6), len(idx))
    edges["hidden_hotspot"] = 0
    edges.loc[edges.index[idx], "hidden_hotspot"] = 1
    return np.exp(log_rate + hidden)


def simulate(G, months: int = 12, seed: int = 11, harsh_per_incident: float = 55.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({"seg_id": d["seg_id"], "u": u, "v": v, **{k: d[k] for k in (
            "length_km", "road_class", "speed_kmph", "curvature", "gradient_pct",
            "junction_deg", "lux_index", "travel_min", "mid_lon", "mid_lat", "ghat",
            "u_lon", "u_lat", "v_lon", "v_lat", "bypass", "near_bypass")}})
    edges = pd.DataFrame(rows).set_index("seg_id", drop=False)
    # two hidden hotspots on the NH near the hilly stretch (median cut + truck-parking dhaba)
    nh = edges[edges["road_class"] == "NH"]
    forced = nh.iloc[(nh["mid_lon"] - 82.40).abs().argsort()[:1]].index.tolist() + \
        nh.iloc[(nh["mid_lon"] - 82.58).abs().argsort()[:1]].index.tolist()
    edges["true_rate"] = true_base_rate(edges, rng, forced_hidden=edges.index.get_indexer(forced))
    # where along each segment risk concentrates (point-like for hidden hotspots)
    edges["hot_t"] = rng.uniform(0.2, 0.8, len(edges))

    # Fleet exposure (million vehicle-km per month): NH busy, MDR sparse -> sparse data
    vol = edges["road_class"].map({"NH": 1.0, "SH": 0.22, "MDR": 0.03}).to_numpy()
    vol = vol * rng.lognormal(0, 0.5, len(edges))
    edges["mvkm_per_month"] = vol * edges["length_km"].to_numpy() * 0.32

    curv_norm = np.clip(edges["curvature"].to_numpy() / 150, 0, 1)
    events, incidents, exposure_rows = [], [], []
    for m in range(1, months + 1):
        cal_month = ((m - 1 + 10) % 12) + 1  # data starts in November
        weather_by_day = [weather_for(cal_month, rng) for _ in range(30)]
        w_counts = pd.Series(weather_by_day).value_counts(normalize=True)
        for w, wshare in w_counts.items():
            expo_h = np.outer(edges["mvkm_per_month"].to_numpy() * wshare, HOUR_TRAFFIC)
            mult = time_multiplier(np.arange(24)[None, :], edges["lux_index"].to_numpy()[:, None],
                                   w, curv_norm[:, None])
            lam_inc = expo_h * edges["true_rate"].to_numpy()[:, None] * mult
            n_inc = rng.poisson(lam_inc)
            n_evt = rng.poisson(lam_inc * harsh_per_incident * rng.lognormal(0, 0.15, lam_inc.shape))
            exposure_rows.append(pd.DataFrame({"seg_id": np.repeat(edges.index.to_numpy(), 24),
                                               "hour": np.tile(np.arange(24), len(edges)),
                                               "month": m, "weather": w,
                                               "mvkm": expo_h.ravel()}))
            for kind, mat, store in (("incident", n_inc, incidents), ("event", n_evt, events)):
                si, hi = np.nonzero(mat)
                cnt = mat[si, hi]
                si, hi = np.repeat(si, cnt), np.repeat(hi, cnt)
                if len(si) == 0:
                    continue
                hid = edges["hidden_hotspot"].to_numpy()[si] == 1
                t = np.where(hid & (rng.random(len(si)) < 0.75),
                             np.clip(edges["hot_t"].to_numpy()[si] + rng.normal(0, 0.03, len(si)), 0, 1),
                             rng.random(len(si)))
                gps = 0.00015  # ~15 m GPS noise
                lon = (edges["u_lon"].to_numpy()[si] * (1 - t) + edges["v_lon"].to_numpy()[si] * t
                       + rng.normal(0, gps, len(si)))
                lat = (edges["u_lat"].to_numpy()[si] * (1 - t) + edges["v_lat"].to_numpy()[si] * t
                       + rng.normal(0, gps, len(si)))
                df = pd.DataFrame({"seg_id": edges.index.to_numpy()[si], "hour": hi, "month": m,
                                   "weather": w, "lon": lon, "lat": lat})
                if kind == "event":
                    df["event_type"] = rng.choice(EVENT_TYPES, len(df), p=EVENT_SHARE)
                else:
                    df["event_type"] = "incident"
                store.append(df)
    ev = pd.concat(events, ignore_index=True)
    inc = pd.concat(incidents, ignore_index=True)
    expo = pd.concat(exposure_rows, ignore_index=True)
    allev = pd.concat([ev, inc], ignore_index=True)
    allev["severity"] = allev["event_type"].map(SEVERITY)
    allev["vehicle_id"] = rng.integers(1, 181, len(allev))
    allev["driver_cohort"] = pd.cut(rng.random(len(allev)), [0, .3, .75, 1.0],
                                    labels=["<2y", "2-7y", ">7y"]).astype(str)
    return edges, allev, expo
