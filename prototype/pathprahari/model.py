"""
Risk hotspot detection + uncertainty-aware segment risk model.

1. Hotspots  : severity- and recency-weighted density clustering of telematics events
               (grid pre-aggregation -> weighted DBSCAN, so it scales to millions of events).
2. Prior     : Poisson GLM on PUBLIC road features (curvature, gradient, junctions, lighting,
               road class) with exposure offset. This is the cold-start model - it scores
               any road in India even when a fleet has never driven it.
3. Posterior : Empirical-Bayes Poisson-Gamma update (Highway Safety Manual style) using the
               fleet's own incidents + surrogate events (harsh braking, near misses).
               Gives a posterior mean AND a 90% credible interval per segment.
4. Context   : learned multipliers for hour-of-day, fatigue window, darkness, fog and rain.
5. Reasons   : per-segment reason codes for dispatchers and drivers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.linear_model import PoissonRegressor

SURROGATE_WEIGHT = {"harsh_brake": 1.0, "harsh_turn": 1.0, "harsh_accel": 0.6,
                    "near_miss": 3.0}


# ----------------------------------------------------------------------------- hotspots
def detect_hotspots(events: pd.DataFrame, now_month: int, seg_exposure_density=None,
                    half_life_months: float = 3.0, cell_m: float = 200.0, eps_m: float = 450.0,
                    top_quantile: float = 0.93, min_events: int = 40):
    """Severity x recency weighted, EXPOSURE-NORMALISED hotspots.

    Raw event counts simply mirror traffic volume (busy highways always "look" dangerous).
    Dividing each event's weight by the traffic density of its segment turns the score into
    risk *per vehicle passing*, which is what a dispatcher actually cares about.
    Returns (hotspot table, events with cluster labels)."""
    ev = events.copy()
    age = (now_month - ev["month"]).clip(lower=0)
    ev["w_raw"] = ev["severity"] * 0.5 ** (age / half_life_months)
    if seg_exposure_density is not None:
        dens = seg_exposure_density / seg_exposure_density.mean()
        ev["w"] = ev["w_raw"] / dens.reindex(ev["seg_id"]).to_numpy().clip(min=0.05)
    else:
        ev["w"] = ev["w_raw"]
    # grid pre-aggregation (~250 m cells) -> O(cells) clustering instead of O(events)
    ky = 111_000.0
    kx = 111_000.0 * np.cos(np.deg2rad(ev["lat"].mean()))
    ev["cx"] = np.floor(ev["lon"] * kx / cell_m).astype(int)
    ev["cy"] = np.floor(ev["lat"] * ky / cell_m).astype(int)
    cells = ev.groupby(["cx", "cy"]).agg(w=("w", "sum"), n=("w", "size"),
                                         lon=("lon", "mean"), lat=("lat", "mean")).reset_index()
    cells.loc[cells["n"] < 8, "w"] = 0.0   # ignore cells with too little evidence
    # only dense cells (top ~7% by severity x recency weight) can seed hotspots
    hot = cells["w"] >= cells["w"].quantile(top_quantile)
    X = np.c_[cells.loc[hot, "lon"] * kx, cells.loc[hot, "lat"] * ky]
    cells["cluster"] = -1
    cells.loc[hot, "cluster"] = DBSCAN(eps=eps_m, min_samples=1).fit(X).labels_
    ev = ev.merge(cells[["cx", "cy", "cluster"]], on=["cx", "cy"], how="left")
    hs = (ev[ev["cluster"] >= 0].groupby("cluster")
          .agg(lon=("lon", "mean"), lat=("lat", "mean"), risk_score=("w", "sum"),
               raw_weighted=("w_raw", "sum"),
               events=("w", "size"),
               incidents=("event_type", lambda s: int((s == "incident").sum())),
               near_misses=("event_type", lambda s: int((s == "near_miss").sum())),
               top_event=("event_type", lambda s: s.value_counts().index[0]),
               peak_hour=("hour", lambda s: int(s.value_counts().index[0])),
               main_segment=("seg_id", lambda s: int(s.value_counts().index[0])))
          .sort_values("risk_score", ascending=False).reset_index())
    ncell = (ev[ev["cluster"] >= 0].assign(c=lambda d: d["cx"].astype(str) + "_" + d["cy"].astype(str))
             .groupby("cluster")["c"].nunique())
    hs["extent_m"] = (hs["cluster"].map(ncell) * cell_m).astype(int)
    hs["risk_score"] = hs["risk_score"] / hs["cluster"].map(ncell)   # intensity per 200 m
    hs = hs[hs["events"] >= min_events].sort_values("risk_score", ascending=False)
    hs = hs.reset_index(drop=True)
    hs["hotspot_id"] = [f"HS-{i+1:03d}" for i in range(len(hs))]
    hs["rank"] = np.arange(1, len(hs) + 1)
    hs["risk_score"] = (100 * hs["risk_score"] / hs["risk_score"].max()).round(1)
    hs["hotspot_id"] = [f"HS-{i+1:03d}" for i in range(len(hs))]
    hs["rank"] = np.arange(1, len(hs) + 1)
    return hs, ev


# ----------------------------------------------------------------------------- model
FEATURES = ["curv", "grad", "junc4", "dark", "is_nh", "is_mdr"]


def feature_matrix(edges: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "curv": edges["curvature"] / 100.0,
        "grad": edges["gradient_pct"] / 5.0,
        "junc4": (edges["junction_deg"] >= 4).astype(float),
        "dark": 1 - edges["lux_index"],
        "is_nh": (edges["road_class"] == "NH").astype(float),
        "is_mdr": (edges["road_class"] == "MDR").astype(float),
    }, index=edges.index)


def _context_design(df: pd.DataFrame, curvnorm: np.ndarray) -> np.ndarray:
    h = df["hour"].to_numpy()
    night = ((h >= 22) | (h <= 5)).astype(float)
    fatigue = ((h >= 2) & (h <= 5)).astype(float)
    dark = df["dark"].to_numpy()
    fog = (df["weather"] == "fog").to_numpy().astype(float)
    rain = (df["weather"] == "rain").to_numpy().astype(float)
    return np.c_[night, night * dark, fatigue, fog, fog * night * dark, rain, rain * curvnorm]


CONTEXT_NAMES = ["night", "night_x_dark", "fatigue_2to5", "fog", "fog_x_night_x_dark",
                 "rain", "rain_x_curve"]


@dataclass
class RiskModel:
    k_dispersion: float = 1.5
    surrogate_weight: float = 1.0   # how much one incident-equivalent of harsh events counts
    prior_glm: PoissonRegressor | None = None
    ctx_glm: PoissonRegressor | None = None
    surrogate_scale: float = 0.0
    seg: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ---- fit
    def fit(self, edges, events, exposure, use_fleet_data: bool = True):
        X = feature_matrix(edges)
        # context-free exposure (mvkm) per segment
        E = exposure.groupby("seg_id")["mvkm"].sum().reindex(edges.index).fillna(0.0)
        inc = events[events["event_type"] == "incident"].groupby("seg_id").size() \
            .reindex(edges.index).fillna(0)
        sur = events[events["event_type"] != "incident"].assign(
            sw=lambda d: d["event_type"].map(SURROGATE_WEIGHT)).groupby("seg_id")["sw"].sum() \
            .reindex(edges.index).fillna(0)
        # surrogate events converted to incident-equivalents (fleet-wide ratio)
        self.surrogate_scale = float(inc.sum() / max(sur.sum(), 1.0))
        # incident-equivalent evidence; surrogate events are a second, noisier look at the same
        # rate, so they add both counts AND matching effective exposure (weight omega)
        w_s = self.surrogate_weight
        y_eq = inc + w_s * self.surrogate_scale * sur

        # context multipliers (learned) -------------------------------------------
        ex = exposure.copy()
        ex["dark"] = (1 - edges["lux_index"]).reindex(ex["seg_id"]).to_numpy()
        cn = np.clip(edges["curvature"].reindex(ex["seg_id"]).to_numpy() / 150, 0, 1)
        cnt = (events.assign(sw=events["event_type"].map(SURROGATE_WEIGHT).fillna(1.0)
                             .where(events["event_type"] != "incident", 1.0))
               .assign(sw=lambda d: np.where(d["event_type"] == "incident", 1.0,
                                             d["sw"] * self.surrogate_scale))
               .groupby(["seg_id", "hour", "month", "weather"])["sw"].sum())
        ex = ex.merge(cnt.rename("y").reset_index(), on=["seg_id", "hour", "month", "weather"],
                      how="left").fillna({"y": 0.0})
        # crude segment base rate for offset (smoothed)
        base = ((y_eq + 0.5) / (E + 0.5 / max(y_eq.sum() / E.sum(), 1e-9)))
        mu = ex["mvkm"].to_numpy() * base.reindex(ex["seg_id"]).to_numpy()
        Z = _context_design(ex, cn)
        keep = mu > 1e-9
        self.ctx_glm = PoissonRegressor(alpha=1e-4, max_iter=500)
        self.ctx_glm.fit(Z[keep], ex["y"].to_numpy()[keep] / mu[keep], sample_weight=mu[keep])
        # normalise so that "day, clear" == 1 and segment mean multiplier == 1
        ex["m"] = self._ctx_mult(Z)
        mbar = (ex.assign(me=ex["m"] * ex["mvkm"]).groupby("seg_id")["me"].sum()
                / ex.groupby("seg_id")["mvkm"].sum()).reindex(edges.index).fillna(1.0)
        E_adj = E * mbar * (1 + w_s)   # context-weighted exposure (+ surrogate share)

        # prior: public features only (cold start) --------------------------------
        mask = E_adj > 0
        self.prior_glm = PoissonRegressor(alpha=1e-3, max_iter=500)
        self.prior_glm.fit(X[mask], (y_eq[mask] / E_adj[mask]), sample_weight=E_adj[mask])
        prior = self.prior_glm.predict(X)

        seg = pd.DataFrame({"exposure_mvkm": E, "exposure_adj": E_adj, "incidents": inc,
                            "surrogate_events": sur, "evidence": y_eq, "prior_rate": prior},
                           index=edges.index)
        if use_fleet_data:
            a = self.k_dispersion + seg["evidence"]
            b = self.k_dispersion / seg["prior_rate"] + seg["exposure_adj"]
        else:
            a = pd.Series(self.k_dispersion, index=seg.index)
            b = self.k_dispersion / seg["prior_rate"]
        seg["post_rate"] = a / b
        seg["ci_lo"] = stats.gamma.ppf(0.05, a, scale=1 / b)
        seg["ci_hi"] = stats.gamma.ppf(0.95, a, scale=1 / b)
        seg["shrink_w"] = (self.k_dispersion / seg["prior_rate"]) / b  # weight on prior
        rel = (seg["ci_hi"] - seg["ci_lo"]) / seg["post_rate"]
        seg["confidence"] = np.select([rel < 1.0, rel < 2.2], ["High", "Medium"], "Low")
        seg["_a"], seg["_b"] = a, b
        self.seg = seg
        self._edges = edges
        return self

    def _ctx_mult(self, Z):
        z0 = np.zeros((1, Z.shape[1]))
        return np.exp(Z @ self.ctx_glm.coef_)  # intercept cancels (relative to day/clear)

    def context_multiplier(self, seg_ids, hours, weather):
        e = self._edges.loc[seg_ids]
        df = pd.DataFrame({"hour": np.asarray(hours) % 24, "dark": 1 - e["lux_index"].to_numpy(),
                           "weather": weather})
        cn = np.clip(e["curvature"].to_numpy() / 150, 0, 1)
        return self._ctx_mult(_context_design(df, cn))

    def context_effects(self):
        return pd.Series(np.exp(self.ctx_glm.coef_), index=CONTEXT_NAMES)

    # ---- reason codes
    def reasons(self, seg_id, hour=None, weather="clear", n=3):
        e = self._edges.loc[seg_id]
        s = self.seg.loc[seg_id]
        X = feature_matrix(self._edges)
        contrib = (X.loc[seg_id] - X.mean()) * self.prior_glm.coef_
        txt = {"curv": f"Sharp curves ({e['curvature']:.0f}°/km)",
               "grad": f"Steep gradient ({e['gradient_pct']:.1f}%)",
               "junc4": "Busy multi-arm junction",
               "dark": f"Poorly lit at night (light index {e['lux_index']:.2f})",
               "is_nh": "High-speed national highway",
               "is_mdr": "Narrow district road"}
        items = [(float(v), txt[k]) for k, v in contrib.items() if v > 0.05]
        fleet_lift = s["post_rate"] / s["prior_rate"]
        if fleet_lift > 1.4 and s["surrogate_events"] > 5:
            items.append((float(np.log(fleet_lift)),
                          f"Fleet harsh-event cluster ({int(s['surrogate_events'])} events, "
                          f"{fleet_lift:.1f}x expected)"))
        if hour is not None:
            m = float(self.context_multiplier([seg_id], [hour], weather)[0])
            if m > 1.3:
                label = {"fog": "Fog + night + darkness", "rain": "Rain on curves"}.get(
                    weather, "Night / fatigue window")
                items.append((float(np.log(m)), f"{label} ({m:.1f}x risk at {hour:02d}:00)"))
        items.sort(reverse=True)
        return [t for _, t in items[:n]]
