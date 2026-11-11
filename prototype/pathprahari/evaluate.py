"""
Performance evaluation.

* Temporal hold-out: fit on months 1-9, test on months 10-12 (no leakage).
* Compares four scorers:
    A. Raw crash counts          - what a dispatcher with an accident list would do
    B. Crash rate                - incidents / exposure, no smoothing, no surrogate events
    C. Public-data prior only    - PathPrahari cold start (zero fleet history)
    D. PathPrahari full          - empirical-Bayes prior + fleet incidents + harsh events
* Metrics: hotspot recovery (recall of truly riskiest 10% segments), rank correlation with
  true risk (all segments and data-sparse segments), predictive Poisson deviance on the
  held-out months, 90% credible-interval coverage, route-level rank correlation.
* Scalability: wall-clock of hotspot detection, model fit and route queries as event
  volume and network size grow.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .model import RiskModel, detect_hotspots
from .network import build_corridor, origin_destination
from .routing import candidate_routes
from .simulate import simulate


def _recall_top(score, truth, frac=0.10):
    k = max(1, int(round(frac * len(truth))))
    top_true = set(truth.sort_values(ascending=False).index[:k])
    top_pred = set(score.sort_values(ascending=False).index[:k])
    return len(top_true & top_pred) / k


def _deviance(y, mu):
    mu = np.clip(mu, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


def temporal_holdout(G, edges, events, exposure, split_month=9, rng_seed=3):
    tr_ev, te_ev = events[events["month"] <= split_month], events[events["month"] > split_month]
    tr_ex, te_ex = exposure[exposure["month"] <= split_month], exposure[exposure["month"] > split_month]
    full = RiskModel().fit(edges, tr_ev, tr_ex, use_fleet_data=True)
    cold = RiskModel().fit(edges, tr_ev, tr_ex, use_fleet_data=False)
    seg = full.seg
    E = seg["exposure_mvkm"].clip(lower=1e-9)
    scores = {
        "A. Raw crash counts": seg["incidents"] + 1e-6 * E,
        "B. Crash rate (no smoothing)": seg["incidents"] / E,
        "C. Public-data prior (cold start)": cold.seg["post_rate"],
        "D. PathPrahari (full)": seg["post_rate"],
    }
    truth = edges["true_rate"]
    sparse = seg.index[seg["exposure_mvkm"] < seg["exposure_mvkm"].quantile(0.5)]

    # held-out expected incidents (context-aware) for deviance
    tex = te_ex.copy()
    tex["ctx"] = full.context_multiplier(tex["seg_id"].to_numpy(), tex["hour"].to_numpy(),
                                         "clear")
    for w in ("fog", "rain"):
        msk = tex["weather"] == w
        if msk.any():
            tex.loc[msk, "ctx"] = full.context_multiplier(tex.loc[msk, "seg_id"].to_numpy(),
                                                          tex.loc[msk, "hour"].to_numpy(), w)
    ctx_expo = (tex["mvkm"] * tex["ctx"]).groupby(tex["seg_id"]).sum().reindex(edges.index).fillna(0)
    y_test = te_ev[te_ev["event_type"] == "incident"].groupby("seg_id").size() \
        .reindex(edges.index).fillna(0).to_numpy()
    total_test = y_test.sum()

    rows = []
    for name, s in scores.items():
        if name.startswith("A"):
            mu = s / s.sum() * total_test  # counts just replay history
        else:
            raw = (s * ctx_expo).to_numpy()
            mu = raw / max(raw.sum(), 1e-12) * total_test if name.startswith("B") else raw
        rows.append({
            "model": name,
            "hotspot_recall_top10pct": _recall_top(s, truth),
            "spearman_all_segments": stats.spearmanr(s, truth)[0],
            "spearman_sparse_segments": stats.spearmanr(s.loc[sparse], truth.loc[sparse])[0],
            "heldout_poisson_deviance": _deviance(y_test, np.asarray(mu)),
        })
    res = pd.DataFrame(rows)
    base = res.loc[0, "heldout_poisson_deviance"]
    res["deviance_vs_counts_pct"] = 100 * (res["heldout_poisson_deviance"] / base - 1)

    # 90% CI coverage of true rate (day/clear base)
    cov = float(((truth >= seg["ci_lo"]) & (truth <= seg["ci_hi"])).mean())

    # route-level rank correlation on random OD pairs (fastest path, 14:00 and 02:00)
    rng = np.random.default_rng(rng_seed)
    nodes = list(G.nodes)
    pr, tr = [], []
    from .routing import score_path, SCALE
    for _ in range(250):
        a, b = rng.choice(len(nodes), 2, replace=False)
        p = nx.shortest_path(G, nodes[a], nodes[b], weight="travel_min")
        if len(p) < 4:
            continue
        h = int(rng.integers(0, 24))
        r = score_path(G, full, p, h, n_mc=50)
        segs = r.seg_ids
        L = edges.loc[segs, "length_km"].to_numpy()
        ctx = full.context_multiplier(segs, r.seg_hours, "clear")
        pr.append(r.risk)
        tr.append(float((edges.loc[segs, "true_rate"].to_numpy() * L * ctx * SCALE).sum()))
    route_rho = stats.spearmanr(pr, tr)[0]
    return res, {"ci90_coverage": cov, "route_spearman": route_rho, "n_routes": len(pr),
                 "n_sparse": len(sparse)}, full


def scalability(event_multipliers=(1, 4, 10, 25), grid_sizes=((34, 11), (70, 20), (140, 40))):
    G = build_corridor()
    edges, ev, ex = simulate(G)
    dens = edges["mvkm_per_month"] / edges["length_km"]
    rows = []
    rng = np.random.default_rng(0)
    for k in event_multipliers:
        big = pd.concat([ev] * k, ignore_index=True)
        big["lon"] += rng.normal(0, 0.00012, len(big))
        big["lat"] += rng.normal(0, 0.00012, len(big))
        t0 = time.perf_counter(); detect_hotspots(big, 12, dens); t1 = time.perf_counter()
        RiskModel().fit(edges, big, ex); t2 = time.perf_counter()
        rows.append({"kind": "events", "n": len(big), "hotspot_s": t1 - t0, "fit_s": t2 - t1})
    for c, r in grid_sizes:
        Gg = build_corridor(c, r)
        e2, ev2, ex2 = simulate(Gg, months=3)
        m = RiskModel().fit(e2, ev2, ex2)
        src, dst = origin_destination(Gg)
        t0 = time.perf_counter()
        for h in (2, 9, 14, 20, 23):
            candidate_routes(Gg, m, src, dst, h)
        rows.append({"kind": "network", "n": Gg.number_of_edges(),
                     "route_query_s": (time.perf_counter() - t0) / 5})
    return pd.DataFrame(rows)
