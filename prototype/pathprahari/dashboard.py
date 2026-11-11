"""
Builds the dispatcher dashboard: one self-contained HTML file (no internet needed) that
holds the model's pre-computed answers for every origin/destination, departure hour and
weather combination, plus hotspots, analytics, the evaluation and the trip log.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .routing import candidate_routes, recommend

HERE = Path(__file__).parent
WEATHERS = ("clear", "fog", "rain")

HINDI = {
    "Sharp curves": "तीखे मोड़",
    "Steep gradient": "खड़ी चढ़ाई/ढलान",
    "Busy multi-arm junction": "व्यस्त चौराहा",
    "Poorly lit at night": "रात में कम रोशनी",
    "High-speed national highway": "तेज़ रफ़्तार हाईवे",
    "Narrow district road": "संकरी सड़क",
    "Fleet harsh-event cluster": "हमारे वाहनों ने यहाँ बार-बार अचानक ब्रेक लगाए",
    "Night / fatigue window": "रात/थकान का समय",
    "Fog + night + darkness": "कोहरा + रात + अंधेरा",
    "Rain on curves": "बारिश में मोड़",
}


def _nearest_node(G, lon, lat):
    return min(G.nodes, key=lambda n: (G.nodes[n]["lon"] - lon) ** 2 + (G.nodes[n]["lat"] - lat) ** 2)


def _route_json(r, node_idx, edges, model, weather):
    L = edges.loc[r.seg_ids, "length_km"].to_numpy()
    cum = np.r_[0, np.cumsum(L)]
    top = []
    for i in np.argsort(-r.seg_risk)[:3]:
        top.append({"s": int(r.seg_ids[i]), "km": [round(float(cum[i]), 1), round(float(cum[i + 1]), 1)],
                    "clk": f"{int(r.seg_hours[i]):02d}:00",
                    "why": model.reasons(r.seg_ids[i], hour=int(r.seg_hours[i]), weather=weather)})
    return {"n": [node_idx[n] for n in r.nodes], "s": [int(s) for s in r.seg_ids],
            "t": round(r.travel_min, 1), "km": round(r.length_km, 1), "r": round(r.risk, 4),
            "lo": round(r.risk_lo, 4), "hi": round(r.risk_hi, 4), "c": r.confidence, "top": top}


def build(G, model, edges, hotspots, log, res, extra, sc, reports, out_path, depot_names=None,
          progress=print):
    nodes = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    seg_ids = list(model.seg.index)
    seg = model.seg

    segs = []
    for sid in seg_ids:
        e = edges.loc[sid]
        segs.append({
            "id": int(sid), "a": node_idx[e["u"]], "b": node_idx[e["v"]], "cls": e["road_class"],
            "len": round(float(e["length_km"]), 2), "rate": round(float(seg.at[sid, "post_rate"]), 4),
            "lo": round(float(seg.at[sid, "ci_lo"]), 4), "hi": round(float(seg.at[sid, "ci_hi"]), 4),
            "conf": seg.at[sid, "confidence"], "lux": round(float(e["lux_index"]), 2),
            "curv": round(float(e["curvature"])), "grad": round(float(e["gradient_pct"]), 1),
            "ev": int(seg.at[sid, "surrogate_events"]), "inc": int(seg.at[sid, "incidents"]),
            "why": model.reasons(sid, hour=None),
        })

    ctx = {}
    hours = np.tile(np.arange(24), len(seg_ids))
    ids_rep = np.repeat(np.array(seg_ids), 24)
    for w in WEATHERS:
        m = model.context_multiplier(ids_rep, hours, w).reshape(len(seg_ids), 24)
        ctx[w] = np.round(m, 2).tolist()

    src = max((n for n in nodes if n[1] == max(j for _, j in nodes) // 2))
    dst = min((n for n in nodes if n[1] == max(j for _, j in nodes) // 2))
    town = _nearest_node(G, 82.15, 25.50)
    ods = [("Depot A → Hub B", src, dst), ("Hub B → Depot A", dst, src), ("Depot A → Town C", src, town)]

    combos = {}
    total = len(ods) * len(WEATHERS) * 24
    k = 0
    for oi, (name, a, b) in enumerate(ods):
        for w in WEATHERS:
            for h in range(24):
                scored, front = candidate_routes(G, model, a, b, h, w)
                rec = recommend(front)
                combos[f"{oi}|{w}|{h}"] = {
                    "opts": {key: _route_json(r, node_idx, edges, model, w) for key, r in rec.items()},
                    "nCand": len(scored)}
                k += 1
                if k % 54 == 0:
                    progress(f"     dashboard: {k}/{total} route scenarios scored")

    hs = hotspots.copy()
    hs_rows = hs[["hotspot_id", "lon", "lat", "risk_score", "events", "incidents", "near_misses",
                  "top_event", "peak_hour", "road_class", "extent_m", "reasons"]].round(5)

    data = {
        "nodes": [[round(G.nodes[n]["lon"], 5), round(G.nodes[n]["lat"], 5)] for n in nodes],
        "segs": segs, "ctx": ctx,
        "ods": [{"name": n, "a": node_idx[a], "b": node_idx[b]} for n, a, b in ods],
        "combos": combos,
        "hotspots": hs_rows.to_dict("records"),
        "log": log.to_dict("records"),
        "eval": {"res": res.round(4).to_dict("records"), "extra": {k: float(v) for k, v in extra.items()},
                 "scal": None if sc is None else sc.replace({np.nan: None}).round(4).to_dict("records"),
                 "ctx": {k: round(float(v), 3) for k, v in model.context_effects().items()},
                 "conf": seg["confidence"].value_counts().to_dict()},
        "reports": reports, "hindi": HINDI,
    }
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item()
                         if hasattr(o, "item") else str(o))
    html = html.replace("/*__DATA__*/null", payload)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
