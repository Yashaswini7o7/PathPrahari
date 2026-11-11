"""
PathPrahari prototype.

    python run_demo.py                 # full run, then opens the dispatch dashboard
    python run_demo.py --quick         # skips the scalability benchmark
    python run_demo.py --no-browser    # does not open the browser at the end

Results are saved in ./outputs. The dashboard is outputs/PathPrahari_Dashboard.html.
Fleet telematics are simulated (see pathprahari/simulate.py); road and lighting features
follow the open-data schema listed in DATA_SOURCES.csv.
"""
from __future__ import annotations

import json
import sys
import time
import webbrowser
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pathprahari import dashboard
from pathprahari import figures as F
from pathprahari.evaluate import scalability, temporal_holdout
from pathprahari.model import RiskModel, detect_hotspots
from pathprahari.network import build_corridor, origin_destination
from pathprahari.routing import (candidate_routes, departure_sweep, halt_suggestion, recommend,
                                 score_path)
from pathprahari.simulate import simulate

OUT = Path(__file__).parent / "outputs"
FIG = OUT / "figures"
REP = OUT / "reports"
for p in (OUT, FIG, REP):
    p.mkdir(parents=True, exist_ok=True)

HI = {  # driver briefing phrases (Hindi)
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


def to_hindi(reason: str) -> str:
    for k, v in HI.items():
        if reason.startswith(k):
            return v
    return reason


def km_marker(route, edges, idx):
    L = edges.loc[route.seg_ids, "length_km"].to_numpy()
    a = L[:idx].sum()
    return a, a + L[idx]


def main(quick=False, open_browser=True):
    t_start = time.time()
    print("\nPathPrahari: pre-dispatch route risk (Team Hackzz)\n")
    print("1/9  Building road network + public-data feature schema ...")
    G = build_corridor()
    src, dst = origin_destination(G)

    print("2/9  Simulating 12 months of fleet telematics (synthetic) ...")
    edges, events, exposure = simulate(G)
    events.sample(frac=1, random_state=0).head(60000).round(6) \
        .to_csv(OUT / "synthetic_telematics_events_sample.csv.gz", index=False)

    print("3/9  Detecting severity- and recency-weighted hotspots ...")
    dens = edges["mvkm_per_month"] / edges["length_km"]
    hotspots, ev_lab = detect_hotspots(events, now_month=12, seg_exposure_density=dens)

    print("4/9  Fitting uncertainty-aware risk model ...")
    model = RiskModel().fit(edges, events, exposure)
    seg = model.seg.join(edges[["road_class", "length_km", "curvature", "gradient_pct",
                                "junction_deg", "lux_index", "mid_lon", "mid_lat"]])
    seg.drop(columns=["_a", "_b"]).round(5).to_csv(OUT / "segment_risk_scores.csv")
    model.context_effects().round(3).rename("multiplier").to_csv(OUT / "learned_context_multipliers.csv")

    hotspots["reasons"] = [" | ".join(model.reasons(s, hour=int(h)))
                           for s, h in zip(hotspots["main_segment"], hotspots["peak_hour"])]
    hotspots["road_class"] = edges.loc[hotspots["main_segment"], "road_class"].to_numpy()
    hotspots.drop(columns=["cluster"]).round(5).to_csv(OUT / "hotspot_registry.csv", index=False)

    print("5/9  Scoring routes, alternatives, departure window ...")
    report = {}
    for label, hour, weather in (("afternoon_clear", 14, "clear"), ("night_clear", 1, "clear"),
                                 ("night_fog", 1, "fog")):
        scored, front = candidate_routes(G, model, src, dst, hour, weather)
        rec = recommend(front)
        halt = halt_suggestion(G, model, rec["Safest"])
        opts = {}
        for k, r in rec.items():
            top = np.argsort(-r.seg_risk)[:3]
            opts[k] = {
                "travel_min": round(r.travel_min, 1), "length_km": round(r.length_km, 1),
                "risk_per_10k_trips": round(r.risk, 3),
                "risk_90pct_interval": [round(r.risk_lo, 3), round(r.risk_hi, 3)],
                "confidence": r.confidence,
                "risk_change_vs_fastest_pct": round(100 * (r.risk / rec["Fastest"].risk - 1), 1),
                "extra_minutes_vs_fastest": round(r.travel_min - rec["Fastest"].travel_min, 1),
                "top_risk_segments": [{
                    "seg_id": int(r.seg_ids[i]),
                    "km": [round(x, 1) for x in km_marker(r, edges, i)],
                    "clock_time": f"{int(r.seg_hours[i]):02d}:00",
                    "reasons": model.reasons(r.seg_ids[i], hour=int(r.seg_hours[i]), weather=weather),
                } for i in top]}
        report[label] = {"departure": f"{hour:02d}:00", "weather": weather, "options": opts,
                         "candidates_evaluated": len(scored),
                         "halt_suggestion": None if halt is None else {
                             k: (v if not isinstance(v, tuple) else list(v)) for k, v in halt.items()}}
        if label == "night_clear":
            night = (scored, rec, halt)
    (OUT / "route_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    scored, rec, halt = night
    fastest_path, safe_path = rec["Fastest"].nodes, rec["Safest"].nodes
    sw_f = departure_sweep(G, model, fastest_path)
    sw_s = departure_sweep(G, model, safe_path)
    pd.DataFrame({"depart_hour": sw_f["depart_hour"], "fastest_route_risk": sw_f["risk"],
                  "bypass_route_risk": sw_s["risk"]}).round(4).to_csv(OUT / "departure_window.csv", index=False)

    # Driver briefing (Hindi + English)
    r = rec["Safest"]
    top = np.argsort(-r.seg_risk)[:3]
    en = [f"PathPrahari pre-trip briefing - Depot A -> Hub B",
          f"Departure 01:00 | Route: {r.name} (bypass) | {r.travel_min:.0f} min | "
          f"risk {r.risk:.1f}/10k trips ({100*(r.risk/rec['Fastest'].risk-1):+.0f}% vs fastest)", "Watch out:"]
    hi = [f"PathPrahari यात्रा-पूर्व सूचना — डिपो A → हब B",
          f"प्रस्थान 01:00 | मार्ग: बाईपास (सुरक्षित विकल्प) | लगभग {r.travel_min:.0f} मिनट", "सावधान रहें:"]
    for k, i in enumerate(top, 1):
        a, b = km_marker(r, edges, i)
        rs = model.reasons(r.seg_ids[i], hour=int(r.seg_hours[i]))
        en.append(f" {k}. km {a:.0f}-{b:.0f} (~{int(r.seg_hours[i]):02d}:00): " + "; ".join(rs))
        hi.append(f" {k}. किमी {a:.0f}–{b:.0f} (~{int(r.seg_hours[i]):02d}:00 बजे): "
                  + ", ".join(dict.fromkeys(to_hindi(x) for x in rs)) + " — रफ़्तार कम रखें")
    if halt and halt.get("type") == "halt":
        en.append("Tip: halt at the lit town before the dark stretch and resume at 05:00.")
        hi.append("सुझाव: अंधेरे हिस्से से पहले रोशनी वाले कस्बे में रुकें, सुबह 5 बजे आगे बढ़ें।")
    elif halt:
        best = int(sw_s.loc[sw_s["risk"].idxmin(), "depart_hour"])
        en.append(f"Tip: this trip crosses unlit roads between 01:00-05:00. Leaving at {best:02d}:00 "
                  f"instead cuts risk by {100*(1-sw_s['risk'].min()/sw_s.loc[1,'risk']):.0f}%.")
        hi.append(f"सुझाव: यह यात्रा रात 1–5 बजे अंधेरी सड़कों से गुज़रती है। {best:02d}:00 बजे निकलने पर "
                  f"जोखिम {100*(1-sw_s['risk'].min()/sw_s.loc[1,'risk']):.0f}% कम होगा।")
    (OUT / "driver_briefing_en.txt").write_text("\n".join(en), encoding="utf-8")
    (OUT / "driver_briefing_hi.txt").write_text("\n".join(hi), encoding="utf-8")

    print("6/9  Dispatch log + post-trip comparison + analytics reports ...")
    rng = np.random.default_rng(5)
    log = []
    for t in range(120):
        h = int(rng.choice(24))
        wx = rng.choice(["clear", "fog", "rain"], p=[.7, .2, .1])
        sc, fr = candidate_routes(G, model, src, dst, h, wx)
        rc = recommend(fr)
        choice = rng.choice(["Fastest", "Balanced", "Safest"], p=[.25, .45, .30])
        rr = rc[choice]
        # post-trip: observed harsh events drawn from ground truth on the driven segments
        true_lam = (edges.loc[rr.seg_ids, "true_rate"].to_numpy() * edges.loc[rr.seg_ids, "length_km"].to_numpy()
                    * model.context_multiplier(rr.seg_ids, rr.seg_hours, wx) / 1e6 * 55 * 2.3e3)
        log.append({"trip_id": f"T{t+1:04d}", "vehicle_id": int(rng.integers(1, 181)),
                    "driver_cohort": rng.choice(["<2y", "2-7y", ">7y"]), "departure_hour": h,
                    "weather": wx, "options_shown": len(fr), "chosen_option": choice,
                    "travel_min": round(rr.travel_min, 1), "risk_per_10k": round(rr.risk, 3),
                    "risk_lo": round(rr.risk_lo, 3), "risk_hi": round(rr.risk_hi, 3),
                    "confidence": rr.confidence,
                    "fastest_risk_per_10k": round(rc["Fastest"].risk, 3),
                    "post_trip_harsh_events": int(rng.poisson(true_lam.sum()))})
    log = pd.DataFrame(log)
    log.to_csv(OUT / "dispatch_log.csv", index=False)
    rho = log[["risk_per_10k", "post_trip_harsh_events"]].corr("spearman").iloc[0, 1]

    ev = events.copy()
    ev["zone"] = pd.cut(ev["lon"], 6, labels=[f"Zone {i}" for i in range(6, 0, -1)]).astype(str)
    ev["road_class"] = edges.loc[ev["seg_id"], "road_class"].to_numpy()
    expo_m = exposure.groupby("month")["mvkm"].sum()
    trend = ev.groupby(["month", "event_type"]).size().unstack(fill_value=0)
    trend["incidents_per_mvkm"] = trend["incident"] / expo_m
    trend["harsh_events_per_mvkm"] = trend.drop(columns="incident").sum(axis=1) / expo_m
    trend.round(4).to_csv(REP / "trend_by_month.csv")
    ev.groupby(["zone", "event_type"]).size().unstack(fill_value=0).to_csv(REP / "events_by_zone.csv")
    ev.groupby(["driver_cohort", "event_type"]).size().unstack(fill_value=0).to_csv(REP / "events_by_driver_cohort.csv")
    seg.groupby("road_class").agg(segments=("post_rate", "size"), mean_risk=("post_rate", "mean"),
                                  low_conf_share=("confidence", lambda s: (s == "Low").mean())) \
        .round(4).to_csv(REP / "risk_by_road_class.csv")
    ev.groupby(["hour"]).size().rename("events").to_csv(REP / "events_by_hour.csv")
    log.groupby("chosen_option").agg(trips=("trip_id", "size"), mean_risk=("risk_per_10k", "mean"),
                                     mean_min=("travel_min", "mean"),
                                     harsh_events=("post_trip_harsh_events", "mean")) \
        .round(3).to_csv(REP / "dispatch_outcomes_by_option.csv")

    print("7/9  Evaluation (temporal hold-out) ...")
    res, extra, _ = temporal_holdout(G, edges, events, exposure)
    res.round(4).to_csv(OUT / "evaluation_results.csv", index=False)
    extra["dispatch_log_spearman_risk_vs_harsh_events"] = float(rho)
    sc = None
    if not quick:
        print("     scalability benchmark ...")
        sc = scalability()
        sc.round(4).to_csv(OUT / "scalability_benchmark.csv", index=False)
    (OUT / "evaluation_summary.json").write_text(json.dumps(
        {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in extra.items()}, indent=2))

    print("8/9  Figures ...")
    F._save(F.risk_map(G, model, rec, hotspots, halt, hour=1), FIG / "risk_heatmap_routes.png")
    F._save(F.pareto(scored, rec), FIG / "risk_vs_time_pareto.png")
    F._save(F.departure_plot(sw_f, sw_s), FIG / "departure_window.png")
    F._save(F.uncertainty_plot(model.seg), FIG / "uncertainty_vs_data.png")
    F._save(F.evaluation_plot(res), FIG / "evaluation_vs_baselines.png")
    F._save(F.context_plot(model.context_effects()), FIG / "learned_context_multipliers.png")
    rf = rec["Fastest"]; topf = list(np.argsort(-rf.seg_risk)[:3])
    F._save(F.segment_strip(rf, edges, [(i, "") for i in topf]), FIG / "route_segment_risk_strip.png")
    if sc is not None:
        F._save(F.scalability_plot(sc), FIG / "scalability.png")

    print("9/9  Building the dispatch dashboard ...")
    zone = (ev[ev["event_type"] == "incident"].groupby("zone").size()
            .reindex([f"Zone {i}" for i in range(1, 7)]).fillna(0))
    harsh_c = ev[ev["event_type"] != "incident"].groupby("driver_cohort").size().reindex(["<2y", "2-7y", ">7y"]).fillna(0)
    road = seg.groupby("road_class").agg(segments=("post_rate", "size"), mean_risk=("post_rate", "mean"),
                                         low_conf_share=("confidence", lambda x: (x == "Low").mean())).reset_index()
    outcomes = log.groupby("chosen_option").agg(trips=("trip_id", "size"), mean_risk=("risk_per_10k", "mean"),
                                                mean_min=("travel_min", "mean"),
                                                harsh_events=("post_trip_harsh_events", "mean")).reset_index()
    reports = {
        "trend": {"month": [["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][(m + 9) % 12]
                            for m in trend.index], "harsh": trend["harsh_events_per_mvkm"].round(1).tolist(),
                  "inc": trend["incidents_per_mvkm"].round(3).tolist()},
        "hour": {"hour": list(range(24)), "events": ev.groupby("hour").size().reindex(range(24)).fillna(0).astype(int).tolist()},
        "zone": {"zone": [z.replace("Zone ", "Z") for z in zone.index], "inc": zone.astype(int).tolist()},
        "cohort": {"cohort": ["< 2 yrs", "2–7 yrs", "> 7 yrs"], "harsh": harsh_c.astype(int).tolist()},
        "road": road.round(4).to_dict("records"), "outcomes": outcomes.round(3).to_dict("records"),
    }
    dash = dashboard.build(G, model, edges, hotspots, log, res, extra, sc, reports,
                           OUT / "PathPrahari_Dashboard.html")

    f, b, s = rec["Fastest"], rec["Balanced"], rec["Safest"]
    print("\n=== Depot A -> Hub B, 01:00 departure, clear weather ===")
    for name, r in (("Fastest", f), ("Balanced", b), ("Safest", s)):
        print(f"  {name:9s} {r.travel_min:6.1f} min  risk {r.risk:6.2f} "
              f"[{r.risk_lo:.2f}, {r.risk_hi:.2f}]  conf {r.confidence}")
    print("\n=== Model check on held-out months 10-12 ===")
    print(res.round(3).to_string(index=False))
    print(f"90% interval coverage: {extra['ci90_coverage']:.0%}   "
          f"route-level rank correlation: {extra['route_spearman']:.2f}")
    print(f"\nDone in {time.time() - t_start:.0f} s. Results saved in: {OUT}")
    print(f"Dashboard: {dash}")
    if open_browser:
        try:
            webbrowser.open(Path(dash).resolve().as_uri())
            print("Opening the dashboard in your browser ...")
        except Exception:
            print("Could not open a browser automatically; open the dashboard file above.")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, open_browser="--no-browser" not in sys.argv)
