"""Figure generation (dark theme, matches the pitch deck)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

BG = "#0F1A2E"
INK = "#E8EDF6"
INK2 = "#9AA8C1"
GRID = "#24324D"
RED, AMBER, GREEN, BLUE = "#E5484D", "#C98A10", "#159C6B", "#3E7FE0"
HEAT = LinearSegmentedColormap.from_list("heat", ["#22304C", "#5B3A3F", "#B5532F", "#F08A24",
                                                  "#FFD166"])

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 12,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def risk_map(G, model, rec, hotspots, path, hour=1):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    segs, vals = [], []
    ids = [d["seg_id"] for _, _, d in G.edges(data=True)]
    ctx = model.context_multiplier(ids, [hour] * len(ids), "clear")
    dens = model.seg.loc[ids, "post_rate"].to_numpy() * ctx
    for (u, v, d) in G.edges(data=True):
        segs.append([(G.nodes[u]["lon"], G.nodes[u]["lat"]), (G.nodes[v]["lon"], G.nodes[v]["lat"])])
    v = np.log1p(dens / np.median(dens))
    lc = LineCollection(segs, cmap=HEAT, linewidths=np.where(v > np.quantile(v, .8), 3.2, 1.6),
                        capstyle="round")
    lc.set_array(v); ax.add_collection(lc)

    def draw(r, color, ls, lw, label):
        xs = [G.nodes[n]["lon"] for n in r.nodes]; ys = [G.nodes[n]["lat"] for n in r.nodes]
        ax.plot(xs, ys, color=BG, lw=lw + 4, solid_capstyle="round", zorder=4)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=5, label=label)

    f, s = rec["Fastest"], rec["Safest"]
    draw(f, RED, (0, (4, 2)), 2.6, f"Fastest  {f.travel_min:.0f} min · risk {f.risk:.1f}")
    draw(s, GREEN, "-", 3.2, f"PathPrahari pick  {s.travel_min:.0f} min · risk {s.risk:.1f}")
    top = hotspots.head(12)
    ax.scatter(top["lon"], top["lat"], s=40 + 3.0 * top["risk_score"], facecolors="none",
               edgecolors=INK, linewidths=1.4, zorder=6)
    for _, h in top.head(4).iterrows():
        ax.annotate(h["hotspot_id"], (h["lon"], h["lat"]), xytext=(6, 8), textcoords="offset points",
                    fontsize=9.5, color=INK, zorder=7, fontweight="bold")
    o, dnode = f.nodes[0], f.nodes[-1]
    for n, lab, ha in ((o, "Depot A", "left"), (dnode, "Hub B", "right")):
        ax.scatter(G.nodes[n]["lon"], G.nodes[n]["lat"], s=110, color=INK, zorder=8,
                   edgecolors=BG, linewidths=2)
        ax.annotate(lab, (G.nodes[n]["lon"], G.nodes[n]["lat"]), xytext=(0, -20),
                    textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    if path is not None and path.get("type") == "halt":
        ax.scatter(path["lon"], path["lat"], marker="s", s=110, color=AMBER, zorder=9,
                   edgecolors=BG, linewidths=2)
        ax.annotate("Halt till 05:00 (lit town)", (path["lon"], path["lat"]), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    # label the hazardous stretch the bypass avoids
    avoided = [n for n in f.nodes if n not in set(s.nodes)]
    if avoided:
        xs = [G.nodes[n]["lon"] for n in avoided]; ys = [G.nodes[n]["lat"] for n in avoided]
        ax.annotate("Hilly, unlit NH stretch\n+ hidden fleet hotspots", (np.mean(xs), max(ys)),
                    xytext=(0, 34), textcoords="offset points", ha="center", fontsize=10.5,
                    color="#FFD166", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#FFD166", lw=1.2))
    ax.set_xlim(min(x for x, _ in sum(segs, [])) - .02, max(x for x, _ in sum(segs, [])) + .02)
    ax.set_ylim(min(y for _, y in sum(segs, [])) - .02, max(y for _, y in sum(segs, [])) + .02)
    ax.set_aspect(1 / np.cos(np.deg2rad(25.4)))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, frameon=False,
                    fontsize=11, labelcolor=INK)
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([]); cb.set_label("Segment risk at 01:00  (low → high)", color=INK2)
    cb.outline.set_edgecolor(GRID)
    ax.set_title(f"Route risk heatmap · departure {hour:02d}:00 · circles = fleet hotspots",
                 color=INK, pad=10)
    return fig


def pareto(scored, rec):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for r in scored:
        ax.scatter(r.travel_min, r.risk, s=50, color=INK2, alpha=.5, zorder=2)
    cols = {"Fastest": RED, "Balanced": AMBER, "Safest": GREEN}
    seen = set()
    for k, r in sorted(rec.items(), key=lambda kv: kv[0] != "Safest"):
        if id(r) in seen:
            continue
        seen.add(id(r))
        ax.errorbar(r.travel_min, r.risk, yerr=[[r.risk - r.risk_lo], [r.risk_hi - r.risk]],
                    fmt="o", ms=11, color=cols[k], ecolor=cols[k], elinewidth=2, capsize=5, zorder=4,
                    mec=BG, mew=2)
        names = [n for n, rr in rec.items() if rr is r]
        off = (14, -4) if k == "Fastest" else (14, 10)
        ax.annotate(" / ".join(names) + f"\n{r.travel_min:.0f} min · risk {r.risk:.1f}",
                    (r.travel_min, r.risk), xytext=off, textcoords="offset points",
                    fontsize=10.5, color=INK, fontweight="bold", va="center")
    f = rec["Fastest"]; s = rec["Safest"]
    far = [r for r in scored if r.travel_min > f.travel_min * 1.30]
    if far:
        fx = np.mean([r.travel_min for r in far]); fy = np.mean([r.risk for r in far])
        ax.annotate(f"Slightly safer, but +{min(r.travel_min for r in far) - f.travel_min:.0f} min\n"
                    "→ not a reasonable trade, not offered", (fx, fy), xytext=(0, -42),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
    ax.annotate("", xy=(s.travel_min, s.risk), xytext=(f.travel_min, f.risk),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5, ls="--"))
    ax.set_xlabel("Travel time (minutes)"); ax.set_ylabel("Expected incidents per 10,000 trips")
    ax.set_title("Risk vs time: every option, with 90% intervals")
    ax.set_xlim(f.travel_min - 8, max(r.travel_min for r in scored) + 22)
    ax.set_ylim(0, None)
    return fig


def departure_plot(sweep_fast, sweep_safe):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.axvspan(-0.5, 5.5, color="#1A2742", zorder=0); ax.axvspan(21.5, 23.5, color="#1A2742", zorder=0)
    ax.plot(sweep_fast["depart_hour"], sweep_fast["risk"], color=RED, lw=2.2, ls=(0, (4, 2)),
            marker="o", ms=5)
    ax.plot(sweep_safe["depart_hour"], sweep_safe["risk"], color=GREEN, lw=2.4, marker="o", ms=5)
    ax.annotate("NH, straight through", (13, sweep_fast["risk"][13]), xytext=(0, 10),
                textcoords="offset points", ha="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.annotate("Bypass route", (13, sweep_safe["risk"][13]), xytext=(0, -18),
                textcoords="offset points", ha="center", color=INK, fontsize=10.5, fontweight="bold")
    j = int(sweep_safe["risk"].idxmin())
    ax.annotate(f"Best window: {int(sweep_safe['depart_hour'][j]):02d}:00",
                (sweep_safe["depart_hour"][j], sweep_safe["risk"][j]), xytext=(18, -32),
                textcoords="offset points", ha="center", color=INK, fontsize=10.5)
    ax.text(2.5, 0.25, "night", ha="center", va="bottom", color=INK2, fontsize=10)
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("Departure time"); ax.set_ylabel("Expected incidents per 10,000 trips")
    ax.set_title("When you leave matters as much as which way")
    ax.set_ylim(0, None)
    return fig


def uncertainty_plot(seg):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    cols = {"High": GREEN, "Medium": AMBER, "Low": RED}
    rel = (seg["ci_hi"] - seg["ci_lo"]) / seg["post_rate"]
    for c in ("Low", "Medium", "High"):
        m = seg["confidence"] == c
        ax.scatter(seg.loc[m, "exposure_mvkm"].clip(lower=1e-3), rel[m], s=16, color=cols[c],
                   alpha=.8, label=f"{c} confidence", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Fleet exposure on segment (million vehicle-km, log scale)")
    ax.set_ylabel("Width of 90% interval ÷ estimate")
    ax.set_title("Knows what it doesn't know")
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    return fig


def evaluation_plot(res):
    fig, ax = plt.subplots(figsize=(7.4, 5))
    labels = ["Raw crash\ncounts", "Crash rate\n(no smoothing)", "Public-data\nprior only", "PathPrahari\nfull"]
    x = np.arange(4); w = 0.38
    a = res["hotspot_recall_top10pct"].to_numpy(); b = res["spearman_all_segments"].to_numpy()
    b1 = ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="Hotspot recall (top 10%)")
    b2 = ax.bar(x + w / 2 + 0.01, b, w, color=GREEN, label="Rank correlation with true risk")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}",
                    ha="center", fontsize=10, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", fontsize=10)
    ax.set_title("Held-out months 10–12: PathPrahari vs baselines")
    ax.grid(axis="x", visible=False)
    return fig


def scalability_plot(sc):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    e = sc[sc["kind"] == "events"]; n = sc[sc["kind"] == "network"]
    axs[0].plot(e["n"] / 1e6, e["hotspot_s"], color=BLUE, marker="o", lw=2, label="Hotspot detection")
    axs[0].plot(e["n"] / 1e6, e["fit_s"], color=GREEN, marker="o", lw=2, label="Risk model fit")
    axs[0].set_xlabel("Telematics events (millions)"); axs[0].set_ylabel("Seconds (single laptop core)")
    axs[0].legend(frameon=False, labelcolor=INK); axs[0].set_title("Scales ~linearly with events")
    axs[0].set_ylim(0, None)
    axs[1].plot(n["n"], n["route_query_s"] * 1000, color=AMBER, marker="o", lw=2)
    for _, r in n.iterrows():
        axs[1].annotate(f"{r['route_query_s']*1000:.0f} ms", (r["n"], r["route_query_s"] * 1000),
                        xytext=(0, 8), textcoords="offset points", ha="center", color=INK, fontsize=10)
    axs[1].set_xlabel("Road segments in network"); axs[1].set_ylabel("Milliseconds per route query")
    axs[1].set_title("Pre-dispatch query latency"); axs[1].set_ylim(0, None)
    fig.tight_layout()
    return fig


def context_plot(eff):
    names = {"night": "Night (22–05h)", "night_x_dark": "…on an unlit road", "fatigue_2to5": "Fatigue window (02–05h)",
             "fog": "Fog", "fog_x_night_x_dark": "Fog + night + unlit", "rain": "Rain",
             "rain_x_curve": "Rain on curves"}
    e = eff.rename(index=names).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.barh(e.index, e.values, color=AMBER, height=0.6)
    for i, v in enumerate(e.values):
        ax.text(v + 0.03, i, f"×{v:.2f}", va="center", color=INK, fontsize=10.5)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xlabel("Learned risk multiplier (1 = daytime, clear)")
    ax.set_title("What the model learned about Indian conditions")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, e.max() * 1.18)
    return fig


def segment_strip(route, edges, reasons_top):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    x = np.cumsum(np.r_[0, edges.loc[route.seg_ids, "length_km"].to_numpy()])
    widths = np.diff(x)
    r = route.seg_risk / widths  # risk per km
    norm = (r - r.min()) / max(np.ptp(r), 1e-9)
    ax.bar(x[:-1], r, width=widths * 0.96, align="edge", color=HEAT(0.15 + 0.85 * norm))
    for k, (i, txt) in enumerate(reasons_top):
        ax.annotate(f"{k+1}", (x[i] + widths[i] / 2, r[i]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("Distance along fastest route (km)"); ax.set_ylabel("Risk per km")
    ax.set_title("Where the danger sits on the route (01:00 departure)")
    ax.grid(axis="x", visible=False)
    return fig
