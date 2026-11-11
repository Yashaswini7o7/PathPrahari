# PathPrahari

Pre-dispatch route risk for Indian truck fleets.
Team Hackzz: Yashaswini Singh, Shaleth Ghosh
Theme: Fleet Route Risk Intelligence (Safer Roads for India)

Before a truck leaves the depot, PathPrahari tells the dispatcher how risky the planned trip is, which road stretches make it risky, and whether a different route or a different departure time would make it safer without costing too much time.

## Running it

**Windows:** double-click `start_windows.bat`
**Mac / Linux:** run `./start_mac_linux.sh` in a terminal

Both install the Python packages listed in `prototype/requirements.txt` and then run `prototype/run_demo.py`. You need Python 3.9 or newer. A full run takes about 40 seconds, and the dispatch dashboard opens in your browser when it finishes.

To run it by hand:

```
cd prototype
pip install -r requirements.txt
python run_demo.py
```

Options: `--quick` skips the scalability benchmark, `--no-browser` stops it from opening the dashboard.

If you only want to look around without installing anything, open `PathPrahari_Dashboard.html` in this folder. It is the dashboard from our own run and works offline.

## The dashboard

| Tab | What you can do |
|---|---|
| Dispatch | Pick a trip, departure hour and weather. See every road coloured by risk at that hour, the route options with time, risk and a 90% range, the stretches that make the route risky and why, and the best departure hour. Assign a route and the Hindi/English driver briefing updates. |
| Hotspots | Map and searchable table of 71 fleet hotspots, ranked by risk per vehicle passing. |
| Analytics | Monthly trends, events by hour, by zone and by driver experience, risk by road type, outcomes by the option dispatchers chose. |
| Model check | How the model did on months it was not trained on, compared with simpler methods, plus what it learned about night, fog and rain, and run times. |
| Trip log | Every dispatch with its score, kept for comparing with what happened on the trip. Downloadable as CSV. |
| About the data | Where each input comes from and how to read the numbers. |

## What a run saves (`prototype/outputs/`)

- `PathPrahari_Dashboard.html`: the dashboard
- `route_report.json`: route options for the main trip at 14:00, 01:00 and 01:00 in fog
- `segment_risk_scores.csv`: risk, 90% range and confidence for every road segment
- `hotspot_registry.csv`: hotspots with reasons
- `dispatch_log.csv` and `reports/`: trip log and analytics tables
- `evaluation_results.csv`, `evaluation_summary.json`, `scalability_benchmark.csv`: the model check
- `driver_briefing_hi.txt`, `driver_briefing_en.txt`
- `figures/`: the charts as PNG files

`screenshots/` in this folder has images of the dashboard taken from our run.

## How it works

1. **Road features from open data.** Curvature and junctions (OpenStreetMap), street lighting (NASA VIIRS night lights), gradients (SRTM), blackspots (MoRTH), weather (Open-Meteo / IMD). Sources and licences are in `DATA_SOURCES.csv`.
2. **Fleet events.** Harsh braking, harsh turns, near-misses and incidents are matched to road segments. Each segment's vehicle-km are counted by hour and weather, so busy roads are not mistaken for dangerous ones.
3. **Hotspots.** Events are weighted by severity and how recent they are (3-month half-life), divided by traffic, then clustered.
4. **Risk per road segment.** A Poisson model on the open-data features gives a starting estimate for every road, even with no fleet history. The fleet's own incidents and harsh events then update it (empirical Bayes, Poisson–Gamma). The result is a risk estimate with a 90% range, labelled High, Medium or Low confidence.
5. **Time and weather.** The model learns how much night, the 2–5 AM fatigue window, unlit roads, fog and rain raise risk. A route is scored for the hour the truck actually reaches each stretch.
6. **Options.** Shortest-path searches over time + λ × risk for a range of λ give the trade-off curve. Dispatchers see Fastest, Balanced and Safest, but never an option more than 30% slower than the fastest. The best departure hour is shown as well.

Code: `prototype/pathprahari/`. `network.py` (roads), `simulate.py` (fleet data), `model.py` (hotspots and risk), `routing.py` (scoring and options), `evaluate.py` (model check), `dashboard.py` (the dashboard), `figures.py` (charts).

## Results from our run

- Depot A to Hub B leaving at 01:00: the state-highway bypass has 32% lower expected risk than the fastest route and takes 17 minutes longer.
- Leaving the same trip at 06:00 instead of 01:00 lowers risk by 70%.
- On months 10–12 (trained on 1–9), the model found 81% of the truly riskiest 10% of roads. Ranking by crash counts found 31%.
- The 90% ranges contained the true risk for 97% of road segments.
- 5 million events take about 6 seconds for hotspots plus model fit on one laptop core. Scoring all options for a trip takes under half a second, even on a network of 7,849 road segments.

## About the fleet data

We could not find any public Indian dataset of truck harsh-braking events with GPS positions. So the prototype simulates 12 months for a 180-truck fleet (2,03,661 events) on a generated road network that uses the same fields as the OpenStreetMap, VIIRS and SRTM loaders. The simulation includes patterns reported in MoRTH data and common on Indian highways: night and fatigue risk, unlit stretches, winter fog, monsoon rain, busy junctions, and hotspots such as median cuts and truck parking near dhabas that road features alone do not reveal.

Because the true risk of each road is known in the simulation, we can measure accuracy exactly. With a real fleet, the simulator is replaced by the fleet's GPS export and nothing else changes.

## Licence

MIT. See `LICENSE`.
