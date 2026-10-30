"""Step 7: Bergson's test — do the machine's motion units belong to the movement
or to the snapshots?

Bergson (Creative Evolution, 1907): ordinary knowledge "takes snapshots" of the
passing reality and strings them together, so "the mechanism of our ordinary
knowledge is of a cinematographical kind". The pipeline does exactly that. Here
the same footage is analysed with every sampled frame (about 15 per second of
film), every 2nd (7.5) and every 4th (3.75), and the resulting 'alphabets' are
compared. If the units were properties of the movement they would survive
coarser snapshots; if they shift, they belong to the cinematograph.

  python 07_bergson.py
Writes out/bergson_summary.csv and out/bergson_cuts.csv
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common import OUT

RATES = {1: "every sampled frame", 2: "every 2nd", 4: "every 4th"}
for n in RATES:
    tag = "dense" if n == 1 else f"dense_every{n}"
    if not (OUT / f"alphabet_{tag}.json").exists():
        subprocess.run([sys.executable, "03_analyse.py", "dense", "--every", str(n)], check=True)

fps = json.load(open(OUT / "video_meta.json"))["fps"]
rows, units_by = [], {}
for n, label in RATES.items():
    tag = "dense" if n == 1 else f"dense_every{n}"
    info = json.load(open(OUT / f"alphabet_{tag}.json"))
    u = pd.read_csv(OUT / f"units_{tag}.csv")
    tr = pd.read_csv(OUT / f"tracks_{tag}.csv")
    # seconds of film during which a wrist was tracked (smoothed and valid)
    tracked = 0.0
    for _, g in tr.dropna(subset=["speed"]).groupby(["shot", "track", "hand"]):
        t = g["t"].to_numpy()
        if len(t) > 1:
            gaps = np.diff(t)
            tracked += gaps[gaps < 1.0].sum()
    units_by[n] = u
    rows.append({
        "every": n, "label": label, "snapshots_per_film_second": round(fps / 2 / n, 2),
        "units": len(u), "tracked_wrist_seconds": round(tracked, 1),
        "units_per_tracked_second": len(u) / tracked if tracked else np.nan,
        "median_unit_s": u["dur"].median(), "still_share": u["still"].mean(),
        "letters": info["best_k"],
        "letter_names": "; ".join(f"{d['name']} ({d['share']*100:.0f}%)" for d in info["letters"]),
    })
summary = pd.DataFrame(rows)

# how many cuts found at full rate survive at the coarser rates?
def cuts(u):
    c = {}
    for (s, t, h), g in u.groupby(["shot", "track", "hand"]):
        c[(s, h)] = np.unique(np.r_[g["t0"].to_numpy(), g["t1"].to_numpy()])
    return c

fine = cuts(units_by[1])
surv = []
for n in (2, 4):
    coarse = cuts(units_by[n])
    tol = 2 * n / fps          # one coarse sampling interval (stride-2 grid x n)
    hit = tot = 0
    for k, tf in fine.items():
        tc = coarse.get(k)
        tot += len(tf)
        if tc is not None and len(tc):
            hit += int(np.sum(np.min(np.abs(tf[:, None] - tc[None, :]), axis=1) <= tol))
    surv.append({"every": n, "fine_cuts": tot, "surviving": hit, "share": hit / tot if tot else np.nan})
summary = summary.merge(pd.DataFrame(surv)[["every", "share"]].rename(columns={"share": "fine_cuts_surviving"}),
                        on="every", how="left")
summary.to_csv(OUT / "bergson_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 200)
print(summary.round(3).to_string(index=False))
