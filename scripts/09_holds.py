"""Step 9: holds — the therbligs of stillness.

Gilbreth counted 'hold', 'unavoidable delay', 'avoidable delay', 'rest for
overcoming fatigue' and 'plan' as separate motions. All of them look the same
to a pose model: a wrist that stays put. This measures how much tracked hand
time is held still: a window of at least HOLD_S seconds in which the smoothed
wrist stays within RADIUS shoulder-widths of where it started.

It also reports how much the model's estimate of a held wrist trembles
(median frame-to-frame step inside holds): the model's own noise, which the
motion-unit step cuts into 'slow short strokes'.

  python 09_holds.py      -> out/holds.csv
"""
import json

import numpy as np
import pandas as pd

from common import OUT

HOLD_S, RADIUS = 0.5, 0.25
SHOTS = {
    17: ("Bricklaying", "original"),
    32: ("Polishing soap", "original"), 33: ("Polishing soap", "original"),
    34: ("Polishing soap", "original"), 36: ("Polishing soap", "Gilbreth"),
    38: ("Labelling cartons", "original"), 40: ("Labelling cartons", "Gilbreth"),
    42: ("Packing soap", "original"), 43: ("Packing soap", "original"),
    44: ("Packing soap", "original"), 46: ("Packing soap", "Gilbreth"),
    47: ("Packing soap", "Gilbreth"),
}
tr = pd.read_csv(OUT / "tracks_dense.csv").dropna(subset=["sx"])
rows = []
for (shot, track, hand), g in tr.groupby(["shot", "track", "hand"]):
    if shot not in SHOTS or len(g) < 10:
        continue
    g = g.sort_values("t")
    t = g["t"].to_numpy(); x = g["sx"].to_numpy() / g["scale"].iloc[0]; y = g["sy"].to_numpy() / g["scale"].iloc[0]
    held = np.zeros(len(t), bool)
    j = 0
    for i in range(len(t)):
        j = max(j, i)
        while j + 1 < len(t) and t[j + 1] - t[j] < 0.3 and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= RADIUS:
            j += 1
        if t[j] - t[i] >= HOLD_S:
            held[i:j + 1] = True
    dt = np.r_[np.diff(t), 0]
    dt[dt > 0.3] = 0
    steps = np.hypot(np.diff(x), np.diff(y))
    inside = held[1:] & held[:-1]
    rows.append({"shot": shot, "task": SHOTS[shot][0], "method": SHOTS[shot][1], "hand": hand,
                 "tracked_s": dt.sum(), "held_s": dt[held].sum(),
                 "tremble_sw": float(np.median(steps[inside])) if inside.any() else np.nan})
df = pd.DataFrame(rows)
df.to_csv(OUT / "holds.csv", index=False)
agg = df.groupby(["task", "method"]).agg(tracked_s=("tracked_s", "sum"), held_s=("held_s", "sum")).reset_index()
agg["held_share"] = agg["held_s"] / agg["tracked_s"]
agg.to_csv(OUT / "holds_by_method.csv", index=False)
print(agg.round(3).to_string(index=False))
print("overall held share: %.3f" % (df["held_s"].sum() / df["tracked_s"].sum()),
      "| median tremble inside holds: %.3f shoulder-widths per sample" % df["tremble_sw"].median())
