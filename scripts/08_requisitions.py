"""Step 8: test Gilbreth's own claim with his own clock.

The intertitles claim that dating requisitions with two hands gives 21% more
output than with one hand (2,300 vs 1,900 an hour), and two hands plus one foot
61% more (3,050). The camera was hand-cranked, so film time cannot be trusted;
the microchronometer in the frame can (05_clock.py reads it).

Cycles were counted frame by frame from full-length filmstrips of each shot
(every 2nd frame): the free hand's arrival at the paper pile marks a new sheet.
The machine's own rhythm-finders (pose self-similarity, image change) found
the period cleanly only for the two-handed shot; for the others they locked on
to half-cycles (fetch and hand-off), so the count here is by eye.

  python 08_requisitions.py      -> out/requisitions.csv
"""
import numpy as np
import pandas as pd

from common import OUT

EVENTS = {  # frame offsets within the shot where the free hand reaches the paper pile
    26: [17, 43, 66, 89, 109, 134, 157, 179, 201, 225, 242, 264],
    28: [2, 18, 36, 52, 68, 84, 116, 136, 160, 180, 198],      # one long pause after 84
    30: [2, 12, 25, 36, 46, 56, 67, 83, 93, 107, 122, 134],
}
METHOD = {26: "one hand", 28: "two hands", 30: "two hands + one foot"}
CLAIM = {26: 1900, 28: 2300, 30: 3050}

clock = pd.read_csv(OUT / "clock_summary.csv").set_index("shot")
rows = []
for shot, ev in EVENTS.items():
    ev = np.array(ev)
    per = (ev[-1] - ev[0]) / (len(ev) - 1)               # video frames per sheet
    dpf = clock.loc[shot, "deg_per_frame"]
    rows.append({"shot": shot, "method": METHOD[shot], "sheets_counted": len(ev) - 1,
                 "frames_per_sheet": per, "median_interval": float(np.median(np.diff(ev))),
                 "clock_deg_per_frame": dpf, "clock_deg_per_sheet": per * dpf,
                 "claim_per_hour": CLAIM[shot]})
df = pd.DataFrame(rows).set_index("shot")
df["film_time_ratio"] = df.loc[26, "frames_per_sheet"] / df["frames_per_sheet"]
df["clock_ratio"] = df.loc[26, "clock_deg_per_sheet"] / df["clock_deg_per_sheet"]
df["claim_ratio"] = df["claim_per_hour"] / df.loc[26, "claim_per_hour"]
df.to_csv(OUT / "requisitions.csv")
print(df.round(3).to_string())
