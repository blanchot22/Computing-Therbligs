"""Step 6: repetition — the fingerprint of a performance.

For the main worker in each shot, describe the upper body in every sampled frame
(shoulders, elbows, wrists, measured from the shoulders in shoulder-widths) and
compare every moment with every other moment. The result is a self-similarity
map: a body that keeps returning to the same postures draws diagonal stripes.

  * period      = the lag at which the body most nearly returns to itself
  * return index ('mechanical index', after Bergson's "something mechanical
                  encrusted on the living") = how closely it returns, 0 to 1
  * cycles      = work cycles counted in the shot
  * clock rate  = cycles per revolution of Gilbreth's microchronometer (shots
                  where the dial is readable), i.e. corrected for hand-cranking

  python 06_rhythm.py
Writes out/rhythm_summary.csv and out/figures/rhythm_<shot>.png
"""
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import find_peaks, savgol_filter

from common import OUT

FIG = OUT / "figures"
KP = [5, 6, 7, 8, 9, 10]      # shoulders, elbows, wrists
KP_THR = 0.5
SHOTS = {  # shot: (task, method)
    26: ("Dating requisitions", "one hand"),
    28: ("Dating requisitions", "two hands"),
    30: ("Dating requisitions", "two hands + foot"),
    17: ("Bricklaying", "original"),
    32: ("Polishing soap", "original"), 33: ("Polishing soap", "original"),
    34: ("Polishing soap", "original"), 36: ("Polishing soap", "Gilbreth"),
    38: ("Labelling cartons", "original"), 40: ("Labelling cartons", "Gilbreth"),
    42: ("Packing soap", "original"), 43: ("Packing soap", "original"),
    44: ("Packing soap", "original"), 46: ("Packing soap", "Gilbreth"),
    47: ("Packing soap", "Gilbreth"),
    67: ("Laboratory study", "one & two hands"), 70: ("Laboratory study", "one & two hands"),
    72: ("Pre-positioning", "Gilbreth"), 74: ("Pre-positioning", "Gilbreth"),
}
GILBRETH_CLAIM = {26: 1900, 28: 2300, 30: 3050}   # requisitions per hour, from the intertitles

fps = json.load(open(OUT / "video_meta.json"))["fps"]
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "dense"
# one worker per shot: take the most confident person in every sampled frame
# (the model often finds the same worker two or three times with overlapping
# boxes, which breaks frame-to-frame tracking)
parts = sorted(OUT.glob(f"kp_{TAG}?.csv")) if not (OUT / f"kp_{TAG}.csv").exists() else [OUT / f"kp_{TAG}.csv"]
det = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["frame", "det"])
frm = pd.concat([pd.read_csv(str(p)[:-4] + "_frames.csv") for p in parts]).drop_duplicates("frame")
shots_tab = pd.read_csv(OUT / "shots.csv").set_index("shot")
MIN_SCORE = 0.3
clock = pd.read_csv(OUT / "clock_summary.csv").set_index("shot") if (OUT / "clock_summary.csv").exists() else None


def pose_series(tr):
    """Frames x (len(KP)*2) array in shoulder-widths from the shoulder midpoint."""
    sw_ok = (tr["s5"] > KP_THR) & (tr["s6"] > KP_THR)
    scale = np.hypot(tr["x5"] - tr["x6"], tr["y5"] - tr["y6"])[sw_ok].median()
    cx = ((tr["x5"] + tr["x6"]) / 2).where(sw_ok).interpolate(limit_direction="both")
    cy = ((tr["y5"] + tr["y6"]) / 2).where(sw_ok).interpolate(limit_direction="both")
    cols = []
    for j in KP:
        ok = tr[f"s{j}"] > KP_THR
        cols.append(((tr[f"x{j}"] - cx) / scale).where(ok))
        cols.append(((tr[f"y{j}"] - cy) / scale).where(ok))
    return np.stack([c.to_numpy(dtype=float) for c in cols], 1), float(scale)


def self_distance(P):
    """Mean keypoint distance between every pair of frames (nan-aware)."""
    n, k = P.shape[0], len(KP)
    xy = P.reshape(n, k, 2)
    D = np.full((n, n), np.nan)
    for i in range(n):
        dd = np.sqrt(((xy[i][None] - xy) ** 2).sum(-1))       # n x k
        D[i] = np.nanmean(np.where(np.isnan(dd), np.nan, dd), axis=1)
    return D


rows = []
for shot, (task, method) in SHOTS.items():
    if shot not in shots_tab.index:
        continue
    a_, b_ = shots_tab.loc[shot, "start"], shots_tab.loc[shot, "end"]
    grid = frm[(frm["frame"] >= a_) & (frm["frame"] <= b_)][["frame"]].sort_values("frame")
    sd = det[(det["frame"] >= a_) & (det["frame"] <= b_) & (det["score"] >= MIN_SCORE)]
    if len(grid) < 30 or sd.empty:
        continue
    top = sd.sort_values("score").groupby("frame").tail(1)
    tr = grid.merge(top, on="frame", how="left").reset_index(drop=True)
    for j in range(17):                     # frames with nobody found: all keypoints missing
        tr[f"s{j}"] = tr[f"s{j}"].fillna(-99)
    P, scale = pose_series(tr)
    frames = tr["frame"].to_numpy()
    D = self_distance(P)
    n = len(D)
    step = float(np.median(np.diff(frames)))                     # video frames per sample
    lags = np.arange(1, n // 2)
    prof = np.array([np.nanmean(np.diagonal(D, L)) for L in lags])
    prof_s = pd.Series(prof).rolling(3, center=True, min_periods=1).mean().to_numpy()
    min_lag = max(2, int(round(0.4 * fps / step)))               # ignore returns faster than 0.4 s
    mins, props = find_peaks(-prof_s[min_lag - 1:], prominence=0.02)
    if len(mins):
        # strongest early return: the most prominent of the first three minima
        cand = mins[:3]
        best = cand[np.argmax(props["prominences"][:3])]
        L = lags[min_lag - 1 + best]
        before = prof_s[:min_lag - 1 + best + 1].max()
        ret = float((before - prof_s[min_lag - 1 + best]) / before)
    else:
        L, ret = np.nan, 0.0
    period_frames = L * step if not np.isnan(L) else np.nan
    span = frames[-1] - frames[0]
    # cycles: peaks of the main axis of postural change
    Pf = pd.DataFrame(P).interpolate(limit_direction="both").fillna(0).to_numpy()
    Pc = Pf - Pf.mean(0)
    pc1 = Pc @ np.linalg.svd(Pc, full_matrices=False)[2][0]
    if len(pc1) > 9:
        pc1 = savgol_filter(pc1, 7, 2)
    peaks, _ = find_peaks(pc1, distance=max(1, int(0.6 * L)) if not np.isnan(L) else 3,
                          prominence=0.25 * np.std(pc1))
    row = {"shot": shot, "task": task, "method": method, "track_samples": n,
           "film_seconds": span / fps, "period_frames": period_frames,
           "period_film_s": period_frames / fps if not np.isnan(period_frames) else np.nan,
           "return_index": ret, "cycles_by_period": span / period_frames if period_frames else np.nan,
           "cycles_by_peaks": len(peaks)}
    if clock is not None and shot in clock.index:
        dpf = clock.loc[shot, "deg_per_frame"]
        row["clock_deg_per_frame"] = dpf
        row["dial_revs_per_cycle"] = period_frames * dpf / 360
        row["cycles_per_dial_rev"] = 360 / (period_frames * dpf)
    if shot in GILBRETH_CLAIM:
        row["gilbreth_claim_per_hour"] = GILBRETH_CLAIM[shot]
    rows.append(row)
    # --- picture: the self-similarity map, light where two moments match -------
    sigma = np.nanmedian(D)
    S = np.exp(-(np.nan_to_num(D, nan=3 * sigma) / sigma) ** 2)
    img = (np.clip(S, 0, 1) ** 1.2 * 255).astype(np.uint8)
    img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
    lamp = np.array([255, 204, 107], np.float32) / 255
    rgb = (img[..., None].astype(np.float32) / 255 * lamp * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).text((10, 8), f"shot {shot} · {task} · {method}", fill=(240, 239, 236))
    im.save(FIG / f"rhythm_{shot:03d}.png")
    np.save(OUT / f"rhythm_{shot:03d}_D.npy", D)
    np.save(OUT / f"rhythm_{shot:03d}_frames.npy", frames)

summary = pd.DataFrame(rows)
base = summary.set_index("shot")
if {26, 28, 30} <= set(base.index) and "cycles_per_dial_rev" in base:
    for s in (26, 28, 30):
        summary.loc[summary["shot"] == s, "rate_vs_one_hand_clock"] = \
            base.loc[s, "cycles_per_dial_rev"] / base.loc[26, "cycles_per_dial_rev"]
        summary.loc[summary["shot"] == s, "rate_vs_one_hand_filmtime"] = \
            base.loc[26, "period_frames"] / base.loc[s, "period_frames"]
        summary.loc[summary["shot"] == s, "gilbreth_ratio"] = GILBRETH_CLAIM[s] / GILBRETH_CLAIM[26]
summary.to_csv(OUT / "rhythm_summary.csv", index=False)
pd.set_option("display.width", 220)
print(summary.round(3).to_string(index=False))
