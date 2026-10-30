"""Step 3: follow the wrists through time and cut their paths into motion units.

  python 03_analyse.py dense

For every densely-sampled shot:
  1. link person detections frame to frame into tracks (box overlap);
  2. for each track take both wrists, drop low-confidence points, fill short
     gaps, smooth, and measure in shoulder-widths so near and far workers are
     comparable;
  3. compute hand speed and cut each path at moments of (near-)stillness: the
     piece between two speed minima is a 'motion unit', the machine's candidate
     therblig (Gilbreth also cut motions where the hand changes what it does);
  4. describe each unit (duration, path length, straightness, peak speed,
     direction, height of the hand) and group similar units with k-means into
     the machine's 'alphabet'.
Writes out/tracks_<tag>.csv, out/units_<tag>.csv, out/alphabet_<tag>.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import L_WRIST, OUT, R_WRIST

ap = argparse.ArgumentParser()
ap.add_argument("tag", nargs="?", default="dense")
ap.add_argument("--every", type=int, default=1,
                help="keep every Nth sampled frame (Bergson test: coarser snapshots)")
args = ap.parse_args()
TAG = args.tag
OUT_TAG = TAG if args.every == 1 else f"{TAG}_every{args.every}"
SMOOTH_S = float(os.environ.get("SMOOTH_S", "0"))  # if set, smoothing window in seconds of film, not samples
if SMOOTH_S: OUT_TAG += f"_smooth{SMOOTH_S}"
MIN_PERSON = 0.6      # person-detection score needed to be tracked
KP_THR = 0.5          # wrist keypoint score (heatmap logit) needed to count
MAX_GAP = 3           # samples of missing wrist we bridge by interpolation
MIN_TRACK = 12        # samples
V_STILL = 0.4         # shoulder-widths per second: below this the hand is 'still'
K_RANGE = range(4, 13)

shots = pd.read_csv(OUT / "shots.csv")
parts = sorted(OUT.glob(f"kp_{TAG}?.csv")) if not (OUT / f"kp_{TAG}.csv").exists() else [OUT / f"kp_{TAG}.csv"]
det = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["frame", "det"]).reset_index(drop=True)
frm = pd.concat([pd.read_csv(str(p)[:-4] + "_frames.csv") for p in parts]).drop_duplicates("frame")
frm = frm.sort_values("frame")
fps = json.load(open(OUT / "video_meta.json"))["fps"]
if args.every > 1:
    keep_frames = set(frm["frame"].iloc[::args.every])
    frm = frm[frm["frame"].isin(keep_frames)]
    det = det[det["frame"].isin(keep_frames)]

shot_of = np.zeros(int(shots["end"].max()) + 1, dtype=int)
for _, s in shots.iterrows():
    shot_of[int(s["start"]):int(s["end"]) + 1] = int(s["shot"])
det["shot"] = shot_of[det["frame"].clip(upper=len(shot_of) - 1)]
frm["shot"] = shot_of[frm["frame"].clip(upper=len(shot_of) - 1)]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link(shot_det, shot_frames):
    """Greedy box-overlap tracking. Returns detections with a 'track' column."""
    pos = {f: i for i, f in enumerate(shot_frames)}
    active, next_id, labels = [], 0, {}
    for f in shot_frames:
        rows = shot_det[shot_det["frame"] == f]
        cand = [(i, [r.bx0, r.by0, r.bx1, r.by1]) for i, r in rows.iterrows()]
        active = [t for t in active if pos[f] - t["pos"] <= MAX_GAP + 1]
        pairs = sorted(((iou(t["box"], b), ti, i) for ti, t in enumerate(active) for i, b in cand),
                       reverse=True)
        used_t, used_d = set(), set()
        for o, ti, i in pairs:
            if o < 0.3 or ti in used_t or i in used_d:
                continue
            used_t.add(ti); used_d.add(i)
            labels[i] = active[ti]["id"]
            active[ti].update(box=dict(cand)[i], pos=pos[f])
        for i, b in cand:
            if i not in used_d:
                labels[i] = next_id
                active.append({"id": next_id, "box": b, "pos": pos[f]})
                next_id += 1
    out = shot_det.copy()
    out["track"] = pd.Series(labels)
    return out


def runs(mask):
    """Start/end (inclusive) indices of consecutive True runs."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2] - 1))


track_rows, unit_rows, linked = [], [], []
for shot_id, sdet in det[det["score"] >= MIN_PERSON].groupby("shot"):
    sframes = frm.loc[frm["shot"] == shot_id, "frame"].tolist()
    sdet = link(sdet, sframes)
    linked.append(sdet)
    for tid, tr in sdet.groupby("track"):
        tr = tr.sort_values("frame")
        if len(tr) < MIN_TRACK:
            continue
        # duplicated film frames (hand-cranked footage transferred to video)
        # give identical detections: keep the first of each identical run
        sig = tr[[f"x{j}" for j in range(17)]].round(0).astype(int).astype(str).agg("|".join, axis=1)
        tr = tr[sig != sig.shift()]
        sw_ok = (tr["s5"] > KP_THR) & (tr["s6"] > KP_THR)
        sw = np.hypot(tr["x5"] - tr["x6"], tr["y5"] - tr["y6"])[sw_ok]
        scale = float(sw.median()) if len(sw) >= 5 else float((tr["by1"] - tr["by0"]).median() / 4)
        if not scale or scale < 5:
            continue
        sh_x = ((tr["x5"] + tr["x6"]) / 2).to_numpy()
        sh_y = ((tr["y5"] + tr["y6"]) / 2).to_numpy()
        t = tr["frame"].to_numpy() / fps
        for wj, hand in ((L_WRIST, "left"), (R_WRIST, "right")):
            x = tr[f"x{wj}"].to_numpy(dtype=float, copy=True)
            y = tr[f"y{wj}"].to_numpy(dtype=float, copy=True)
            ok = tr[f"s{wj}"].to_numpy() > KP_THR
            x[~ok] = np.nan; y[~ok] = np.nan
            xs = pd.Series(x).interpolate(limit=MAX_GAP, limit_area="inside").to_numpy()
            ys = pd.Series(y).interpolate(limit=MAX_GAP, limit_area="inside").to_numpy()
            valid = ~np.isnan(xs)
            sx, sy, speed = np.full_like(xs, np.nan), np.full_like(ys, np.nan), np.full_like(xs, np.nan)
            for a, b in runs(valid):
                n = b - a + 1
                if n < 5:
                    continue
                w = min(7, n if n % 2 else n - 1)
                if SMOOTH_S:
                    dts = np.median(np.diff(t[a:b + 1]))
                    w = int(round(SMOOTH_S / dts)); w = max(3, w if w % 2 else w + 1); w = min(w, n if n % 2 else n - 1)
                po = min(2, w - 1)
                sx[a:b + 1] = savgol_filter(xs[a:b + 1], w, po)
                sy[a:b + 1] = savgol_filter(ys[a:b + 1], w, po)
                vx = np.gradient(sx[a:b + 1], t[a:b + 1])
                vy = np.gradient(sy[a:b + 1], t[a:b + 1])
                speed[a:b + 1] = np.hypot(vx, vy) / scale
            for i in range(len(t)):
                track_rows.append((shot_id, tid, hand, int(tr["frame"].iloc[i]), t[i], x[i], y[i],
                                   sx[i], sy[i], speed[i], scale, sh_x[i], sh_y[i]))
            # --- motion units ---------------------------------------------------
            for a, b in runs(~np.isnan(speed)):
                if b - a + 1 < 6:
                    continue
                sp = speed[a:b + 1]
                dt = np.median(np.diff(t[a:b + 1]))
                mins, _ = find_peaks(-sp, prominence=max(0.15, 0.2 * np.percentile(sp, 90)),
                                     distance=max(1, int(0.15 / dt)))
                cuts = [0] + list(mins) + [len(sp) - 1]
                for c0, c1 in zip(cuts[:-1], cuts[1:]):
                    if c1 - c0 < 2:
                        continue
                    i0, i1 = a + c0, a + c1
                    px, py = sx[i0:i1 + 1], sy[i0:i1 + 1]
                    steps = np.hypot(np.diff(px), np.diff(py))
                    path = steps.sum() / scale
                    dx, dy = (px[-1] - px[0]) / scale, (py[-1] - py[0]) / scale
                    disp = np.hypot(dx, dy)
                    unit_rows.append({
                        "shot": shot_id, "track": tid, "hand": hand,
                        "f0": int(tr["frame"].iloc[i0]), "f1": int(tr["frame"].iloc[i1]),
                        "t0": t[i0], "t1": t[i1], "dur": t[i1] - t[i0],
                        "path": path, "disp": disp, "dx": dx, "dy": dy,
                        "straight": disp / path if path > 1e-6 else 0.0,
                        "peak": float(np.nanmax(speed[i0:i1 + 1])),
                        "mean_speed": path / max(t[i1] - t[i0], 1e-6),
                        "rel_y": float(np.nanmean((py - sh_y[i0:i1 + 1]) / scale)),
                        "scale": scale,
                    })

tracks = pd.DataFrame(track_rows, columns=["shot", "track", "hand", "frame", "t", "x", "y", "sx", "sy",
                                           "speed", "scale", "sh_x", "sh_y"])
units = pd.DataFrame(unit_rows)
tracks.to_csv(OUT / f"tracks_{OUT_TAG}.csv", index=False)
pd.concat(linked).to_csv(OUT / f"det_tracks_{OUT_TAG}.csv", index=False)
print(f"{tracks.groupby(['shot', 'track']).ngroups} tracks, {len(units)} motion units")
if len(units) < 20:
    units.to_csv(OUT / f"units_{OUT_TAG}.csv", index=False)
    raise SystemExit("too few units to cluster")

# --- the machine's alphabet ------------------------------------------------------
units["still"] = units["peak"] < V_STILL
ang = np.arctan2(units["dy"], units["dx"])
feats = pd.DataFrame({
    "log_dur": np.log(units["dur"].clip(lower=0.05)),
    "log_path": np.log(units["path"] + 0.05),
    "straight": units["straight"],
    "log_peak": np.log(units["peak"] + 0.05),
    "dir_x": np.cos(ang) * units["straight"],
    "dir_y": np.sin(ang) * units["straight"],
    "rel_y": units["rel_y"].clip(-3, 3),
})
X = StandardScaler().fit_transform(feats)
scores = {}
for k in K_RANGE:
    if k >= len(units):
        break
    lab = KMeans(k, n_init=20, random_state=0).fit_predict(X)
    scores[k] = float(silhouette_score(X, lab, sample_size=min(4000, len(X)), random_state=0))
best_k = max(scores, key=scores.get)
km = KMeans(best_k, n_init=20, random_state=0).fit(X)
units["letter"] = km.labels_
units["dist"] = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
km17 = KMeans(17, n_init=20, random_state=0).fit(X) if len(units) > 60 else None
if km17 is not None:
    units["letter17"] = km17.labels_

COMPASS = ["rightward", "down-right", "downward", "down-left", "leftward", "up-left", "upward", "up-right"]


def describe(g):
    peak, path, straight = g["peak"].median(), g["path"].median(), g["straight"].median()
    if peak < V_STILL:
        return "stillness"
    speed = "fast" if peak > 3 else ("brisk" if peak > 1.5 else "slow")
    size = "long" if path > 1.2 else ("medium" if path > 0.5 else "short")
    if straight < 0.55:
        shape = "looping" if straight < 0.3 else "curved"
        return f"{speed} {size} {shape}"
    a = np.degrees(np.arctan2(g["dy"].median(), g["dx"].median())) % 360
    return f"{speed} {size} {COMPASS[int(((a + 22.5) % 360) // 45)]} stroke"


letters = []
for k, g in units.groupby("letter"):
    medoid = g.sort_values("dist").index[0]
    letters.append({
        "letter": int(k), "name": describe(g), "n": int(len(g)),
        "share": float(len(g) / len(units)),
        "median_duration_s": float(g["dur"].median()), "median_path_sw": float(g["path"].median()),
        "median_peak_speed": float(g["peak"].median()), "median_straightness": float(g["straight"].median()),
        "median_hand_height": float(g["rel_y"].median()),
        "medoid_unit": int(medoid), "shots": sorted(int(s) for s in g["shot"].unique()),
    })
letters.sort(key=lambda d: -d["n"])
units.to_csv(OUT / f"units_{OUT_TAG}.csv", index=True, index_label="unit")
json.dump({"silhouette_by_k": scores, "best_k": int(best_k), "n_units": int(len(units)),
           "n_still_units": int(units["still"].sum()), "letters": letters,
           "params": {"MIN_PERSON": MIN_PERSON, "KP_THR": KP_THR, "V_STILL": V_STILL}},
          open(OUT / f"alphabet_{OUT_TAG}.json", "w"), indent=2)
print("silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
for d in letters:
    print(f"  letter {d['letter']:2d}: {d['name']:32s} n={d['n']:4d}  dur={d['median_duration_s']:.2f}s"
          f"  path={d['median_path_sw']:.2f}sw  peak={d['median_peak_speed']:.2f}sw/s")
