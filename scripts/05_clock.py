"""Step 5: read Gilbreth's microchronometer — a custom annotation.

The camera was hand-cranked, so film time drifts; Gilbreth put a fast-running
clock (the microchronometer: white hand on a black dial) in the frame to give
every exposure its true time. This reads the hand's angle in every frame:

  1. the dial's pivot = centre of the disc the moving hand sweeps (temporal
     variance inside a hand-marked box around the dial);
  2. per frame, unroll the dial into a polar strip and average each spoke over
     the hand's length; subtract the shot's median strip (removes the fixed
     ticks and the GILBRETH lettering); the hand is the brightest remaining spoke;
  3. unwrap the angle through time; its slope = how much clock-time passes per
     video frame. Comparing slopes between shots gives their relative cranking
     speed, and counts of work cycles per dial revolution are clock-true.

  python 05_clock.py            # all shots listed in DIALS
Writes out/clock_<shot>.csv and out/clock_summary.csv, out/figures/clock_<shot>.jpg
"""
import cv2
import numpy as np
import pandas as pd

from common import OUT, iter_frames, video_path

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
# approximate box around the microchronometer in each shot: x0, y0, x1, y1
# (shots 40 and 44 also show a dial, but it is small and the camera drifts: unreadable)
DIALS = {
    26: (0, 190, 115, 345),
    28: (0, 190, 110, 340),
    30: (0, 120, 140, 310),
}


def read_shot(shot, box, video, shots):
    a, b = int(shots.loc[shot, "start"]), int(shots.loc[shot, "end"])
    x0, y0, x1, y1 = box
    crops, idx = [], []
    for i, rgb in iter_frames(video, wanted=set(range(a, b + 1))):
        crops.append(cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32))
        idx.append(i)
    stack = np.stack(crops)
    med = np.median(stack, axis=0)
    # 1. the hand in each frame = bright pixels that are not part of the fixed dial;
    #    fit a line through them (the hand is a bar through the pivot)
    lines = []
    for g in stack:
        diff = cv2.GaussianBlur(g - med, (0, 0), 1.0)
        thr = max(20.0, np.percentile(diff, 98.5))
        ys, xs = np.nonzero(diff > thr)
        if len(xs) < 15:
            lines.append(None)
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        m = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - m, full_matrices=False)
        elong = sv[0] / (sv[1] + 1e-6)
        lines.append((m, vt[0], pts, elong))
    # 2. pivot = point closest to all the hand lines (least squares, then trimmed)
    use = [l for l in lines if l is not None and l[3] > 3]

    def solve(ls):
        A, bb = np.zeros((2, 2)), np.zeros(2)
        for m, u, _, _ in ls:
            P = np.eye(2) - np.outer(u, u)
            A += P; bb += P @ m
        return np.linalg.solve(A, bb)

    piv = solve(use)
    for _ in range(3):
        dist = [np.linalg.norm((np.eye(2) - np.outer(u, u)) @ (piv - m)) for m, u, _, _ in use]
        keep = [l for l, dd in zip(use, dist) if dd < np.percentile(dist, 70)]
        piv = solve(keep)
    cx, cy = float(piv[0]), float(piv[1])
    # 3. angle of the pointer: the end of the bar that reaches farther from the pivot
    ang, contrast, reach = [], [], []
    for l in lines:
        if l is None or l[3] <= 3:
            ang.append(np.nan); contrast.append(0.0); reach.append(0.0)
            continue
        m, u, pts, elong = l
        t = (pts - piv) @ u
        hi, lo = np.percentile(t, 97), -np.percentile(t, 3)
        direc = u if hi >= lo else -u
        ang.append(float(np.degrees(np.arctan2(direc[1], direc[0])) % 360))
        contrast.append(float(elong))
        reach.append(float(max(hi, lo)))
    ang, contrast = np.array(ang), np.array(contrast)
    R = float(np.nanpercentile(reach, 90))
    ok = ~np.isnan(ang) & (np.array(reach) > 0.6 * R)
    # 4. unwrap through time (steps well under 180 degrees per frame)
    a_f = pd.Series(ang).where(ok).interpolate(limit_direction="both").to_numpy()
    d = (np.diff(a_f) + 180) % 360 - 180
    unwrapped = np.r_[a_f[0], a_f[0] + np.cumsum(d)]
    f = np.array(idx, float)
    # clock speed from clean readings only: step between consecutive confident
    # frames; drop impossible steps (misreads) and average what is left
    raw = np.array(ang)
    inc = (np.diff(raw) + 180) % 360 - 180
    good = ok[1:] & ok[:-1] & (np.diff(f) == 1) & (inc > -3) & (inc < 40)
    slope = float(inc[good].mean())
    # variability between clean runs (cranking was not steady)
    run_rates, cur = [], []
    for g_, v in zip(good, inc):
        if g_:
            cur.append(v)
        elif len(cur) >= 6:
            run_rates.append(np.mean(cur)); cur = []
        else:
            cur = []
    if len(cur) >= 6:
        run_rates.append(np.mean(cur))
    resid_fit = np.zeros(len(f))
    unwrapped = raw[0] + slope * (f - f[0])
    out = pd.DataFrame({"frame": idx, "angle": ang, "unwrapped": unwrapped, "contrast": contrast,
                        "ok": ok})
    out.to_csv(OUT / f"clock_{shot:03d}.csv", index=False)
    # check image: dial crop with the detected hand drawn on a few frames
    tiles = []
    for j in np.linspace(0, len(stack) - 1, 6).astype(int):
        t = cv2.cvtColor(stack[j].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        c = (int(cx * 3), int(cy * 3))
        if not np.isnan(ang[j]):
            th = np.radians(ang[j])
            e = (int((cx + R * 0.9 * np.cos(th)) * 3), int((cy + R * 0.9 * np.sin(th)) * 3))
            cv2.line(t, c, e, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(t, c, 4, (0, 200, 255), -1)
        cv2.putText(t, f"frame {idx[j]}  {ang[j]:.0f} deg", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(str(FIG / f"clock_{shot:03d}.jpg"), np.concatenate(tiles, 1))
    return {"shot": shot, "frames": len(idx), "pivot_x": cx + x0, "pivot_y": cy + y0, "radius": R,
            "deg_per_frame": slope, "revolutions": slope * (f[-1] - f[0]) / 360,
            "clean_steps": int(good.sum()), "n_runs": len(run_rates),
            "run_rate_min": float(min(run_rates)) if run_rates else np.nan,
            "run_rate_max": float(max(run_rates)) if run_rates else np.nan,
            "share_confident": float(ok.mean())}


if __name__ == "__main__":
    video = video_path()
    shots = pd.read_csv(OUT / "shots.csv").set_index("shot")
    rows = [read_shot(s, box, video, shots) for s, box in DIALS.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "clock_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
