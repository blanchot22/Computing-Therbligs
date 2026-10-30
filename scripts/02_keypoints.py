"""Step 2: find bodies and 17 joints with the toolkit's Keypoint R-CNN weights.

  python 02_keypoints.py overview --stride 10            # whole film, ~3 frames/s
  python 02_keypoints.py dense --stride 2 --shots 12,14  # chosen shots, ~15 frames/s

Writes out/kp_<tag>.csv (one row per detected person) and out/kp_<tag>_frames.csv
(every frame that was examined, including ones where nobody was found).
Safe to interrupt: it resumes where it stopped.
"""
import argparse
import csv
import time

import pandas as pd

from common import OUT, iter_frames, load_keypoint_model, run_keypoints, video_meta, video_path

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--shots", default="", help="comma-separated shot ids (default: all)")
ap.add_argument("--min-size", type=int, default=600)
args = ap.parse_args()

video = video_path()
meta = video_meta(video)
shots = pd.read_csv(OUT / "shots.csv")
if args.shots:
    keep = {int(s) for s in args.shots.split(",")}
    shots = shots[shots["shot"].isin(keep)]
wanted = set()
for _, s in shots.iterrows():
    wanted.update(range(int(s["start"]), int(s["end"]) + 1, args.stride))

det_path = OUT / f"kp_{args.tag}.csv"
frm_path = OUT / f"kp_{args.tag}_frames.csv"
done = set()
if frm_path.exists():
    done = set(pd.read_csv(frm_path)["frame"].astype(int))
todo = sorted(wanted - done)
print(f"{len(wanted)} frames wanted, {len(done)} already done, {len(todo)} to go")
if not todo:
    raise SystemExit

model = load_keypoint_model(args.min_size)
header = ["frame", "t", "det", "score", "bx0", "by0", "bx1", "by1"]
for j in range(17):
    header += [f"x{j}", f"y{j}", f"s{j}"]
new_det, new_frm = not det_path.exists(), not frm_path.exists()
fd, ff = open(det_path, "a", newline=""), open(frm_path, "a", newline="")
wd, wf = csv.writer(fd), csv.writer(ff)
if new_det:
    wd.writerow(header)
if new_frm:
    wf.writerow(["frame", "t", "n_det", "max_score"])

t0, n = time.time(), 0
for idx, rgb in iter_frames(video, wanted=set(todo)):
    dets = run_keypoints(model, rgb)
    t = idx / meta["fps"]
    for d, det in enumerate(dets):
        row = [idx, round(t, 3), d, round(det["score"], 4)] + [round(v, 1) for v in det["box"]]
        for j in range(17):
            row += [round(float(det["kp"][j, 0]), 1), round(float(det["kp"][j, 1]), 1),
                    round(float(det["kp"][j, 2]), 2)]
        wd.writerow(row)
    wf.writerow([idx, round(t, 3), len(dets), round(max([d["score"] for d in dets], default=0), 4)])
    n += 1
    if n % 50 == 0:
        fd.flush(); ff.flush()
        rate = (time.time() - t0) / n
        print(f"{n}/{len(todo)} frames, {rate:.2f}s/frame, ~{rate * (len(todo) - n) / 60:.0f} min left",
              flush=True)
fd.close(); ff.close()
print("done", n, "frames in %.1f min" % ((time.time() - t0) / 60))
