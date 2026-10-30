"""Step 1: cut the film into shots with the Distant Viewing Toolkit's shot
detector (TransNetV2), save a keyframe per shot, flag and read intertitles."""
import json
import re
import subprocess

import cv2
import numpy as np
import pandas as pd

import dvt
from common import OUT, iter_frames, video_meta, video_path

video = video_path()
meta = video_meta(video)
print("film:", video.name, meta)
json.dump(meta, open(OUT / "video_meta.json", "w"), indent=2)

# --- 1. shot boundaries -------------------------------------------------------
sb = dvt.AnnoShotBreaks()
res = sb.run(str(video))
starts, ends = res["scenes"]["start"], res["scenes"]["end"]
fps = meta["fps"]
shots = pd.DataFrame({"shot": np.arange(len(starts)), "start": starts, "end": ends})
shots["start_s"] = shots["start"] / fps
shots["end_s"] = (shots["end"] + 1) / fps
shots["dur_s"] = shots["end_s"] - shots["start_s"]
shots["mid"] = ((shots["start"] + shots["end"]) // 2).astype(int)
np.save(OUT / "shot_frame_pred.npy", res["frames"]["single_frame_pred"])
print(len(shots), "shots; median length %.1fs" % shots["dur_s"].median())

# --- 2. keyframes + simple per-shot image statistics --------------------------
kdir = OUT / "keyframes"
kdir.mkdir(exist_ok=True)
wanted = set(shots["mid"].tolist())
stats = {}
for idx, rgb in iter_frames(video, wanted=wanted):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(kdir / f"frame_{idx:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    stats[idx] = {"mean": float(gray.mean()), "std": float(gray.std())}
shots["mean_lum"] = shots["mid"].map(lambda m: stats.get(m, {}).get("mean"))
shots["std_lum"] = shots["mid"].map(lambda m: stats.get(m, {}).get("std"))


# --- 3. intertitles: OCR the keyframe with tesseract --------------------------
def ocr(path):
    """Return (confident text, number of confident words) for a keyframe.
    Tesseract's per-word confidence separates real lettering (>75) from the
    noise it reads into grain, faces and machinery (<70)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:  # light text on dark card -> invert for tesseract
        bw = 255 - bw
    tmp = path.with_suffix(".ocr.png")
    cv2.imwrite(str(tmp), bw)
    tsv = subprocess.run(["tesseract", str(tmp), "stdout", "--psm", "6", "tsv"],
                         capture_output=True, text=True).stdout
    tmp.unlink(missing_ok=True)
    words = []
    for line in tsv.splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 12 and f[11].strip():
            words.append((f[11].strip(), float(f[10])))
    good = [w for w, c in words if c >= 75 and re.search(r"[A-Za-z]{2,}", w)]
    return " ".join(good), sum(1 for w in good if re.search(r"[A-Za-z]{3,}", w))


texts, counts = [], []
for _, s in shots.iterrows():
    t, n = ocr(kdir / f"frame_{int(s['mid']):06d}.jpg")
    texts.append(t)
    counts.append(n)
shots["ocr"] = texts
shots["n_words"] = counts
shots["intertitle"] = shots["n_words"] >= 2
shots.to_csv(OUT / "shots.csv", index=False)
print(shots["intertitle"].sum(), "shots look like intertitles")
print(shots[shots["intertitle"]][["shot", "start_s", "ocr"]].to_string(max_colwidth=90))
