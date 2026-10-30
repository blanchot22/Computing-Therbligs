"""Step 4: pictures.

  python 04_figures.py contact                     # the film at a glance
  python 04_figures.py cyclegraph SHOT TRACK [--hands left,right] [--name NAME]
  python 04_figures.py alphabet [dense]            # the machine's letters
  python 04_figures.py remainder [overview]        # candidate frames the model gets wrong
  python 04_figures.py frame FRAME [--tag overview]  # one frame with its skeletons
"""
import argparse
import json
import subprocess
import string

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import OUT, dets_frame, draw_skeleton, iter_frames, read_frame, video_path

FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
FONT_REG = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
# light-trail colours (RGB 0-1): warm lamp for the left hand, cool lamp for the right
LAMP = {"left": (1.00, 0.80, 0.42), "right": (0.62, 0.86, 1.00)}


def font(size, regular=False):
    try:
        return ImageFont.truetype(FONT_REG if regular else FONT, size)
    except OSError:
        return ImageFont.load_default()


def tc(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------- contact
def contact(cols=10, tile_w=192):
    shots = pd.read_csv(OUT / "shots.csv")
    frames = pd.read_csv(OUT / "kp_overview_frames.csv") if (OUT / "kp_overview_frames.csv").exists() else None
    tile_h = int(tile_w * 3 / 4)
    pad, lab = 6, 34
    rows = int(np.ceil(len(shots) / cols))
    W, H = cols * (tile_w + pad) + pad, rows * (tile_h + lab + pad) + pad
    sheet = Image.new("RGB", (W, H), (14, 14, 13))
    d = ImageDraw.Draw(sheet)
    f1, f2 = font(13), font(11, regular=True)
    for i, s in shots.iterrows():
        r, c = divmod(i, cols)
        x, y = pad + c * (tile_w + pad), pad + r * (tile_h + lab + pad)
        im = Image.open(OUT / "keyframes" / f"frame_{int(s['mid']):06d}.jpg").convert("RGB")
        sheet.paste(im.resize((tile_w, tile_h), Image.LANCZOS), (x, y))
        share = None
        if frames is not None:
            sf = frames[(frames["frame"] >= s["start"]) & (frames["frame"] <= s["end"])]
            if len(sf):
                share = float((sf["max_score"] >= 0.8).mean())
        # bar under the tile: share of sampled frames where the model is sure it sees a person
        d.rectangle([x, y + tile_h + 2, x + tile_w - 1, y + tile_h + 5], fill=(52, 52, 49))
        if share:
            d.rectangle([x, y + tile_h + 2, x + int((tile_w - 1) * share), y + tile_h + 5], fill=(255, 204, 107))
        label = f"{int(s['shot'])}  ·  {tc(s['start_s'])}  ·  {s['dur_s']:.1f}s"
        d.text((x, y + tile_h + 9), label, font=f1, fill=(232, 231, 226))
        if s["intertitle"]:
            d.text((x, y + tile_h + 23), "intertitle", font=f2, fill=(160, 158, 150))
    sheet.save(FIG / "film_at_a_glance.jpg", quality=88)
    print("wrote", FIG / "film_at_a_glance.jpg", sheet.size)


# --------------------------------------------------------------------------- cyclegraph
def _glow(layer):
    return np.clip(layer + 1.2 * cv2.GaussianBlur(layer, (0, 0), 2.5)
                   + 0.9 * cv2.GaussianBlur(layer, (0, 0), 9), 0, 1)


def _draw_trails(canvas, pts_by_hand, scale, upto=None):
    """Draw chronocyclegraph-style trails: a fine line plus a 'flash' dot at every
    sample (spacing shows speed, as the pulsing lamps did for Gilbreth)."""
    for hand, pts in pts_by_hand.items():
        col = LAMP[hand]
        prev = None
        for fr, x, y in pts:
            if upto is not None and fr > upto:
                break
            if np.isnan(x):
                prev = None
                continue
            p = (int(x * scale), int(y * scale))
            if prev is not None:
                cv2.line(canvas, prev, p, tuple(0.55 * c for c in col), max(1, scale // 2), cv2.LINE_AA)
            cv2.circle(canvas, p, max(2, scale), col, -1, cv2.LINE_AA)
            prev = p


def cyclegraph(shot, track, hands=("left", "right"), name=None, tag="dense", scale=2, darken=0.33,
               animate=True):
    video = video_path()
    tracks = pd.read_csv(OUT / f"tracks_{tag}.csv")
    tr = tracks[(tracks["shot"] == shot) & (tracks["track"] == track)]
    if tr.empty:
        raise SystemExit(f"no track {track} in shot {shot}")
    f0, f1 = int(tr["frame"].min()), int(tr["frame"].max())
    pts = {h: list(tr[tr["hand"] == h].sort_values("frame")[["frame", "sx", "sy"]].itertuples(index=False))
           for h in hands}
    name = name or f"cyclegraph_shot{shot:03d}_track{track}"
    mid = read_frame(video, (f0 + f1) // 2)
    h, w = mid.shape[:2]

    def base_of(rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = cv2.resize(g, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        return np.repeat(g[..., None], 3, axis=2) * darken

    # still: the whole track's trails over the middle frame
    layer = np.zeros((h * scale, w * scale, 3), np.float32)
    _draw_trails(layer, pts, scale)
    img = np.clip(base_of(mid) + _glow(layer), 0, 1)
    cv2.imwrite(str(FIG / f"{name}.jpg"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    # side-by-side: the plain frame next to the machine's reading of it
    plain = cv2.resize(mid, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    pair = np.concatenate([plain, (img * 255).astype(np.uint8)], axis=1)
    cv2.imwrite(str(FIG / f"{name}_pair.jpg"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", FIG / f"{name}.jpg", f"frames {f0}-{f1}")
    if not animate:
        return
    # animation: the trail accumulates over the moving film
    W2, H2 = (w * scale) // 2 * 2, (h * scale) // 2 * 2
    fps = json.load(open(OUT / "video_meta.json"))["fps"]
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(FIG / f"{name}.mp4")],
        stdin=subprocess.PIPE)
    for idx, rgb in iter_frames(video, wanted=set(range(f0, f1 + 1))):
        layer = np.zeros((h * scale, w * scale, 3), np.float32)
        _draw_trails(layer, pts, scale, upto=idx)
        frame = np.clip(base_of(rgb) * 1.6 + _glow(layer), 0, 1)
        ff.stdin.write((frame[:H2, :W2] * 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", FIG / f"{name}.mp4")


# --------------------------------------------------------------------------- alphabet
def alphabet(tag="dense", per_letter=40, cell=300, cols=4, span=1.6):
    units = pd.read_csv(OUT / f"units_{tag}.csv").set_index("unit")
    info = json.load(open(OUT / f"alphabet_{tag}.json"))
    tracks = pd.read_csv(OUT / f"tracks_{tag}.csv")
    groups = {k: g for k, g in tracks.groupby(["shot", "track", "hand"])}
    letters = info["letters"]
    rows = int(np.ceil(len(letters) / cols))
    lab = 64
    W, H = cols * cell, rows * (cell + lab)
    sheet = np.zeros((H, W, 3), np.float32)
    sheet[:] = (14 / 255, 14 / 255, 13 / 255)
    rng = np.random.default_rng(0)
    meta = []
    for i, L in enumerate(letters):
        r, c = divmod(i, cols)
        ox, oy = c * cell, r * (cell + lab)
        layer = np.zeros((cell, cell, 3), np.float32)
        members = units[units["letter"] == L["letter"]]
        pick = members.sort_values("dist").index[:per_letter // 2].tolist()
        rest = [u for u in members.index if u not in pick]
        pick += list(rng.choice(rest, size=min(len(rest), per_letter - len(pick)), replace=False)) if rest else []
        for u in pick + [L["medoid_unit"]]:
            row = units.loc[u]
            g = groups[(row["shot"], row["track"], row["hand"])]
            seg = g[(g["frame"] >= row["f0"]) & (g["frame"] <= row["f1"])].dropna(subset=["sx"])
            if len(seg) < 2:
                continue
            px = (seg["sx"].to_numpy() - seg["sx"].iloc[0]) / row["scale"]
            py = (seg["sy"].to_numpy() - seg["sy"].iloc[0]) / row["scale"]
            pts = np.stack([cell / 2 + px / span * cell / 2, cell / 2 + py / span * cell / 2], 1).astype(int)
            medoid = u == L["medoid_unit"]
            col = (1.0, 0.86, 0.55) if medoid else (0.55, 0.52, 0.45)
            cv2.polylines(layer, [pts], False, col, 3 if medoid else 1, cv2.LINE_AA)
            if medoid:
                cv2.circle(layer, tuple(pts[0]), 5, (1.0, 0.86, 0.55), 1, cv2.LINE_AA)
                cv2.circle(layer, tuple(pts[-1]), 4, (1.0, 0.86, 0.55), -1, cv2.LINE_AA)
        cv2.circle(layer, (cell // 2, cell // 2), 2, (0.4, 0.4, 0.38), -1, cv2.LINE_AA)
        sheet[oy:oy + cell, ox:ox + cell] = np.maximum(sheet[oy:oy + cell, ox:ox + cell], _glow(layer))
        meta.append((ox, oy, L))
    im = Image.fromarray((sheet * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for i, (ox, oy, L) in enumerate(meta):
        d.text((ox + 14, oy + 10), string.ascii_uppercase[i], font=font(30), fill=(255, 220, 140))
        d.text((ox + 14, oy + cell + 4), L["name"], font=font(16), fill=(232, 231, 226))
        d.text((ox + 14, oy + cell + 26),
               f"{L['n']} units · {L['share'] * 100:.0f}% · {L['median_duration_s']:.2f}s",
               font=font(14, regular=True), fill=(160, 158, 150))
    im.save(FIG / f"alphabet_{tag}.png")
    print("wrote", FIG / f"alphabet_{tag}.png", im.size)


# --------------------------------------------------------------------------- remainder
def remainder(tag="overview", per_sheet=12):
    video = video_path()
    shots = pd.read_csv(OUT / "shots.csv")
    det = pd.read_csv(OUT / f"kp_{tag}.csv")
    frm = pd.read_csv(OUT / f"kp_{tag}_frames.csv")
    inter = np.zeros(int(shots["end"].max()) + 1, bool)
    for _, s in shots[shots["intertitle"]].iterrows():
        inter[int(s["start"]):int(s["end"]) + 1] = True
    frm["intertitle"] = inter[frm["frame"]]
    rng = np.random.default_rng(1)
    cats = {
        # nobody found (or barely) in footage that is not a title card
        "unseen": frm[(~frm["intertitle"]) & (frm["max_score"] < 0.3)],
        # someone found on a title card
        "phantom": frm[(frm["intertitle"]) & (frm["max_score"] >= 0.5)],
        # the model is unsure
        "doubt": frm[(frm["max_score"] >= 0.3) & (frm["max_score"] < 0.7)],
    }
    for name, rows in cats.items():
        if rows.empty:
            print(name, ": none")
            continue
        pick = rows.sample(min(per_sheet, len(rows)), random_state=1).sort_values("frame")
        wanted = set(pick["frame"].astype(int))
        tiles = []
        for idx, rgb in iter_frames(video, wanted=wanted):
            img = rgb.copy()
            for score, box, kp in dets_frame(det[det["frame"] == idx]):
                if score < 0.2:
                    continue
                img = draw_skeleton(img, kp, kp_thresh=0.5)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 204, 107), 1)
                cv2.putText(img, f"{score:.2f}", (int(box[0]) + 3, int(box[1]) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 204, 107), 1, cv2.LINE_AA)
            cv2.putText(img, f"frame {idx}", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (320, 240)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        cv2.imwrite(str(FIG / f"remainder_{name}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(name, len(rows), "frames ->", FIG / f"remainder_{name}.jpg")


def one_frame(frame, tag="overview", name=None):
    video = video_path()
    det = pd.read_csv(OUT / f"kp_{tag}.csv")
    rgb = read_frame(video, frame)
    img = rgb.copy()
    for score, box, kp in dets_frame(det[det["frame"] == frame]):
        if score >= 0.5:
            img = draw_skeleton(img, kp, kp_thresh=0.5)
    out = FIG / f"{name or f'skeleton_{frame:06d}'}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate([rgb, img], 1), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--hands", default="left,right")
    ap.add_argument("--name")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()
    if a.what == "contact":
        contact()
    elif a.what == "cyclegraph":
        cyclegraph(int(a.args[0]), int(a.args[1]), hands=tuple(a.hands.split(",")), name=a.name,
                   tag=a.tag or "dense", animate=not a.no_anim)
    elif a.what == "alphabet":
        alphabet(a.args[0] if a.args else "dense")
    elif a.what == "remainder":
        remainder(a.args[0] if a.args else "overview")
    elif a.what == "frame":
        one_frame(int(a.args[0]), tag=a.tag or "overview", name=a.name)


# --------------------------------------------------------------------------- top-detection trails
def top_tracks(tag, shots, kp_thr=0.5, max_gap=3):
    """Wrist paths of the most confident person in each sampled frame (one-worker
    shots), smoothed, in the same format as tracks_<tag>.csv (track = 0)."""
    from scipy.signal import savgol_filter
    parts = sorted(OUT.glob(f"kp_{tag}?.csv")) if not (OUT / f"kp_{tag}.csv").exists() else [OUT / f"kp_{tag}.csv"]
    det = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["frame", "det"])
    frm = pd.concat([pd.read_csv(str(p)[:-4] + "_frames.csv") for p in parts]).drop_duplicates("frame")
    tab = pd.read_csv(OUT / "shots.csv").set_index("shot")
    rows = []
    for shot in shots:
        a, b = tab.loc[shot, "start"], tab.loc[shot, "end"]
        grid = frm[(frm["frame"] >= a) & (frm["frame"] <= b)][["frame"]].sort_values("frame")
        top = det[(det["frame"] >= a) & (det["frame"] <= b) & (det["score"] >= 0.3)]
        top = top.sort_values("score").groupby("frame").tail(1)
        tr = grid.merge(top, on="frame", how="left")
        for j, hand in ((9, "left"), (10, "right")):
            ok = tr[f"s{j}"].fillna(-99) > kp_thr
            x = tr[f"x{j}"].where(ok).interpolate(limit=max_gap, limit_area="inside")
            y = tr[f"y{j}"].where(ok).interpolate(limit=max_gap, limit_area="inside")
            sx, sy = x.copy(), y.copy()
            valid = x.notna().to_numpy()
            idx = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
            for s0, s1 in zip(idx[::2], idx[1::2]):
                if s1 - s0 >= 5:
                    w = min(7, (s1 - s0) if (s1 - s0) % 2 else (s1 - s0) - 1)
                    sx.iloc[s0:s1] = savgol_filter(x.iloc[s0:s1], w, 2)
                    sy.iloc[s0:s1] = savgol_filter(y.iloc[s0:s1], w, 2)
            for f, xx, yy in zip(tr["frame"], sx, sy):
                rows.append((shot, 0, hand, int(f), xx, yy))
    out = pd.DataFrame(rows, columns=["shot", "track", "hand", "frame", "sx", "sy"])
    out.to_csv(OUT / f"tracks_top_{tag}.csv", index=False)
    return out
