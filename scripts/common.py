"""Shared helpers for the Therbligs pilot.

Models come from the Distant Viewing Toolkit (dvt 1.1.0):
  * shot boundaries: dvt.AnnoShotBreaks (TransNetV2)
  * body keypoints:  the toolkit's Keypoint R-CNN weights (dvt_detect_keypoint.pt),
                     loaded into torchvision's KeypointRCNN so the input size and
                     proposal counts can be set (same weights, ~12x faster on CPU
                     than the toolkit's traced model, keypoints within 1 px).
  * image embeddings: dvt.AnnoEmbed (EfficientNet)
"""
import os
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("THERBLIGS_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]
L_WRIST, R_WRIST = 9, 10


def video_path():
    """The film: first .mp4/.mpeg/.ogv in data/."""
    for ext in ("*.mp4", "*.m4v", "*.mov", "*.mpeg", "*.mpg", "*.ogv", "*.webm"):
        found = sorted(DATA.glob(ext))
        if found:
            return found[0]
    raise FileNotFoundError("Put the film in " + str(DATA))


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    meta = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    meta["seconds"] = meta["frames"] / meta["fps"] if meta["fps"] else None
    return meta


def iter_frames(path, wanted=None):
    """Yield (frame_index, rgb_image) sequentially; if `wanted` (a set) is
    given, only those frames are decoded to RGB (others are just grabbed)."""
    cap = cv2.VideoCapture(str(path))
    idx = 0
    last = max(wanted) if wanted else None
    while True:
        if wanted is not None and idx not in wanted:
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if last is not None and idx > last:
                break
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        idx += 1
        if last is not None and idx > last:
            break
    cap.release()


def read_frame(path, index):
    """Read one frame (RGB) by sequential decoding (accurate for H.264)."""
    for i, img in iter_frames(path, wanted={index}):
        return img
    return None


def load_keypoint_model(min_size=600):
    """The Distant Viewing Toolkit's Keypoint R-CNN weights in torchvision."""
    import torch
    import torchvision  # noqa: F401  (registers torchvision ops for the traced file)
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from dvt.utils import _download_file

    torch.set_num_threads(int(os.environ.get("THREADS", max(1, os.cpu_count() or 1))))
    traced = torch.jit.load(_download_file("dvt_detect_keypoint.pt"))
    state = {k[len("model."):]: v for k, v in traced.state_dict().items()}
    del traced
    model = keypointrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=min_size, max_size=int(round(min_size * 4 / 3)),
        rpn_pre_nms_top_n_test=300, rpn_post_nms_top_n_test=300,
        box_detections_per_img=10, box_score_thresh=0.2,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def run_keypoints(model, rgb):
    """Return a list of detections: dict(score, box[4], kp[17,3] = x, y, kp_score)."""
    import torch
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    with torch.inference_mode():
        out = model([x])[0]
    dets = []
    for i in range(len(out["scores"])):
        kp = out["keypoints"][i].numpy().copy()
        kp[:, 2] = out["keypoints_scores"][i].numpy()
        dets.append({
            "score": float(out["scores"][i]),
            "box": out["boxes"][i].numpy().tolist(),
            "kp": kp,
        })
    return dets


def draw_skeleton(rgb, kp, color=(255, 196, 64), kp_thresh=1.0, thickness=2):
    """Draw a COCO skeleton on a copy of an RGB image."""
    img = rgb.copy()
    for a, b in SKELETON:
        if kp[a, 2] > kp_thresh and kp[b, 2] > kp_thresh:
            cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     color, thickness, cv2.LINE_AA)
    for j in range(17):
        if kp[j, 2] > kp_thresh:
            r = 4 if j in (L_WRIST, R_WRIST) else 3
            c = (255, 255, 255) if j in (L_WRIST, R_WRIST) else color
            cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), r, c, -1, cv2.LINE_AA)
    return img


def dets_frame(df_frame):
    """Rows of the detections table for one frame -> list of (score, box, kp)."""
    out = []
    for _, r in df_frame.iterrows():
        kp = np.array([[r[f"x{j}"], r[f"y{j}"], r[f"s{j}"]] for j in range(17)], dtype=float)
        out.append((r["score"], [r["bx0"], r["by0"], r["bx1"], r["by1"]], kp))
    return out
