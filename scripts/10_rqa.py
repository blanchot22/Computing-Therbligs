"""Step 10: determinism — a steadier measure of repetition (recurrence analysis).

From each shot's self-similarity matrix (06_rhythm.py): two moments 'recur' if
their upper-body postures differ by at most EPS shoulder-widths (mean over
shoulders, elbows, wrists), ignoring moments closer than EXCL samples in time.
Determinism = share of recurring pairs that lie on diagonal lines at least LMIN
samples long, i.e. that come back as whole repeated sequences.

  python 10_rqa.py      -> out/rqa.csv
"""
import numpy as np
import pandas as pd

from common import OUT

EPS, LMIN, EXCL = 0.3, 6, 6        # 0.3 shoulder-widths; 6 samples = 0.4 s of film


def rqa(D):
    n = len(D)
    R = (D <= EPS) & ~np.isnan(D)
    for k in range(-EXCL, EXCL + 1):
        idx = np.arange(max(0, -k), min(n, n - k))
        R[idx, idx + k] = False
    rec = R.sum()
    det = 0
    for k in range(EXCL + 1, n):
        d = np.diagonal(R, k).astype(int)
        edges = np.diff(np.r_[0, d, 0])
        L = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        det += 2 * L[L >= LMIN].sum()
    valid = (~np.isnan(D)).sum()
    return rec / valid if valid else np.nan, det / rec if rec else np.nan


r = pd.read_csv(OUT / "rhythm_summary.csv")
rows = []
for _, s in r.iterrows():
    D = np.load(OUT / f"rhythm_{int(s['shot']):03d}_D.npy")
    rr, det = rqa(D)
    rows.append({"shot": int(s["shot"]), "task": s["task"], "method": s["method"],
                 "samples": int(s["track_samples"]), "recurrence_rate": rr, "determinism": det})
out = pd.DataFrame(rows)
out.to_csv(OUT / "rqa.csv", index=False)
print(out.round(3).to_string(index=False))
