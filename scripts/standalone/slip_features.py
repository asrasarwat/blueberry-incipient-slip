#!/usr/bin/env python3
"""slip_features.py - shared feature definition for training and the live detector."""
import bisect
from collections import deque
import numpy as np

REF_WIN = 1.0
BASE = 3.0
RATE_WIN = 0.5
SMOOTH_N = 5
CONTACT_FRAC, CENTRE_FRAC, EDGE_FRAC = 0.10, 0.5, 0.7
FINGERS = ("index", "thumb")
SIGNALS = ("trans", "edge_lag", "fz", "edge_tr", "ctr_tr")

def feature_names():
    out = []
    for f in FINGERS:
        for k in SIGNALS:
            if k == "fz":
                out += [f"{f}_fz_rel", f"{f}_fz_rate"]
            else:
                out += [f"{f}_{k}_z", f"{f}_{k}_d", f"{f}_{k}_rate"]
    return out + ["fz_rel_sum"]

FEATS = feature_names()

def signals_from_row(r):
    return {"trans": np.hypot(r["all_dx"], r["all_dy"]), "edge_lag": r["edge_lag_px"], "fz": r["fz_N"],
            "edge_tr": np.hypot(r["edge_dx"], r["edge_dy"]), "ctr_tr": np.hypot(r["ctr_dx"], r["ctr_dy"])}

def _resize_mask(mask, shape):
    H, W = shape
    h, w = mask.shape
    ri = np.clip((np.arange(H) * h / H).astype(int), 0, h - 1)
    ci = np.clip((np.arange(W) * w / W).astype(int), 0, w - 1)
    return mask[np.ix_(ri, ci)]

class ContactRef:
    def __init__(self):
        self.F, self.M, self.ok = [], [], False
        self.buf = deque(maxlen=SMOOTH_N)

    def add(self, F, M):
        self.F.append(np.asarray(F, float)); self.M.append(np.asarray(M, float))

    def finish(self):
        if len(self.F) < 3:
            return False
        fz_ref = np.abs(np.stack(self.F)[..., 2]).mean(0)
        self.Mref = np.stack(self.M).mean(0)
        cm = _resize_mask((fz_ref > CONTACT_FRAC * fz_ref.max()) & (fz_ref > 1e-3), self.Mref.shape[:2])
        if cm.sum() < 8:
            return False
        rr, cc = np.mgrid[0:cm.shape[0], 0:cm.shape[1]]
        w = cm.astype(float)
        r0, c0 = (rr * w).sum() / w.sum(), (cc * w).sum() / w.sum()
        d = np.hypot(rr - r0, cc - c0)
        req = np.sqrt(w.sum() / np.pi)
        self.cm, self.centre, self.edge = cm, cm & (d < CENTRE_FRAC * req), cm & (d >= EDGE_FRAC * req)
        self.F, self.M, self.ok = [], [], True
        return True

    def shear(self, F, M):
        self.buf.append(np.asarray(M, float) - self.Mref)
        D = np.mean(self.buf, axis=0)
        a = np.median(D[self.cm], axis=0)
        c = D[self.centre].mean(0) if self.centre.any() else np.array([np.nan, np.nan])
        e = D[self.edge].mean(0) if self.edge.any() else np.array([np.nan, np.nan])
        return {"all_dx": a[0], "all_dy": a[1], "ctr_dx": c[0], "ctr_dy": c[1], "edge_dx": e[0], "edge_dy": e[1],
                "edge_lag_px": float(np.hypot(*(c - e))), "fz_N": float(np.abs(np.asarray(F)[..., 2]).sum())}

class FeatureStream:
    def __init__(self, t0):
        self.t0 = t0
        self.t = {f: [] for f in FINGERS}
        self.v = {f: {k: [] for k in SIGNALS} for f in FINGERS}
        self.base = None

    def push(self, t, finger, row):
        s = signals_from_row(row)
        self.t[finger].append(t)
        for k in SIGNALS:
            self.v[finger][k].append(float(s[k]))

    def _at(self, f, k, t):
        i = bisect.bisect_right(self.t[f], t) - 1
        return np.nan if i < 0 else self.v[f][k][i]

    def _make_base(self):
        self.base = {}
        for f in FINGERS:
            tt = np.asarray(self.t[f])
            m = (tt >= self.t0) & (tt < self.t0 + BASE)
            for k in SIGNALS:
                x = np.asarray(self.v[f][k])[m]
                x = x[np.isfinite(x)]
                mu = x.mean() if len(x) else 0.0
                sd = max(x.std(), 1e-3) if len(x) > 1 else 1.0
                self.base[(f, k)] = (mu, sd)

    def ready(self, t):
        return t >= self.t0 + BASE and all(len(self.t[f]) > 5 for f in FINGERS)

    def vector(self, t):
        if not self.ready(t):
            return None
        if self.base is None:
            self._make_base()
        out = {}
        for f in FINGERS:
            for k in SIGNALS:
                g, g0 = self._at(f, k, t), self._at(f, k, t - RATE_WIN)
                mu, sd = self.base[(f, k)]
                rate = (g - g0) / RATE_WIN
                if k == "fz":
                    out[f"{f}_fz_rel"] = g / max(mu, 0.05)
                    out[f"{f}_fz_rate"] = rate
                else:
                    out[f"{f}_{k}_z"], out[f"{f}_{k}_d"], out[f"{f}_{k}_rate"] = (g - mu) / sd, g - mu, rate
        out["fz_rel_sum"] = (out["index_fz_rel"] + out["thumb_fz_rel"]) / 2
        x = np.array([out[n] for n in FEATS], float)
        x[~np.isfinite(x)] = 0.0
        return x


class NumpyGBM:
    def __init__(self, path):
        z = np.load(path)
        self.feat, self.thr = z["feat"], z["thr"]
        self.left, self.right = z["left"], z["right"]
        self.leaf, self.value = z["leaf"].astype(bool), z["value"]
        self.base = float(z["base"])
        self.depth = int(z["depth"])
        self.threshold, self.sustain_s, self.dt = float(z["threshold"]), float(z["sustain_s"]), float(z["dt"])
        self.ti = np.arange(self.feat.shape[0])

    def predict_p(self, x):
        x = np.asarray(x, float).ravel()
        node = np.zeros(len(self.ti), dtype=np.int64)
        for _ in range(self.depth + 1):
            lf = self.leaf[self.ti, node]
            go_left = x[self.feat[self.ti, node]] <= self.thr[self.ti, node]
            nxt = np.where(go_left, self.left[self.ti, node], self.right[self.ti, node])
            node = np.where(lf, node, nxt)
        raw = self.base + self.value[self.ti, node].sum()
        return float(1.0 / (1.0 + np.exp(-raw)))


class NumpyGBM:
    def __init__(self, path):
        z = np.load(path)
        self.feat, self.thr = z["feat"], z["thr"]
        self.left, self.right = z["left"], z["right"]
        self.leaf, self.value = z["leaf"].astype(bool), z["value"]
        self.base = float(z["base"])
        self.depth = int(z["depth"])
        self.threshold, self.sustain_s, self.dt = float(z["threshold"]), float(z["sustain_s"]), float(z["dt"])
        self.ti = np.arange(self.feat.shape[0])

    def predict_p(self, x):
        x = np.asarray(x, float).ravel()
        node = np.zeros(len(self.ti), dtype=np.int64)
        for _ in range(self.depth + 1):
            lf = self.leaf[self.ti, node]
            go_left = x[self.feat[self.ti, node]] <= self.thr[self.ti, node]
            nxt = np.where(go_left, self.left[self.ti, node], self.right[self.ti, node])
            node = np.where(lf, node, nxt)
        raw = self.base + self.value[self.ti, node].sum()
        return float(1.0 / (1.0 + np.exp(-raw)))
