import os, time, numpy as np, joblib
import slip_features as SF
pkl = os.path.expanduser("~/grasp_dataset/slip_model_v4.pkl")
out = os.path.expanduser("~/grasp_dataset/slip_model_v4_np.npz")
M = joblib.load(pkl); m = M["model"]
preds = [it[0] for it in m._predictors]
n = max(len(p.nodes) for p in preds); T = len(preds)
feat = np.zeros((T, n), np.int64); thr = np.zeros((T, n)); left = np.zeros((T, n), np.int64)
right = np.zeros((T, n), np.int64); leaf = np.ones((T, n), np.uint8); value = np.zeros((T, n))
for i, p in enumerate(preds):
    nd = p.nodes; k = len(nd)
    feat[i, :k] = nd["feature_idx"]; thr[i, :k] = nd["num_threshold"]
    left[i, :k] = nd["left"]; right[i, :k] = nd["right"]
    leaf[i, :k] = nd["is_leaf"]; value[i, :k] = nd["value"]
depth = int(m.max_depth or 30); base = float(np.ravel(m._baseline_prediction)[0])
np.savez(out, feat=feat, thr=thr, left=left, right=right, leaf=leaf, value=value, base=base, depth=depth,
         threshold=M["threshold"], sustain_s=M["sustain_s"], dt=M["dt"], feats=np.array(M["feats"]))
G = SF.NumpyGBM(out)
X = np.random.default_rng(0).normal(0, 3, (2000, len(M["feats"])))
a = m.predict_proba(X)[:, 1]; b = np.array([G.predict_p(x) for x in X])
t = time.time(); [G.predict_p(x) for x in X[:500]]; ms = (time.time() - t) / 500 * 1000
print(f"saved {out}: {T} trees, max diff {np.abs(a - b).max():.2e}, {ms:.3f} ms per prediction")
