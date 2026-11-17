import os, json, numpy as np, pandas as pd
src = open("dong_baseline.py").read(); exec(src[:src.index("G = pd.read_csv")])   # reuse your functions only
H = os.path.expanduser("~"); G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
cache = []
for s in C.SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json")); hold, loosen, drop = float(ev["hold"]), float(ev["loosen"]), float(r.drop_camera_s)
        z = np.load(f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}/taxels.npz")
        ser = [x for x in (finger(z, k, key, hold, loosen) for k, key in enumerate(("index", "thumb"))) if x is not None]
        cam = G.loc[r.folder, "cam_0.3mm"] if r.folder in G.index else np.nan
        if ser: cache.append((ser, hold, loosen, drop, cam))
    print("cached", s, flush=True)
def on(ser, b0, b1, t0, t1, K, NM):
    best = None
    for t, rr in ser:
        b = (t >= b0) & (t < b1)
        if b.sum() < 10: continue
        thr = rr[b].mean() + K * max(rr[b].std(), 1e-6)
        o = C.first_sustained(t, (rr > thr).sum(1) > NM, t0, t1, 0.25)
        if o is not None and (best is None or o < best): best = o
    return best
print(f"{'K':>4} {'N':>3} | detected | hold FA | before gross slip (onset) | median lead")
for K in (2, 3, 4, 5, 6, 8, 10):
    for NM in (1, 2, 3, 5):
        det = fa = 0; leads = []
        for ser, hold, loosen, drop, cam in cache:
            o = on(ser, hold, loosen, loosen, drop, K, NM); mid = (hold + loosen) / 2
            fa += on(ser, hold, mid, mid, loosen, K, NM) is not None
            if o is not None:
                det += 1
                if pd.notna(cam): leads.append(cam - 0.125 - o)
        x = np.array(leads)
        print(f"{K:4.0f} {NM:3d} | {det:2d}/{len(cache)}    | {fa:2d}/{len(cache)}   | {int((x >= 0).sum()):2d}/{len(x):2d}                     | {np.median(x):+.2f} s")
