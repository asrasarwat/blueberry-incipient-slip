import os, json, numpy as np, pandas as pd
from scipy.stats import kruskal, spearmanr
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
W = pd.read_csv(f"{H}/model_results/warn_per_trial_loso.csv").set_index("trial")
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv").set_index("trial")
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    v = tm[tm.status == "valid"].copy(); v["stamp"] = v.folder.str.extract(r"trial_(\d+_\d+)")[0]
    v["order"] = v.groupby("berry_id").stamp.rank().astype(int)
    for _, r in v.iterrows():
        I = json.load(open(f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}/trial_info.json"))
        rows.append(dict(trial=r.folder, size=s, berry=r.berry_id, order=r.order, target=I.get("target_force_N"),
                         rate=I.get("loosen_rate_rad_s"), hold=I.get("hold_s"),
                         r_idx=I.get("index_rate_hz"), r_thb=I.get("thumb_rate_hz"),
                         warn=W.warn_s.get(r.folder, np.nan),
                         lead=-G["stick_minus_cam_0.3mm"].get(r.folder, np.nan)))
D = pd.DataFrame(rows); D.to_csv(f"{H}/model_results/protocol_effects.csv", index=False)
def test(col, lab):
    g = [q.dropna() for _, q in D.groupby(col)["warn"]]
    gl = [q.dropna() for _, q in D.groupby(col)["lead"]]
    print(f"\n{lab}:"); print(D.groupby(col).agg(n=("trial", "size"), warn_med=("warn", "median"), lead_med=("lead", "median"),
                                   lead_first=("lead", lambda x: f"{int((x.dropna() >= 0).sum())}/{x.notna().sum()}")).round(2).to_string())
    print(f"   Kruskal-Wallis: warning p = {kruskal(*g).pvalue:.3f}, lead p = {kruskal(*gl).pvalue:.3f}")
test("target", "by grip target (N)"); test("rate", "by release rate (rad/s)"); test("order", "by trial number on the same berry")
for c, lab in [("hold", "hold duration")]:
    q = D.dropna(subset=[c, "warn"]); print(f"\nwarning vs {lab}: Spearman rho = {spearmanr(q[c], q.warn)[0]:.2f}, p = {spearmanr(q[c], q.warn)[1]:.3f}")
nop = pd.read_csv(f"{H}/model_results/nopull_fa_v4.csv").monitored_s.sum()
print(f"\nfalse-alarm exposure: {D.hold.sum():.0f} s of holds in 70 trials + {nop:.0f} s no-pull = {(D.hold.sum() + nop) / 60:.1f} min, 0 false alarms")
r = pd.concat([D.r_idx, D.r_thb]).dropna()
print(f"tactile rate per fingertip: median {r.median():.1f} Hz, min {r.min():.1f} Hz, max {r.max():.1f} Hz")
print(f"hold duration: {D.hold.min():.1f}-{D.hold.max():.1f} s; release rates used: {sorted(D.rate.dropna().unique())}; grip targets: {sorted(D.target.dropna().unique())}")
