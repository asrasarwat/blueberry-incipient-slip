import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv"); G["size"] = G["size"].map(norm); G = G.set_index("trial")
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    for _, r in tm[tm.status == "valid"].iterrows():
        T = f"{H}/grasp_data/blueberry/size_{s}mm/{r.folder}"
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json")); hold, loosen = float(ev["hold"]), float(ev["loosen"])
        info = json.load(open(f"{T}/trial_info.json")); rate = float(info["loosen_rate_rad_s"])
        z = np.load(f"{T}/taxels.npz"); tg = np.arange(hold, float(r.drop_camera_s), 0.02); F = np.zeros_like(tg)
        for k, key in enumerate(("index", "thumb")):
            m = (z["new"][:, k] == 1) & (z["stale"][:, k] == 0)
            F += np.interp(tg, z["t"][m], np.abs(z[key][m][..., 2].astype(float)).sum(axis=(1, 2)))
        onset = G.loc[r.folder, "stick_onset"] if r.folder in G.index else np.nan
        t1 = onset if np.isfinite(onset) and onset - loosen > 1.5 else loosen + 2.5
        w = (tg >= loosen + 0.5) & (tg <= t1)
        if w.sum() < 20: continue
        theta = rate * (tg[w] - loosen); slope = np.polyfit(theta, F[w], 1)[0]
        hold_F = F[(tg >= hold) & (tg < loosen)].mean()
        lead = -G.loc[r.folder, "stick_minus_cam_0.3mm"] if r.folder in G.index else np.nan
        rows.append(dict(trial=r.folder, size=s, rate=rate, k_N_per_rad=-slope, k_rel=-slope / hold_F, hold_F=hold_F,
                         window_s=t1 - loosen - 0.5, lead=lead))
D = pd.DataFrame(rows); D.to_csv(f"{H}/claim1_results/stiffness_proxy.csv", index=False)
print(f"{len(D)} trials\n\nper size (median):")
print(D.groupby("size")[["k_N_per_rad", "k_rel", "hold_F", "window_s"]].median().round(2).to_string())
print(f"\nk across sizes: Kruskal-Wallis p = {kruskal(*[g.k_N_per_rad for _, g in D.groupby('size')]).pvalue:.4f}")
rho, p = spearmanr(D["size"].astype(float), D.k_N_per_rad); print(f"k vs size: Spearman rho = {rho:+.2f}, p = {p:.4f}")
big = D["size"].isin(["20.0", "21.0"])
print(f"k, 20-21 mm vs 14-18 mm: median {D[big].k_N_per_rad.median():.2f} vs {D[~big].k_N_per_rad.median():.2f} N/rad, "
      f"Mann-Whitney p = {mannwhitneyu(D[big].k_N_per_rad, D[~big].k_N_per_rad).pvalue:.4f}")
q = D.dropna(subset=["lead"]); rho, p = spearmanr(q.k_N_per_rad, q.lead); print(f"k vs stick-ratio lead: Spearman rho = {rho:+.2f}, p = {p:.4f} (n = {len(q)})")
rho, p = spearmanr(q.k_rel, q.lead); print(f"k/hold force vs lead: Spearman rho = {rho:+.2f}, p = {p:.4f}")
rho, p = spearmanr(D.rate, D.k_N_per_rad); print(f"check, k vs release rate: Spearman rho = {rho:+.2f}, p = {p:.4f} (should be ~0)")
