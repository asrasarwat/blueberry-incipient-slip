import os, sys, json, glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/grasp_data_regrasp/blueberry/trial_20261106_142738")
I = json.load(open(f"{T}/trial_info.json")); RG = I["regrasp"]
fire, cmd, rest = float(RG["fire"]["fired_elapsed_s"]), float(RG["command_elapsed_s"]), float(RG["grip_restored_s"])
ev = {}
for l in open(f"{T}/events.csv"):
    parts = l.strip().split(",")
    if len(parts) >= 3:
        try: ev[parts[0]] = float(parts[2])
        except ValueError: pass
loosen = ev.get("loosen", fire - 10)
print(f"trial {os.path.basename(T)}: berry {I.get('berry_id')} {I.get('diameter_mm')} mm, trigger {RG.get('trigger')}, "
      f"loosen {loosen:.2f}, fire {fire:.2f}, command +{(cmd - fire) * 1000:.0f} ms, restored {rest:.2f} s, result {RG.get('result')}")
det = pd.read_csv(f"{T}/regrasp_detector.csv")
mod = pd.read_csv(f"{T}/regrasp_model.csv") if os.path.exists(f"{T}/regrasp_model.csv") else None

ts = pd.read_csv(f"{T}/camera_cam2_timestamps.csv")
tcol = next(c for c in ts.columns if "elapsed" in c.lower() or c.lower() in ("t", "t_s", "time_s"))
fcol = next((c for c in ts.columns if "frame" in c.lower() or "idx" in c.lower() or "index" in c.lower()), None)
print(f"camera timestamps: time column '{tcol}', frame column '{fcol}'")
pngs = {int(re.findall(r"(\d+)", os.path.basename(p))[-1]): p for p in glob.glob(f"{T}/camera_cam2/*.png")}
fr = ts[fcol].astype(int).values if fcol else np.arange(len(ts))
ft = ts[tcol].astype(float).values
have = np.array([n in pngs for n in fr])
def frame_at(t):
    i = np.where(have)[0][np.argmin(np.abs(ft[have] - t))]
    return pngs[fr[i]], ft[i]
roi = json.load(open(f"{T}/roi.json"))["berry"] if os.path.exists(f"{T}/roi.json") else None

snaps = [(loosen - 1.0, "before loosening"), (fire, "slip detected"), (fire + 0.5, "+0.5 s"), (fire + 5.0, "+5 s, held")]
fig = plt.figure(figsize=(7.16, 5.2))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25])
a = fig.add_subplot(gs[0, :])
if mod is not None:
    a.plot(mod.elapsed_s, mod.p_slip, lw=1, color="k", label="predicted slip probability")
    a.axhline(float(mod.threshold.iloc[0]), color="tab:red", ls=":", lw=1, label=f"threshold {float(mod.threshold.iloc[0]):.3f}")
    a.set_ylabel("p(slip)", fontsize=8)
a.axvline(fire, color="tab:red", lw=1); a.axvline(loosen, color="0.4", ls="--", lw=1)
a.legend(fontsize=6.5, loc="upper left"); a.set_title("(a) learned detector", fontsize=8)
b = fig.add_subplot(gs[1, :], sharex=a)
for f, g in det.groupby("finger"):
    b.plot(g.elapsed_s, g.fz_sum_N, lw=1, label=f"{f} normal force")
b.axvline(loosen, color="0.4", ls="--", lw=1, label="loosening starts")
b.axvline(fire, color="tab:red", lw=1, label=f"alarm (command +{(cmd - fire) * 1000:.0f} ms)")
b.axvspan(fire, fire + rest, color="tab:green", alpha=0.2, label=f"grip restored in {rest:.2f} s")
b.set_ylabel("force (N)", fontsize=8); b.set_xlabel("time (s)", fontsize=8)
b.legend(fontsize=6.5, ncol=3, loc="upper left", framealpha=0.9); b.set_title("(b) fingertip normal force", fontsize=8)
b.set_xlim(loosen - 3, fire + 6)
for ax in (a, b): ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
plt.setp(a.get_xticklabels(), visible=False)
for j, (t, lab) in enumerate(snaps):
    ax = fig.add_subplot(gs[2, j]); p, tt = frame_at(t)
    im = plt.imread(p)
    if roi:
        x, y, w, h = roi; pad = int(0.6 * max(w, h))
        im = im[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    im = im[..., :3] if im.ndim == 3 else im
    lo, hi = np.percentile(im, [1, 99.5]); im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    ax.imshow(im, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"({'cdef'[j]}) {lab}\nt = {tt:.1f} s", fontsize=7)
fig.tight_layout(h_pad=0.4)
out = os.path.expanduser("~/Desktop/fig_closed_loop")
for e in ("png", "pdf"): fig.savefig(f"{out}.{e}", dpi=300)
print(f"saved {out}.png/.pdf")
