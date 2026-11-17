import os, glob, argparse
import numpy as np, pandas as pd
import claim1_check as C
import analyze_slip as A
from gross_slip_check import rel_displacement

ROOT = "/media/dataset_storage/toy/blueberry"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=os.path.expanduser("~/track_pilot"))
    ap.add_argument("--px-per-mm", type=float, default=29.45)
    a = ap.parse_args()
    rows = []
    for size in ("size_15.7mm", "size_17.7mm"):
        for tr in sorted(glob.glob(f"{ROOT}/{size}/supported*_trial_*") + glob.glob(f"{ROOT}/{size}/release*_trial_*")):
            name = os.path.basename(tr)
            kind = "supported" if name.startswith("supported") else "release"
            ev = A.load_events(tr); hold, loosen = ev["hold"], ev["loosen"]
            end = ev.get("drop", ev.get("post", loosen + 60))
            z = np.load(f"{tr}/taxels.npz")
            ser = [x for x in (C.finger_series(z, k, key, hold) for k, key in enumerate(("index", "thumb"))) if x is not None]
            on = C.onsets(ser, hold, loosen, loosen, end)["min_stick_frac"] if ser else None
            row = dict(size=size[5:], kind=kind, trial=name, loosen=loosen, stick_after_loosen=None if on is None else on - loosen)
            f = f"{a.track}/{name}_track.csv"
            if os.path.exists(f) and on is not None:
                t, d, note = rel_displacement(pd.read_csv(f), loosen)
                if t is not None:
                    dmm = d / a.px_per_mm
                    hold_m = (t >= hold) & (t < loosen)
                    i = np.argmin(np.abs(t - on))
                    win = (t >= loosen) & (t <= on + 1.0)
                    row.update(hold_noise_mm=np.nanpercentile(dmm[hold_m], 95) if hold_m.any() else np.nan,
                               disp_at_stick_mm=dmm[i], max_disp_to_stick_plus1s_mm=np.nanmax(dmm[win]) if win.any() else np.nan)
            else:
                row["note"] = "no track" if on is not None else "no stick onset"
            rows.append(row)
            if on is not None:
                print(f"contact sheet: python3 contact_sheet.py {tr} {on - 2:.1f} {on + 3:.1f}")
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + D.round(3).to_string(index=False))
    if "disp_at_stick_mm" in D:
        print("\nmedian per group (mm):")
        print(D.groupby(["size", "kind"])[["hold_noise_mm", "disp_at_stick_mm", "max_disp_to_stick_plus1s_mm"]]
              .median().round(3).to_string())
    os.makedirs(os.path.expanduser("~/claim1_results"), exist_ok=True)
    D.to_csv(os.path.expanduser("~/claim1_results/supported_vs_release_motion.csv"), index=False)


if __name__ == "__main__":
    main()
