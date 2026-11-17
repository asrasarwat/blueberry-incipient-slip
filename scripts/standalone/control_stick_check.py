import os, glob, numpy as np
import claim1_check as C
import analyze_slip as A
ROOT = "/media/dataset_storage/toy/blueberry"
for size in ("size_15.7mm", "size_17.7mm"):
    for tr in sorted(glob.glob(f"{ROOT}/{size}/supported*_trial_*") + glob.glob(f"{ROOT}/{size}/release*_trial_*")):
        name = os.path.basename(tr)
        try:
            ev = A.load_events(tr); hold, loosen = ev["hold"], ev["loosen"]
            end = ev.get("drop", ev.get("post", loosen + 60))
            z = np.load(f"{tr}/taxels.npz")
            ser = [x for x in (C.finger_series(z, k, key, hold) for k, key in enumerate(("index", "thumb"))) if x is not None]
            if not ser:
                print(f"{size} {name:45s} no contact"); continue
            on = C.onsets(ser, hold, loosen, loosen, end)["min_stick_frac"]
            mid = (hold + loosen) / 2
            fa = C.onsets(ser, hold, mid, mid, loosen)["min_stick_frac"]
            print(f"{size} {name:45s} loosen {loosen:6.1f}  end {end:6.1f}  stick onset "
                  f"{'NONE' if on is None else f'{on - loosen:+.1f} s after loosen'}   hold FA {fa is not None}")
        except Exception as e:
            print(f"{size} {name:45s} FAILED {type(e).__name__}: {e}")
