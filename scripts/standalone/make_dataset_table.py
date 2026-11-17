import os, json, pandas as pd, numpy as np
H = os.path.expanduser("~"); SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
FA = pd.read_csv(f"{H}/claim1_results/failure_analysis.csv").set_index("trial")
rows = []
for s in SIZES:
    tm = pd.read_csv(f"{H}/grasp_data/blueberry/size_{s}mm/trial_map.csv")
    v = tm[tm.status == "valid"]
    rel = []
    for _, r in v.iterrows():
        ev = json.load(open(f"{H}/shear_{s}mm/{r.folder}_events.json"))
        rel.append(float(r.drop_camera_s) - float(ev["loosen"]))
    hf = FA.reindex(v.folder).hold_force.dropna()
    rows.append([s.replace(".0", ""), v.berry_id.nunique(), len(v), int((tm.status == "flagged").sum()),
                 f"{hf.median():.1f}", f"{np.median(rel):.1f}", f"{min(rel):.1f}--{max(rel):.1f}"])
T = pd.DataFrame(rows, columns=["Size (mm)", "Berries", "Valid", "Flagged", "Hold force (N)", "Release to drop (s)", "Range (s)"])
tot = ["All", int(T.Berries.sum()), int(T.Valid.sum()), int(T.Flagged.sum()), "", "", ""]
print(T.to_string(index=False))
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Dataset. Hold force is the median summed normal force of both fingertips during the hold; release-to-drop is the time from the start of the release to the camera-confirmed drop.}",
       r"\label{tab:dataset}", r"\begin{tabular}{lcccccc}", r"\toprule", " & ".join(T.columns) + r" \\", r"\midrule"]
tex += [" & ".join(str(x) for x in r) + r" \\" for r in rows]
tex += [r"\midrule", " & ".join(str(x) for x in tot) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{H}/paper_figures/table_dataset.tex", "w").write("\n".join(tex) + "\n"); print("saved ~/paper_figures/table_dataset.tex")
