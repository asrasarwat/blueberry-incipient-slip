import os
import pandas as pd, numpy as np, os
H = os.path.expanduser("~"); norm = lambda v: f"{float(v):.1f}"
A = pd.read_csv(f"{H}/claim1_results/claim1_per_trial.csv");          A["size"] = A["size"].map(norm)
G = pd.read_csv(f"{H}/claim1_results/gross_slip_per_trial.csv");       G["size"] = G["size"].map(norm)
B = pd.read_csv(f"{H}/claim1_results/dong_baseline_per_trial.csv");    B["size"] = B["size"].map(norm)
E = pd.read_csv(f"{H}/claim1_results/entropy_baseline_per_trial.csv"); E["size"] = E["size"].map(norm)
SIZES = ["14.0", "15.0", "16.0", "17.0", "18.0", "20.0", "21.0"]
def grp(det, n, fa, x):
    x = x.dropna(); return [f"{det}/{n}", int(fa), f"{int((x <= 0).sum())}/{len(x)}", f"{-x.median():+.2f}"]
def stats(a, g, b, e):
    return [len(a)] + grp(a.min_stick_frac_onset.notna().sum(), len(a), a.min_stick_frac_hold_FA.sum(), g["stick_minus_cam_0.3mm"]) \
                    + grp(e.onset.notna().sum(), len(e), e.hold_FA.sum(), e.minus_cam03) \
                    + grp(b.onset.notna().sum(), len(b), b.hold_FA.sum(), b.minus_cam03)
rows = [[s.replace(".0", "")] + stats(*(X[X["size"] == s] for X in (A, G, B, E))) for s in SIZES]
rows.append([r"\textbf{All}"] + stats(A, G, B, E))
sub = ["Det.", "FA", "Before GS", "Lead (s)"]
cols = ["Size (mm)", "n"] + sub * 3
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
tex = [r"\begin{table*}[t]", r"\centering",
       r"\caption{Per-size onset results for three marker-field measures under the same onset rule. Det.: drops detected; FA: false alarms during stable holding; Before GS: onsets preceding 0.3~mm camera-measured gross slip; Lead: median lead over gross slip (positive = tactile first). The stick ratio was pre-registered as the primary measure.}",
       r"\label{tab:per_size}", r"\setlength{\tabcolsep}{3pt}", r"\small",
       r"\begin{tabular}{lc cccc cccc cccc}", r"\toprule",
       r" & & \multicolumn{4}{c}{Stick ratio (pre-registered)} & \multicolumn{4}{c}{Displacement entropy~\cite{yuan2015}} & \multicolumn{4}{c}{Rigid-body residual~\cite{dong2019}} \\",
       r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
       " & ".join(cols) + r" \\", r"\midrule"]
for i, r in enumerate(rows):
    if i == len(rows) - 1: tex.append(r"\midrule")
    tex.append(" & ".join(str(v) for v in r) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{H}/paper_figures/table_per_size.tex", "w").write("\n".join(tex) + "\n")
print("\nsaved ~/paper_figures/table_per_size.tex")
