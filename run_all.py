"""Run the complete reproducible pipeline in order and write final headline files."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from config import *

STAGES=["01_audit.py","02_clean_features.py","03_metrics.py","04_models.py","05_visuals.py"]

def run_stage(script):
    print(f"\n{'='*72}\nRUNNING {script}\n{'='*72}",flush=True)
    subprocess.run([sys.executable,str(PROJECT_DIR/script)],cwd=PROJECT_DIR,check=True)

def finalize():
    pm=pd.read_parquet(TABLE_DIR/"play_metrics.parquet")
    ab=pd.read_csv(TABLE_DIR/"pressure_ablation.csv"); allrow=ab[ab.variant.eq("all_new")].iloc[0]
    mixed=pd.read_csv(TABLE_DIR/"disguise_drag_mixed_effects.csv")
    ris=mixed[(mixed.outcome.eq("reactionLag"))&mixed.term.eq("RIS")]
    # Game bootstrap for pressure-vs-no-pressure UPT difference.
    rng=np.random.default_rng(RANDOM_SEED); games=pm.gameId.unique(); diffs=[]
    for _ in range(BOOTSTRAP_REPS):
        chosen=rng.choice(games,len(games),replace=True); b=pd.concat([pm[pm.gameId.eq(g)] for g in chosen],ignore_index=True)
        diffs.append(b.loc[b.pressure.eq(1),"UPT"].mean()-b.loc[b.pressure.eq(0),"UPT"].mean())
    dlo,dhi=np.quantile(diffs,[.025,.975]); d=np.mean(diffs)
    rows=[
      {"result":"Pressure vs no-pressure UPT difference (s)","estimate":d,"ci_low":dlo,"ci_high":dhi},
      {"result":"All-new-metrics pressure AUC gain","estimate":allrow.AUC_gain,"ci_low":allrow.AUC_gain_low,"ci_high":allrow.AUC_gain_high},
      {"result":"All-new-metrics log-loss gain","estimate":allrow.logLoss_gain,"ci_low":allrow.logLoss_gain_low,"ci_high":allrow.logLoss_gain_high}]
    if len(ris): rows.append({"result":"Adjusted RIS coefficient on observed RL (s/unit)","estimate":ris.iloc[0].coefficient,"ci_low":ris.iloc[0].low,"ci_high":ris.iloc[0].high})
    pd.DataFrame(rows).to_csv(TABLE_DIR/"headline_results.csv",index=False)
    mixed_text=(f"; adjusted RIS–RL coefficient {ris.iloc[0].coefficient:.3f} s/unit (95% CI {ris.iloc[0].low:.3f} to {ris.iloc[0].high:.3f})" if len(ris) else "; the mixed-effects fit was unstable and is reported as a limitation")
    readme=[
      "When a pass rush approaches from outside the quarterback's body-facing view cone, how long does the quarterback take to turn toward it and move, does a defense's disguise (rushers who looked like non-rushers before the snap, or who reveal their rush late) make that lag longer, and does the lag explain hurries, hits, and sacks?",
      f"We analyzed {len(pm):,} passing plays from {pm.gameId.nunique()} games using 10-Hz tracking, PFF roles, game-grouped validation, censored survival analysis, and a Weeks 1–5 to Weeks 6–8 time split.",
      "New metrics: Unseen Pressure Time (UPT), Reaction Lag (RL), Pocket Movement Over Expected (PMOE), Rush Intent Surprise (RIS), Rush Reveal Time (RRT), and Disguise Drag.",
      f"Headline results: pressured plays had {d:.3f}s more UPT (95% CI {dlo:.3f} to {dhi:.3f}); all new metrics changed pressure AUC by {allrow.AUC_gain:+.3f} (95% CI {allrow.AUC_gain_low:+.3f} to {allrow.AUC_gain_high:+.3f}){mixed_text}.",
      "Run with python run_all.py; main limitation: shoulder-pad body-facing orientation is not eye gaze, and observational associations cannot establish that disguise caused the outcome."
    ]
    (PROJECT_DIR/"README.md").write_text("\n".join(readme)+"\n",encoding="utf-8")
    assert len((PROJECT_DIR/"README.md").read_text(encoding="utf-8").splitlines())==5
    print("\n[FINALIZATION COMPLETE]\nSaved real-number headline results and verified README.md has exactly 5 lines.\nAll tables, figures, and animations are under project/outputs/.\nReproduce with: python run_all.py")

def main():
    for stage in STAGES: run_stage(stage)
    finalize()

if __name__=="__main__": main()
