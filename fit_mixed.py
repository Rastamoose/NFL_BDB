"""Refit only Disguise Drag mixed models after statsmodels compatibility update."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import TABLE_DIR
warnings.filterwarnings("ignore")
pm=pd.read_parquet(TABLE_DIR/"play_metrics.parquet")
rows=[]
def fit_one(outcome,data):
    needed=[outcome,"RIS","RRT_mean","numRushers","numBlockers","coverageType","playAction","dropBackType","down","yardsToGo","qbId","defense"]
    data=data.dropna(subset=needed).reset_index(drop=True)
    formula=f"{outcome} ~ RIS + RRT_mean + numRushers + numBlockers + C(coverageType) + playAction + C(dropBackType) + down + yardsToGo"
    fit=smf.mixedlm(formula,data,groups=data.qbId,vc_formula={"defense":"0+C(defense)"}).fit(reml=False,method="lbfgs",maxiter=80)
    ci=fit.conf_int()
    for term in ["RIS","RRT_mean"]:
        rows.append({"outcome":outcome,"term":term,"coefficient":fit.params[term],"low":ci.loc[term,0],"high":ci.loc[term,1],"p":fit.pvalues[term]})
obs=pm[(pm.hasUnseenThreat.eq(1))&pm.reactionObserved.eq(1)].dropna(subset=["reactionLag","RIS","RRT_mean"]).copy()
fit_one("reactionLag",obs)
fit_one("PMOE",pm.dropna(subset=["PMOE","RIS","RRT_mean"]).copy())
pd.DataFrame(rows).to_csv(TABLE_DIR/"disguise_drag_mixed_effects.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
