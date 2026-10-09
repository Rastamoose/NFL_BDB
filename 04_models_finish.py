"""Resume the model pipeline from saved rush-intent predictions."""
import warnings, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_absolute_error
from lightgbm import LGBMClassifier, LGBMRegressor
from lifelines import CoxPHFitter, KaplanMeierFitter
from common import *
warnings.filterwarnings("ignore"); rng=np.random.default_rng(RANDOM_SEED)

def folds(d): return GroupKFold(n_splits=min(GROUP_FOLDS,d.gameId.nunique()))
def metrics(y,p): return roc_auc_score(y,p),log_loss(y,np.clip(p,1e-6,1-1e-6))
def pressure_oof(d,cols):
 X=pd.get_dummies(d[cols],dummy_na=True).replace([np.inf,-np.inf],np.nan); y=d.pressure.astype(int).to_numpy(); p=np.zeros(len(d))
 for tr,va in folds(d).split(X,y,d.gameId):
  m=LGBMClassifier(n_estimators=80,num_leaves=15,learning_rate=.05,random_state=RANDOM_SEED,verbosity=-1).fit(X.iloc[tr],y[tr]); p[va]=m.predict_proba(X.iloc[va])[:,1]
 return p
def boot(d,p,b):
 vals=[]; games=d.gameId.unique()
 for _ in range(BOOTSTRAP_REPS):
  chosen=rng.choice(games,len(games),replace=True); ix=np.concatenate([np.flatnonzero(d.gameId.to_numpy()==g) for g in chosen]); y=d.pressure.to_numpy()[ix]
  vals.append((roc_auc_score(y,p[ix])-roc_auc_score(y,b[ix]),log_loss(y,np.clip(b[ix],1e-6,1-1e-6))-log_loss(y,np.clip(p[ix],1e-6,1-1e-6))))
 return np.quantile(vals,[.025,.975],axis=0)
def main():
 pm=pd.read_parquet(TABLE_DIR/"play_metrics_base.parquet")
 context=pd.read_csv(DATA_DIR/"plays.csv",usecols=["gameId","playId","defendersInBox"])
 pm=pm.merge(context,on=["gameId","playId"],how="left")
 sp=pd.read_parquet(TABLE_DIR/"defender_snap_predictions.parquet")
 def ris(x,name):
  p=np.clip(x.rushProbability,1e-6,1-1e-6); z=-np.where(x.actualRush.eq(1),np.log(p),np.log(1-p)); return x.assign(z=z).groupby(["gameId","playId"]).z.sum().rename(name)
 pm=pm.merge(ris(sp[sp.phase.eq("snap")],"RIS"),on=["gameId","playId"],how="left").merge(ris(sp[sp.phase.eq("preSnap")],"RIS_preSnap"),on=["gameId","playId"],how="left")
 reveal=pd.read_parquet(TABLE_DIR/"rusher_reveal_times.parquet"); rrt=reveal.groupby(["gameId","playId"]).agg(RRT_mean=("revealTime","mean"),RRT_max=("revealTime","max"),RRT_censoredShare=("revealCensored","mean")).reset_index(); pm=pm.merge(rrt,on=["gameId","playId"],how="left")
 results=pd.read_csv(TABLE_DIR/"model_results_partial.csv").to_dict("records")
 # PMOE
 rf=["nearestRusherSnap","meanRusherSpeedSnap","meanBlockerDistanceSnap","staticRoomSnap","velocityRoomSnap","pocketAreaSnap","numRushers","numBlockers","down","yardsToGo","playAction","defendersInBox"]
 reg=pm.dropna(subset=["roomGain05"]); X=reg[rf].replace([np.inf,-np.inf],np.nan); pred=np.zeros(len(reg))
 for tr,va in folds(reg).split(X,reg.roomGain05,reg.gameId):
  m=LGBMRegressor(n_estimators=100,num_leaves=15,learning_rate=.05,random_state=RANDOM_SEED,verbosity=-1).fit(X.iloc[tr],reg.roomGain05.iloc[tr]); pred[va]=m.predict(X.iloc[va])
 reg=reg.copy(); reg["PMOE"]=reg.roomGain05-pred; pm=pm.merge(reg[["gameId","playId","PMOE"]],on=["gameId","playId"],how="left")
 results.append({"model":"PMOE","variant":"group_game_oof","R2":r2_score(reg.roomGain05,pred),"MAE":mean_absolute_error(reg.roomGain05,pred)})
 tr=reg.week.le(5); tm=LGBMRegressor(n_estimators=100,num_leaves=15,learning_rate=.05,random_state=RANDOM_SEED,verbosity=-1).fit(X[tr],reg.loc[tr,"roomGain05"]); tp=tm.predict(X[~tr]); results.append({"model":"PMOE","variant":"weeks_1_5_to_6_8","R2":r2_score(reg.loc[~tr,"roomGain05"],tp),"MAE":mean_absolute_error(reg.loc[~tr,"roomGain05"],tp)}); joblib.dump(tm,TABLE_DIR/"pmoe_model.joblib")
 try:
  import shap; sx=X[~tr].sample(min(500,(~tr).sum()),random_state=RANDOM_SEED); sv=shap.TreeExplainer(tm).shap_values(sx); pd.DataFrame({"feature":sx.columns,"meanAbsSHAP":np.abs(sv).mean(0)}).sort_values("meanAbsSHAP",ascending=False).to_csv(TABLE_DIR/"pmoe_shap.csv",index=False)
 except Exception as e: pd.DataFrame({"feature":rf,"meanAbsSHAP":np.nan}).to_csv(TABLE_DIR/"pmoe_shap.csv",index=False)
 print(f"  PMOE R2={r2_score(reg.roomGain05,pred):.3f}",flush=True)
 # Survival
 s=pm[pm.hasUnseenThreat.eq(1)].dropna(subset=["reactionLag","RIS","RRT_mean"]).copy(); cols=["reactionLag","reactionObserved","RIS","RRT_mean","numRushers","numBlockers","UPT","playAction","yardsToGo"]
 try:
  c=CoxPHFitter(penalizer=.05).fit(s[cols],"reactionLag","reactionObserved"); c.summary.reset_index().to_csv(TABLE_DIR/"survival_coefficients.csv",index=False); results.append({"model":"reaction_survival","variant":"cox","concordance":c.concordance_index_})
 except Exception as e: pd.DataFrame({"term":["failure"],"note":[str(e)]}).to_csv(TABLE_DIR/"survival_coefficients.csv",index=False)
 s["disguiseTertile"]=pd.qcut(s.RIS.rank(method="first"),3,labels=["low","middle","high"]); curves=[]
 for label,g in s.groupby("disguiseTertile",observed=True):
  km=KaplanMeierFitter().fit(g.reactionLag,g.reactionObserved)
  curves.extend({"tertile":label,"time":t,"survival":v} for t,v in zip(km.survival_function_.index,km.survival_function_.iloc[:,0]))
 pd.DataFrame(curves).to_csv(TABLE_DIR/"survival_curves.csv",index=False)
 # Pressure ablation
 base=["timeToThrow","numRushers","numBlockers","maxSTRAIN","pocketAreaMin","staticRoomSnap","nearestRusherSnap","down","yardsToGo","playAction","defendersInBox","week","coverageType","dropBackType"]
 adds={"UPT":["UPT","unseenPressureShare","maxUnseenThreats"],"RL":["reactionLag","reactionCensored"],"PMOE":["PMOE"],"RIS":["RIS","RIS_preSnap"],"RRT":["RRT_mean","RRT_max"],"velocity_room":["velocityRoomSnap","velocityRoomEnd","velocityRoomGain05"]}
 bp=pressure_oof(pm,base); ba,bl=metrics(pm.pressure,bp); rows=[{"variant":"baseline","AUC":ba,"AUC_gain":0,"AUC_gain_low":0,"AUC_gain_high":0,"logLoss":bl,"logLoss_gain":0,"logLoss_gain_low":0,"logLoss_gain_high":0}]; predictions={"baseline":bp}
 for name,cx in {**adds,"all_new":sum(adds.values(),[])}.items():
  p=pressure_oof(pm,base+cx); a,l=metrics(pm.pressure,p); ci=boot(pm,p,bp); rows.append({"variant":name,"AUC":a,"AUC_gain":a-ba,"AUC_gain_low":ci[0,0],"AUC_gain_high":ci[1,0],"logLoss":l,"logLoss_gain":bl-l,"logLoss_gain_low":ci[0,1],"logLoss_gain_high":ci[1,1]}); predictions[name]=p
 pd.DataFrame(rows).to_csv(TABLE_DIR/"pressure_ablation.csv",index=False); pd.DataFrame({"gameId":pm.gameId,"playId":pm.playId,"actual":pm.pressure,**predictions}).to_parquet(TABLE_DIR/"pressure_predictions.parquet",index=False)
 f=base+sum(adds.values(),[]); XX=pd.get_dummies(pm[f],dummy_na=True).replace([np.inf,-np.inf],np.nan); mask=pm.week.le(5); mm=LGBMClassifier(n_estimators=100,num_leaves=15,random_state=RANDOM_SEED,verbosity=-1).fit(XX[mask],pm.loc[mask,"pressure"]); pp=mm.predict_proba(XX[~mask])[:,1]; results.append({"model":"pressure","variant":"weeks_1_5_to_6_8","AUC":roc_auc_score(pm.loc[~mask,"pressure"],pp),"logLoss":log_loss(pm.loc[~mask,"pressure"],pp)})
 print(f"  pressure AUC baseline={ba:.3f}, all-new={rows[-1]['AUC']:.3f}",flush=True)
 # Mixed models and shrinkage
 mixed=[]
 try:
  import statsmodels.formula.api as smf; obs=pm[(pm.hasUnseenThreat.eq(1))&pm.reactionObserved.eq(1)].dropna(subset=["reactionLag","RIS","RRT_mean"]); fit=smf.mixedlm("reactionLag ~ RIS + RRT_mean + numRushers + numBlockers + C(coverageType) + playAction + C(dropBackType) + down + yardsToGo",obs,groups=obs.qbId,vc_formula={"defense":"0+C(defense)"}).fit(reml=False,method="lbfgs",maxiter=80); ci=fit.conf_int(); mixed.extend({"outcome":"reactionLag","term":x,"coefficient":fit.params[x],"low":ci.loc[x,0],"high":ci.loc[x,1],"p":fit.pvalues[x]} for x in ["RIS","RRT_mean"])
 except Exception as e: mixed=[{"outcome":"model_failure","term":str(e),"coefficient":np.nan,"low":np.nan,"high":np.nan,"p":np.nan}]
 pd.DataFrame(mixed).to_csv(TABLE_DIR/"disguise_drag_mixed_effects.csv",index=False)
 def lead(group,value,asc):
  d=pm.dropna(subset=[value]); mean=d[value].mean(); g=d.groupby(group).agg(snaps=(value,"size"),raw=(value,"mean"),raw_sd=(value,"std")).reset_index(); sh=g.snaps/(g.snaps+MIN_LEADERBOARD_SNAPS); g["score"]=mean+(g.raw-mean)*sh; h=1.96*g.raw_sd.fillna(0)/np.sqrt(g.snaps)*sh; g["low"]=g.score-h; g["high"]=g.score+h; return g[g.snaps.ge(MIN_LEADERBOARD_SNAPS)].sort_values("score",ascending=asc)
 lead("qbId","reactionLag",True).to_csv(TABLE_DIR/"qb_disguise_resistance.csv",index=False); lead("defense","reactionLag",False).to_csv(TABLE_DIR/"defense_disguise_yield.csv",index=False)
 pm.to_parquet(TABLE_DIR/"play_metrics.parquet",index=False); pd.DataFrame(results).to_csv(TABLE_DIR/"model_results.csv",index=False)
 print("[STAGE 3 COMPLETE]\nFinal metrics and model outputs saved.\nGame-grouped and time-split evaluations completed.\nCensored reaction and disguise outputs completed.")
if __name__=="__main__": main()
