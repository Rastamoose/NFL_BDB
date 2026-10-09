"""Stage 3: rush intent, PMOE, survival, pressure ablation, and disguise models."""
from __future__ import annotations
import json, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier, LGBMRegressor
from lifelines import CoxPHFitter, KaplanMeierFitter
from common import *

warnings.filterwarnings("ignore")
RNG=np.random.default_rng(RANDOM_SEED)

def prep(num,cat):
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])

def group_folds(df, requested=GROUP_FOLDS):
    """Return safe game-grouped folds and fail clearly on smoke-run inputs."""
    n_games = df.gameId.nunique()
    if n_games < 2:
        raise ValueError(f"Grouped validation requires at least 2 games; found {n_games}. Run 03_metrics.py without NFL_MAX_GAMES.")
    return GroupKFold(n_splits=min(requested, n_games))

def oof_classification(df, features_num, features_cat, model_factory, folds=GROUP_FOLDS):
    y=df.actualRush.astype(int).to_numpy(); pred=np.zeros(len(df)); groups=df.gameId.to_numpy()
    for tr,va in group_folds(df,folds).split(df,y,groups):
        pipe=Pipeline([("prep",prep(features_num,features_cat)),("model",model_factory())])
        pipe.fit(df.iloc[tr],y[tr]); pred[va]=pipe.predict_proba(df.iloc[va])[:,1]
    return pred

def crossfit_apply(train, target, features_num, features_cat, model_factory, folds=GROUP_FOLDS):
    """Predict every target row using a model that excluded its game."""
    pred=np.full(len(target),np.nan,dtype=float)
    y=train.actualRush.astype(int).to_numpy()
    game_splits=list(group_folds(train,folds).split(train,y,train.gameId))
    for tr,va in game_splits:
        held=set(train.iloc[va].gameId.unique())
        target_idx=np.flatnonzero(target.gameId.isin(held).to_numpy())
        if not len(target_idx): continue
        pipe=Pipeline([("prep",prep(features_num,features_cat)),("model",model_factory())])
        pipe.fit(train.iloc[tr],y[tr]); pred[target_idx]=pipe.predict_proba(target.iloc[target_idx])[:,1]
    return pred

def classify_metrics(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    return roc_auc_score(y,p),log_loss(y,p)

def pressure_oof(df,features):
    # Dummy-column discovery is outcome-blind; LightGBM handles missing values natively.
    X=pd.get_dummies(df[features],dummy_na=True).replace([np.inf,-np.inf],np.nan)
    y=df.pressure.astype(int).to_numpy(); pred=np.zeros(len(df)); groups=df.gameId
    for tr,va in group_folds(df).split(X,y,groups):
        m=LGBMClassifier(n_estimators=80,num_leaves=15,learning_rate=.04,random_state=RANDOM_SEED,verbosity=-1)
        m.fit(X.iloc[tr],y[tr]); pred[va]=m.predict_proba(X.iloc[va])[:,1]
    return pred

def game_bootstrap(df,pred,base_pred=None):
    games=df.gameId.unique(); vals=[]
    for _ in range(BOOTSTRAP_REPS):
        chosen=RNG.choice(games,len(games),replace=True)
        idx=np.concatenate([np.flatnonzero(df.gameId.to_numpy()==g) for g in chosen])
        y=df.pressure.to_numpy()[idx]
        if len(np.unique(y))<2: continue
        auc=roc_auc_score(y,pred[idx]); ll=log_loss(y,np.clip(pred[idx],1e-6,1-1e-6))
        if base_pred is None: vals.append((auc,ll))
        else: vals.append((auc-roc_auc_score(y,base_pred[idx]),log_loss(y,np.clip(base_pred[idx],1e-6,1-1e-6))-ll))
    a=np.asarray(vals); return np.quantile(a,[.025,.975],axis=0) if len(a) else np.full((2,2),np.nan)

def main():
    pm=pd.read_parquet(TABLE_DIR/"play_metrics_base.parquet")
    snap=pd.read_parquet(TABLE_DIR/"defender_snap_features.parquet")
    dframe=pd.read_parquet(TABLE_DIR/"defender_frame_features.parquet")
    results=[]

    # A. Snap-frame rush intent: logistic vs HistGB vs LightGBM.
    num=["xFromLOS","yFromBall","distanceBall","spacingNearest","oSin","oCos","down","yardsToGo","defendersInBox"]
    cat=["position","formation","personnelO","personnelD","coverage","coverageType"]
    train=snap[snap.phase.eq("snap")].reset_index(drop=True)
    factories={
      "logistic":lambda:LogisticRegression(max_iter=500,class_weight="balanced",random_state=RANDOM_SEED),
      "hist_gradient":lambda:HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.08,random_state=RANDOM_SEED),
      "lightgbm":lambda:LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.06,class_weight="balanced",random_state=RANDOM_SEED,verbosity=-1)}
    preds={}
    for name,factory in factories.items():
        p=oof_classification(train,num,cat,factory); preds[name]=p
        auc,ll=classify_metrics(train.actualRush,p)
        results.append({"model":"rush_intent_snap","variant":name,"AUC":auc,"logLoss":ll})
    best=min(preds,key=lambda k:classify_metrics(train.actualRush,preds[k])[1])
    train["rushProbability"]=preds[best]
    frac,mean=calibration_curve(train.actualRush,train.rushProbability,n_bins=10,strategy="quantile")
    pd.DataFrame({"meanPredicted":mean,"fractionPositive":frac}).to_csv(TABLE_DIR/"rush_intent_calibration.csv",index=False)

    # Apply the selected model out-of-game to true last-pre-snap rows.
    best_factory=factories[best]
    presnap=snap[snap.phase.eq("preSnap")].copy().reset_index(drop=True)
    if len(presnap):
        presnap["rushProbability"]=crossfit_apply(train,presnap,num,cat,best_factory)
    else:
        presnap["rushProbability"]=np.nan
    # Save an all-data model for later inference; reported RIS remains cross-fitted.
    best_pipe=Pipeline([("prep",prep(num,cat)),("model",best_factory())]).fit(train,train.actualRush)
    joblib.dump(best_pipe,TABLE_DIR/"rush_intent_model.joblib")
    def ris_table(x,name):
        p=np.clip(x.rushProbability,1e-6,1-1e-6)
        surpr=-np.where(x.actualRush.eq(1),np.log(p),np.log(1-p))
        return x.assign(surprise=surpr).groupby(["gameId","playId"]).surprise.sum().rename(name)
    pm=pm.merge(ris_table(train,"RIS"),on=["gameId","playId"],how="left")
    pm=pm.merge(ris_table(presnap,"RIS_preSnap"),on=["gameId","playId"],how="left")
    pd.concat([train,presnap],ignore_index=True).to_parquet(TABLE_DIR/"defender_snap_predictions.parquet",index=False)
    pd.DataFrame(results).to_csv(TABLE_DIR/"model_results_partial.csv",index=False)
    print(f"  snap rush-intent complete: {best}, AUC={classify_metrics(train.actualRush,preds[best])[0]:.3f}",flush=True)

    # B. Frame-level rush intent. Cross-fit by game, then predict every frame only
    # from models that excluded that frame's game (RRT is therefore out-of-game).
    fnum=["timeSinceSnap","xFromLOS","yFromBall","distanceQB","speed","acceleration","displacement","orientationToQB","dirToQB","oSin","oCos"]
    fcat=["position"]
    dframe=dframe.reset_index(drop=True)
    if len(dframe)>FRAME_MODEL_MAX_ROWS:
        sample=dframe.groupby("gameId",group_keys=False).sample(frac=min(1,FRAME_MODEL_MAX_ROWS/len(dframe)),random_state=RANDOM_SEED)
    else: sample=dframe.copy()
    frame_factory=lambda:LGBMClassifier(n_estimators=80,num_leaves=15,learning_rate=.05,class_weight="balanced",random_state=RANDOM_SEED,verbosity=-1)
    dframe["rushProbability"]=crossfit_apply(sample,dframe,fnum,fcat,frame_factory,folds=3).astype("float32")
    fp=dframe.loc[sample.index,"rushProbability"].to_numpy()
    fauc,fll=classify_metrics(sample.actualRush,fp); results.append({"model":"rush_intent_frame","variant":"lightgbm_game_oof","AUC":fauc,"logLoss":fll})
    frame_pipe=Pipeline([("prep",prep(fnum,fcat)),("model",frame_factory())]).fit(sample,sample.actualRush)
    joblib.dump(frame_pipe,TABLE_DIR/"rush_intent_frame_model.joblib")
    dframe.to_parquet(TABLE_DIR/"defender_frame_predictions.parquet",index=False)
    pd.DataFrame(results).to_csv(TABLE_DIR/"model_results_partial.csv",index=False)
    print(f"  frame rush-intent complete: AUC={fauc:.3f} across {len(dframe):,} rows",flush=True)
    # Eventual rushers only; if never >= threshold, use window end and flag censored.
    eventual=dframe[dframe.actualRush.eq(1)].copy()
    reveal=[]
    for key,g in eventual.groupby(["gameId","playId","nflId"],sort=False):
        hit=g[g.rushProbability.ge(RUSH_REVEAL_PROBABILITY)]
        rt=float(hit.timeSinceSnap.iloc[0]) if len(hit) else float(g.timeSinceSnap.max())
        reveal.append((*key,rt,int(hit.empty)))
    reveal=pd.DataFrame(reveal,columns=["gameId","playId","nflId","revealTime","revealCensored"])
    reveal.to_parquet(TABLE_DIR/"rusher_reveal_times.parquet",index=False)
    rrt=reveal.groupby(["gameId","playId"]).agg(RRT_mean=("revealTime","mean"),RRT_max=("revealTime","max"),RRT_censoredShare=("revealCensored","mean")).reset_index()
    pm=pm.merge(rrt,on=["gameId","playId"],how="left")

    # C. PMOE: OOF room-gain prediction, explicitly excluding QB identity.
    reg_features=["nearestRusherSnap","meanRusherSpeedSnap","meanBlockerDistanceSnap","staticRoomSnap","velocityRoomSnap","pocketAreaSnap","numRushers","numBlockers","down","yardsToGo","playAction","defendersInBox"]
    reg=pm.dropna(subset=["roomGain05"]).copy(); X=reg[reg_features].replace([np.inf,-np.inf],np.nan)
    pred=np.zeros(len(reg))
    for tr,va in group_folds(reg).split(X,reg.roomGain05,reg.gameId):
        m=LGBMRegressor(n_estimators=100,num_leaves=15,learning_rate=.04,random_state=RANDOM_SEED,verbosity=-1)
        m.fit(X.iloc[tr],reg.roomGain05.iloc[tr]); pred[va]=m.predict(X.iloc[va])
    reg["PMOE"]=reg.roomGain05-pred
    pm=pm.merge(reg[["gameId","playId","PMOE"]],on=["gameId","playId"],how="left")
    results.append({"model":"PMOE","variant":"group_game_oof","R2":r2_score(reg.roomGain05,pred),"MAE":mean_absolute_error(reg.roomGain05,pred)})
    time_train=reg.week.le(5); tm=LGBMRegressor(n_estimators=100,num_leaves=15,learning_rate=.04,random_state=RANDOM_SEED,verbosity=-1).fit(X[time_train],reg.loc[time_train,"roomGain05"])
    tp=tm.predict(X[~time_train]); results.append({"model":"PMOE","variant":"weeks_1_5_to_6_8","R2":r2_score(reg.loc[~time_train,"roomGain05"],tp),"MAE":mean_absolute_error(reg.loc[~time_train,"roomGain05"],tp)})
    joblib.dump(tm,TABLE_DIR/"pmoe_model.joblib")
    pm.to_parquet(TABLE_DIR/"play_metrics_partial.parquet",index=False)
    pd.DataFrame(results).to_csv(TABLE_DIR/"model_results_partial.csv",index=False)
    print(f"  PMOE complete: grouped R2={r2_score(reg.roomGain05,pred):.3f}",flush=True)
    try:
        import shap
        ex=shap.TreeExplainer(tm); sx=X[~time_train].sample(min(1000,(~time_train).sum()),random_state=RANDOM_SEED)
        sv=ex.shap_values(sx)
        pd.DataFrame(np.abs(sv).mean(axis=0),index=sx.columns,columns=["meanAbsSHAP"]).sort_values("meanAbsSHAP",ascending=False).to_csv(TABLE_DIR/"pmoe_shap.csv")
    except Exception as e:
        pd.DataFrame({"feature":reg_features,"meanAbsSHAP":np.nan,"note":str(e)}).to_csv(TABLE_DIR/"pmoe_shap.csv",index=False)

    # D. Censored reaction-lag survival model.
    surv=pm[pm.hasUnseenThreat.eq(1)].copy()
    scox=["reactionLag","reactionObserved","RIS","RRT_mean","numRushers","numBlockers","UPT","playAction","yardsToGo"]
    cs=surv[scox].replace([np.inf,-np.inf],np.nan).dropna()
    if len(cs)>50 and cs.reactionObserved.nunique()>1:
        try:
            cox=CoxPHFitter(penalizer=.05).fit(cs,duration_col="reactionLag",event_col="reactionObserved")
            cox.summary.reset_index().to_csv(TABLE_DIR/"survival_coefficients.csv",index=False)
            results.append({"model":"reaction_survival","variant":"cox","concordance":cox.concordance_index_})
        except Exception as e:
            pd.DataFrame({"term":["model_failure"],"note":[str(e)]}).to_csv(TABLE_DIR/"survival_coefficients.csv",index=False)
        # Nonparametric curves remain available even if the adjusted Cox fit fails.
        surv["disguiseTertile"]=pd.qcut(surv.RIS.rank(method="first"),3,labels=["low","middle","high"])
        curves=[]
        for label,g in surv.groupby("disguiseTertile",observed=True):
            km=KaplanMeierFitter().fit(g.reactionLag,event_observed=g.reactionObserved,label=str(label))
            for t,s in zip(km.survival_function_.index,km.survival_function_.iloc[:,0]): curves.append({"tertile":label,"time":t,"survival":s})
        pd.DataFrame(curves).to_csv(TABLE_DIR/"survival_curves.csv",index=False)
    else:
        pd.DataFrame(columns=["term","note"]).to_csv(TABLE_DIR/"survival_coefficients.csv",index=False)
        pd.DataFrame(columns=["tertile","time","survival"]).to_csv(TABLE_DIR/"survival_curves.csv",index=False)

    # E. Pressure ablation with game-grouped OOF and game bootstrap CIs.
    base=["timeToThrow","numRushers","numBlockers","maxSTRAIN","pocketAreaMin","staticRoomSnap","nearestRusherSnap","down","yardsToGo","playAction","defendersInBox","week","coverageType","dropBackType"]
    additions={"UPT":["UPT","unseenPressureShare","maxUnseenThreats"],"RL":["reactionLag","reactionCensored"],
      "PMOE":["PMOE"],"RIS":["RIS","RIS_preSnap"],"RRT":["RRT_mean","RRT_max"],"velocity_room":["velocityRoomSnap","velocityRoomEnd","velocityRoomGain05"]}
    model_df=pm.copy(); allpred={}; ab=[]
    allpred["baseline"]=pressure_oof(model_df,base)
    bauc,bll=classify_metrics(model_df.pressure,allpred["baseline"]); bci=game_bootstrap(model_df,allpred["baseline"])
    ab.append({"variant":"baseline","AUC":bauc,"AUC_gain":0,"AUC_gain_low":0,"AUC_gain_high":0,"logLoss":bll,"logLoss_gain":0,"logLoss_gain_low":0,"logLoss_gain_high":0})
    for name,cols in {**additions,"all_new":sum(additions.values(),[])}.items():
        p=pressure_oof(model_df,base+cols); allpred[name]=p; auc,ll=classify_metrics(model_df.pressure,p); ci=game_bootstrap(model_df,p,allpred["baseline"])
        ab.append({"variant":name,"AUC":auc,"AUC_gain":auc-bauc,"AUC_gain_low":ci[0,0],"AUC_gain_high":ci[1,0],"logLoss":ll,"logLoss_gain":bll-ll,"logLoss_gain_low":ci[0,1],"logLoss_gain_high":ci[1,1]})
    pd.DataFrame(ab).to_csv(TABLE_DIR/"pressure_ablation.csv",index=False)
    pd.DataFrame({"gameId":pm.gameId,"playId":pm.playId,"actual":pm.pressure,**{k:v for k,v in allpred.items()}}).to_parquet(TABLE_DIR/"pressure_predictions.parquet",index=False)
    pm.to_parquet(TABLE_DIR/"play_metrics_partial.parquet",index=False)
    print("  pressure ablation complete",flush=True)
    # Required weeks 1-5 -> 6-8 check for full model.
    f=base+sum(additions.values(),[]); X=pd.get_dummies(model_df[f],dummy_na=True).replace([np.inf,-np.inf],np.nan)
    tr=model_df.week.le(5); mm=LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.04,random_state=RANDOM_SEED,verbosity=-1).fit(X[tr],model_df.loc[tr,"pressure"])
    pt=mm.predict_proba(X[~tr])[:,1]; results.append({"model":"pressure","variant":"weeks_1_5_to_6_8","AUC":roc_auc_score(model_df.loc[~tr,"pressure"],pt),"logLoss":log_loss(model_df.loc[~tr,"pressure"],pt)})

    # F. Mixed effects for Disguise Drag; failures are recorded rather than hidden.
    mixed_rows=[]
    try:
        import statsmodels.formula.api as smf
        obs=pm[(pm.hasUnseenThreat.eq(1)) & pm.reactionObserved.eq(1)].dropna(subset=["reactionLag","RIS","RRT_mean"]).copy()
        formula="reactionLag ~ RIS + RRT_mean + numRushers + numBlockers + C(coverageType) + playAction + C(dropBackType) + down + yardsToGo"
        fit=smf.mixedlm(formula,obs,groups=obs.qbId,vc_formula={"defense":"0+C(defense)"}).fit(reml=False,method="lbfgs",maxiter=200)
        ci=fit.conf_int();
        for term in ["RIS","RRT_mean"]: mixed_rows.append({"outcome":"reactionLag","term":term,"coefficient":fit.params.get(term,np.nan),"low":ci.loc[term,0],"high":ci.loc[term,1],"p":fit.pvalues.get(term,np.nan)})
        fit2=smf.mixedlm("PMOE ~ RIS + RRT_mean + numRushers + numBlockers + C(coverageType) + playAction + C(dropBackType) + down + yardsToGo",pm.dropna(subset=["PMOE","RIS","RRT_mean"]),groups=pm.dropna(subset=["PMOE","RIS","RRT_mean"]).qbId,vc_formula={"defense":"0+C(defense)"}).fit(reml=False,method="lbfgs",maxiter=200)
        ci2=fit2.conf_int();
        for term in ["RIS","RRT_mean"]: mixed_rows.append({"outcome":"PMOE","term":term,"coefficient":fit2.params.get(term,np.nan),"low":ci2.loc[term,0],"high":ci2.loc[term,1],"p":fit2.pvalues.get(term,np.nan)})
    except Exception as e: mixed_rows.append({"outcome":"model_failure","term":str(e),"coefficient":np.nan,"low":np.nan,"high":np.nan,"p":np.nan})
    pd.DataFrame(mixed_rows).to_csv(TABLE_DIR/"disguise_drag_mixed_effects.csv",index=False)

    # Shrunk leaderboards based on residualized outcomes (simple empirical-Bayes shrinkage).
    def leaderboard(group,name,value,ascending):
        d=pm.dropna(subset=[value]); overall=d[value].mean()
        g=d.groupby(group).agg(snaps=(value,"size"),raw=(value,"mean"),raw_sd=(value,"std")).reset_index()
        shrink=g.snaps/(g.snaps+MIN_LEADERBOARD_SNAPS)
        g["score"]=overall+(g.raw-overall)*shrink
        half=1.96*g.raw_sd.fillna(0)/np.sqrt(g.snaps)*shrink
        g["low"]=g.score-half; g["high"]=g.score+half
        return g[g.snaps.ge(MIN_LEADERBOARD_SNAPS)].sort_values("score",ascending=ascending).rename(columns={group:name})
    leaderboard("qbId","qbId","reactionLag",True).to_csv(TABLE_DIR/"qb_disguise_resistance.csv",index=False)
    leaderboard("defense","defense","reactionLag",False).to_csv(TABLE_DIR/"defense_disguise_yield.csv",index=False)
    pm.to_parquet(TABLE_DIR/"play_metrics.parquet",index=False)
    pd.DataFrame(results).to_csv(TABLE_DIR/"model_results.csv",index=False)
    stage_summary("STAGE 3 COMPLETE",[
      f"Snap rush-intent winner: {best}; AUC {classify_metrics(train.actualRush,preds[best])[0]:.3f}.",
      f"PMOE game-grouped R² {r2_score(reg.roomGain05,pred):.3f}; fitted censored RL and mixed-effects models.",
      f"Pressure ablation tested baseline plus each new metric with {BOOTSTRAP_REPS} game-level bootstrap replicates."])

if __name__=="__main__": main()
