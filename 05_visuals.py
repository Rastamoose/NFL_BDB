"""Stage 4: showcase animations and publication-style static figures."""
from __future__ import annotations
import math, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Wedge, Polygon
from scipy.spatial import ConvexHull
from common import *

OFF="#0072B2"; DEF="#D55E00"; UNSEEN="#CC79A7"; SEEN="#E69F00"; BALL="#6B3E26"; GREEN="#2F6B3C"

def select_showcases(pm):
    specs=[
      ("sack_high_upt",pm[pm.sack.eq(1)].sort_values("UPT",ascending=False),"Sack with high unseen-pressure time"),
      ("clean_long_throw",pm[pm.pressure.eq(0)].sort_values("timeToThrow",ascending=False),"Clean pocket with a long time to throw"),
      ("high_ris_hurry",pm[pm.hurry.eq(1)].sort_values("RIS",ascending=False),"High-surprise rush producing a hurry"),
      ("high_ris_no_pressure",pm[pm.pressure.eq(0)].sort_values("RIS",ascending=False),"High-surprise rush without pressure"),
      ("fast_reaction_escape",pm[(pm.reactionObserved.eq(1))&pm.pressure.eq(0)].sort_values("reactionLag"),"Fast QB reaction and no pressure"),
      ("long_reaction_sack",pm[(pm.reactionObserved.eq(1))&pm.sack.eq(1)].sort_values("reactionLag",ascending=False),"Long QB reaction ending in a sack")]
    rows=[]; used=set()
    for slug,candidates,label in specs:
        row=None
        for _,r in candidates.iterrows():
            key=(int(r.gameId),int(r.playId))
            if key not in used: row=r; used.add(key); break
        if row is None:
            row=pm.sort_values("UPT",ascending=False).iloc[len(rows)]; key=(int(row.gameId),int(row.playId)); used.add(key)
        rows.append({"slug":slug,"criterion":label,**row.to_dict()})
    return pd.DataFrame(rows)

def draw_field(ax,los):
    ax.clear(); ax.set_facecolor(GREEN); ax.set_xlim(0,120); ax.set_ylim(0,53.3)
    for x in range(10,111,5): ax.axvline(x,color="white",lw=.5 if x%10 else 1,alpha=.65)
    ax.axvspan(0,10,color="#1f5130"); ax.axvspan(110,120,color="#1f5130")
    if pd.notna(los): ax.axvline(los,color="#F0E442",lw=2,ls="--",label="line of scrimmage")
    ax.set_xlabel("Field length (yards)"); ax.set_ylabel("Field width (yards)")

def animate_play(row, dprob, fm, rm):
    gid,pid=int(row.gameId),int(row.playId)
    clean=pd.read_parquet(CLEAN_DIR/f"clean_{gid}.parquet")
    play=clean[(clean.playId.eq(pid))&clean.inWindow].copy()
    frames=sorted(play.frameId.unique()); step=max(1,math.ceil(len(frames)/ANIMATION_MAX_FRAMES)); frames=frames[::step]
    snap=int(play.snapFrame.iloc[0]); los=float(play.absoluteYardlineNumber.iloc[0]); qid=int(play.loc[play.pff_role.astype(str).eq("Pass"),"nflId"].iloc[0])
    pff_probs=dprob[(dprob.gameId.eq(gid))&dprob.playId.eq(pid)]
    fmet=fm[(fm.gameId.eq(gid))&fm.playId.eq(pid)].set_index("frameId")
    rmet=rm[(rm.gameId.eq(gid))&rm.playId.eq(pid)]
    fig=plt.figure(figsize=(15,9)); gs=fig.add_gridspec(2,2,height_ratios=[2.1,1],width_ratios=[2.3,1],hspace=.28,wspace=.18)
    field=fig.add_subplot(gs[0,0]); live=fig.add_subplot(gs[0,1]); ts=fig.add_subplot(gs[1,0]); probs=fig.add_subplot(gs[1,1])
    # Static time-series context with moving cursors.
    tt=fmet.index.to_numpy(); time=(tt-snap)/FPS
    ts.plot(time,fmet.qbRoom,color=OFF,label="QB Voronoi room (yd²)"); ts.plot(time,fmet.pocketArea,color="#009E73",label="blocker-hull area (yd²)")
    ts2=ts.twinx(); ts2.step(time,fmet.unseenThreatCount,color=UNSEEN,label="unseen threats",where="post")
    ts.set_title("Pocket and unseen-pressure metrics"); ts.set_xlabel("Seconds after snap"); ts.set_ylabel("Area (yd²)"); ts2.set_ylabel("Threat count")
    lines=ts.get_lines()+ts2.get_lines(); ts.legend(lines,[x.get_label() for x in lines],fontsize=7,loc="upper right")
    for evf,e in play.groupby("frameId").event.first().items():
        if str(e)!="None" and evf>=snap: ts.axvline((evf-snap)/FPS,color="gray",alpha=.25,lw=.8)
    for did,g in pff_probs.groupby("nflId"):
        probs.plot(g.timeSinceSnap,g.rushProbability,lw=1,label=f"#{int(play.loc[play.nflId.eq(did),'jerseyNumber'].dropna().iloc[0])}")
    probs.axhline(RUSH_REVEAL_PROBABILITY,color="black",ls="--",lw=.8); probs.set_ylim(0,1); probs.set_title("Each defender's predicted P(rush)"); probs.set_xlabel("Seconds after snap"); probs.set_ylabel("Probability"); probs.legend(ncol=3,fontsize=6)
    cur1=ts.axvline(0,color="black",lw=2); cur2=probs.axvline(0,color="black",lw=2)
    def update(i):
        frame=frames[i]; fr=play[play.frameId.eq(frame)]; draw_field(field,los)
        qb=fr[fr.nflId.eq(qid)].iloc[0]; others=fr[fr.nflId.notna()&~fr.nflId.eq(qid)]
        poly=voronoi_polygon(qb.x,qb.y,others[["x","y"]].to_numpy())
        field.add_patch(Polygon(poly,closed=True,color="#56B4E9",alpha=.22,zorder=1))
        center=90-float(qb.o_smooth); field.add_patch(Wedge((qb.x,qb.y),ROOM_RADIUS_YARDS,center-VIEW_CONE_HALF_ANGLE,center+VIEW_CONE_HALF_ANGLE,color="#F0E442",alpha=.18,zorder=1))
        blockers=fr[fr.pff_role.astype(str).eq("Pass Block")][["x","y"]].to_numpy()
        if len(blockers)>=3:
            try:
                h=ConvexHull(blockers); hp=blockers[h.vertices]; field.plot(np.r_[hp[:,0],hp[0,0]],np.r_[hp[:,1],hp[0,1]],color="#009E73",lw=2)
            except Exception: pass
        rnow=rmet[rmet.frameId.eq(frame)].set_index("nflId") if len(rmet) else pd.DataFrame()
        for _,p in fr[fr.nflId.notna()].iterrows():
            role=str(p.pff_role); color=OFF if p.is_offense else DEF
            if role=="Pass Rush" and not rnow.empty and p.nflId in rnow.index: color=UNSEEN if rnow.loc[p.nflId,"unseen"] else SEEN
            field.scatter(p.x,p.y,s=145,color=color,edgecolor="white",zorder=3)
            field.text(p.x,p.y,str(int(p.jerseyNumber)),ha="center",va="center",fontsize=5.5,color="white",zorder=4)
            field.arrow(p.x,p.y,p.vx*.25,p.vy*.25,color="white",width=.015,head_width=.45,alpha=.65,zorder=2)
        ball=fr[fr.team.astype(str).eq("football")]
        if len(ball): field.scatter(ball.x,ball.y,s=65,color=BALL,marker="D",edgecolor="white",zorder=5)
        t=(frame-snap)/FPS; metric=fmet.loc[frame] if frame in fmet.index else fmet.iloc[-1]
        live.clear(); live.axis("off"); live.set_title("Live metrics",fontweight="bold")
        status="not triggered" if pd.isna(row.firstUnseenFrame) else ("waiting/censored" if not row.reactionObserved else f"response at {row.reactionLag:.2f}s")
        text=(f"Time since snap: {t:.1f} s\nNearest rusher: {metric.nearestRusherDistance:.2f} yd\nQB room: {metric.qbRoom:.1f} yd²\nVelocity-aware room: {metric.velocityAwareRoom:.1f} yd²\nBlocker-hull area: {metric.pocketArea:.1f} yd²\nUnseen threats now: {int(metric.unseenThreatCount)}\nUPT so far: {metric.uptSoFar:.1f} s\nReaction lag: {status}\nRIS: {row.RIS:.2f}\nMean RRT: {row.RRT_mean:.2f} s")
        live.text(.02,.92,text,va="top",fontsize=11,linespacing=1.55)
        ev=fr.event.iloc[0]; field.set_title(f"Play {pid} | frame {frame} | {ev if str(ev)!='None' else 'tracking'}")
        cur1.set_xdata([t,t]); cur2.set_xdata([t,t])
        fig.suptitle(f"{row.criterion}: {str(row.description)[:115]}\nBlue offense · orange defense · pink unseen rusher · yellow seen rusher",fontsize=11,fontweight="bold")
        return []
    ani=FuncAnimation(fig,update,frames=len(frames),blit=False)
    out=ANIMATION_DIR/f"{row.slug}_game_{gid}_play_{pid}.gif"; ani.save(out,writer=PillowWriter(fps=ANIMATION_FPS),dpi=80); plt.close(fig)
    return out

def takeaway(fig,text): fig.text(.5,.01,"Takeaway: "+text,ha="center",fontsize=9,style="italic")

def main():
    pm=pd.read_parquet(TABLE_DIR/"play_metrics.parquet"); fm=pd.read_parquet(TABLE_DIR/"frame_metrics.parquet"); rm=pd.read_parquet(TABLE_DIR/"rusher_frame_metrics.parquet"); dp=pd.read_parquet(TABLE_DIR/"defender_frame_predictions.parquet")
    players=pd.read_csv(DATA_DIR/"players.csv")
    showcases=select_showcases(pm); showcases.to_csv(TABLE_DIR/"showcase_plays.csv",index=False)
    captions=[]
    for i,row in showcases.iterrows():
        print(f"  animation {i+1}/{len(showcases)}: {row.slug}")
        expected=ANIMATION_DIR/f"{row.slug}_game_{int(row.gameId)}_play_{int(row.playId)}.gif"
        out=expected if expected.exists() else animate_play(row,dp,fm,rm)
        captions.append(f"- `{out.name}` — {row.criterion}. Selected by the stated metric rule; {row.description}")
    (ANIMATION_DIR/"SHOWCASE_CAPTIONS.md").write_text("# Showcase selection\n\n"+"\n".join(captions)+"\n\nColors are colorblind-safe. Body-facing cones use shoulder-pad orientation, not gaze. MP4 was skipped because ffmpeg is unavailable.\n",encoding="utf-8")

    # Reveal-time timeline per defender.
    rrt=pd.read_parquet(TABLE_DIR/"rusher_reveal_times.parquet"); first=showcases.iloc[0]; x=rrt[(rrt.gameId.eq(first.gameId))&rrt.playId.eq(first.playId)].sort_values("revealTime")
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh(x.nflId.astype(int).astype(str),x.revealTime,color=DEF); ax.axvline(RUSH_REVEAL_PROBABILITY,color="none"); ax.set(xlabel="Seconds after snap",ylabel="Defender nflId",title="Rush reveal time by eventual rusher"); takeaway(fig,"Later bars indicate a rusher whose assignment remained ambiguous longer."); fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(FIGURE_DIR/"reveal_time_timeline.png",dpi=160); plt.close(fig)

    # QB dumbbell: low vs high disguise within QB.
    q=pm.dropna(subset=["RIS","reactionLag"]).copy(); q["tier"]=pd.qcut(q.RIS.rank(method="first"),3,labels=["low","mid","high"]); z=q[q.tier.isin(["low","high"])].pivot_table(index="qbId",columns="tier",values="reactionLag",aggfunc="mean",observed=True).dropna(); z.columns=z.columns.astype(str); counts=q.groupby("qbId").size(); z=z[counts.reindex(z.index).ge(MIN_LEADERBOARD_SNAPS)]; names=players.set_index("nflId")["displayName"]; z["displayName"]=z.index.to_series().map(names)
    fig,ax=plt.subplots(figsize=(9,max(4,len(z)*.32))); y=np.arange(len(z)); ax.hlines(y,z.low,z.high,color="gray"); ax.scatter(z.low,y,color=OFF,label="low disguise"); ax.scatter(z.high,y,color=UNSEEN,label="high disguise"); ax.set_yticks(y,z.displayName); ax.set_xlabel("Mean reaction lag (s)"); ax.set_title("QB reaction lag: low vs high rush disguise"); ax.legend(); takeaway(fig,"Rightward movement under high disguise means a longer measured response lag."); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(FIGURE_DIR/"qb_disguise_dumbbell.png",dpi=160); plt.close(fig)

    # Defense leaderboard with approximate 95% uncertainty bars after shrinkage.
    lb=pd.read_csv(TABLE_DIR/"defense_disguise_yield.csv").head(15); shown=lb.iloc[::-1]; fig,ax=plt.subplots(figsize=(8,6)); ax.barh(shown.defense,shown.score,color=DEF)
    if {"low","high"}.issubset(shown.columns):
        ax.errorbar(shown.score,shown.defense,xerr=[shown.score-shown.low,shown.high-shown.score],fmt="none",ecolor="black",capsize=3)
    ax.set_xlabel("Shrunk mean reaction lag (s)"); ax.set_title("Defense Disguise Yield leaderboard"); takeaway(fig,"Bars are shrunk toward the league mean; whiskers show approximate 95% uncertainty for qualified defenses."); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(FIGURE_DIR/"defense_disguise_yield.png",dpi=160); plt.close(fig)

    # Survival curves (produce a clear placeholder if the optional fit had insufficient data).
    sc_path=TABLE_DIR/"survival_curves.csv"; sc=pd.read_csv(sc_path) if sc_path.exists() else pd.DataFrame(); fig,ax=plt.subplots(figsize=(8,5))
    if len(sc):
        for label,g in sc.groupby("tertile"): ax.step(g.time,g.survival,where="post",label=f"{label} disguise")
        ax.legend(); note="Higher curves indicate quarterbacks remain unresponsive longer, including censored plays."
    else:
        ax.text(.5,.5,"Survival curves unavailable: insufficient events or model failure",ha="center",va="center",transform=ax.transAxes); note="No curve is shown when the censored reaction sample is insufficient."
    ax.set(xlabel="Seconds since first unseen threat",ylabel="Probability response has not occurred",title="Reaction-lag survival by disguise tertile"); takeaway(fig,note); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(FIGURE_DIR/"reaction_survival.png",dpi=160); plt.close(fig)

    # AUC ablation.
    ab=pd.read_csv(TABLE_DIR/"pressure_ablation.csv"); fig,ax=plt.subplots(figsize=(9,5)); colors=["gray" if v=="baseline" else OFF for v in ab.variant]; ax.bar(ab.variant,ab.AUC_gain,color=colors); ax.errorbar(np.arange(len(ab)),ab.AUC_gain,yerr=[ab.AUC_gain-ab.AUC_gain_low,ab.AUC_gain_high-ab.AUC_gain],fmt="none",ecolor="black",capsize=3); ax.axhline(0,color="black",lw=.8); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("AUC gain over baseline"); ax.set_title("Do new metrics improve pressure prediction?"); takeaway(fig,"Positive bars add out-of-game predictive information beyond standard rush and pocket controls."); fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(FIGURE_DIR/"pressure_auc_ablation.png",dpi=160); plt.close(fig)

    # SHAP importance.
    sh=pd.read_csv(TABLE_DIR/"pmoe_shap.csv").head(15); feature=sh.iloc[:,0].astype(str); val=pd.to_numeric(sh.meanAbsSHAP,errors="coerce").fillna(0); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(feature[::-1],val[::-1],color="#009E73"); ax.set_xlabel("Mean |SHAP value|"); ax.set_title("PMOE model feature importance"); takeaway(fig,"Larger values influence expected room gain more; importance is predictive, not causal."); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(FIGURE_DIR/"pmoe_shap_summary.png",dpi=160); plt.close(fig)

    # Sensitivity heatmap at default closing threshold.
    se=pd.read_csv(TABLE_DIR/"sensitivity.csv"); se=se[np.isclose(se.closing,THREAT_CLOSING_SPEED_YPS)]; piv=se.pivot(index="cone",columns="distance",values="difference"); fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(piv,cmap="coolwarm",aspect="auto"); ax.set_xticks(range(len(piv.columns)),piv.columns); ax.set_yticks(range(len(piv.index)),piv.index); ax.set(xlabel="Threat distance (yd)",ylabel="View-cone half-angle (°)",title="Sensitivity: pressure minus no-pressure UPT"); fig.colorbar(im,ax=ax,label="UPT difference (s)"); takeaway(fig,"Stable sign across cells means the main contrast is not driven by one threshold."); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(FIGURE_DIR/"sensitivity_heatmap.png",dpi=160); plt.close(fig)

    # Mediation-style association diagram with estimated unadjusted correlations.
    vals=pm[["RIS","UPT","reactionLag","PMOE","pressure"]].corr(numeric_only=True); fig,ax=plt.subplots(figsize=(10,3)); ax.axis("off"); nodes=[("Disguise\nRIS",.08),("Unseen pressure\nUPT",.32),("Reaction\nlag",.54),("Movement\nPMOE",.75),("PFF\npressure",.94)];
    for text,x0 in nodes: ax.text(x0,.55,text,ha="center",va="center",bbox=dict(boxstyle="round",fc="#E8F1F8",ec=OFF),transform=ax.transAxes)
    keys=[("RIS","UPT"),("UPT","reactionLag"),("reactionLag","PMOE"),("PMOE","pressure")]
    for j,(a,b) in enumerate(keys): x1=nodes[j][1]+.07; x2=nodes[j+1][1]-.07; ax.annotate("",xy=(x2,.55),xytext=(x1,.55),xycoords="axes fraction",arrowprops=dict(arrowstyle="->")); ax.text((x1+x2)/2,.65,f"r={vals.loc[a,b]:.2f}",ha="center",transform=ax.transAxes)
    ax.set_title("Association pathway (descriptive, not a causal mediation claim)"); takeaway(fig,"The diagram organizes measured associations; unmeasured play design prevents causal interpretation."); fig.tight_layout(rect=[0,.08,1,1]); fig.savefig(FIGURE_DIR/"mediation_style_diagram.png",dpi=160); plt.close(fig)
    stage_summary("STAGE 4 COMPLETE",[f"Rendered {len(showcases)} GIF showcase replays with field geometry and live metrics.","Saved eight static figures covering reveal, QB/defense summaries, survival, ablation, SHAP, sensitivity, and pathway associations.","Every figure includes a plain-English takeaway; captions explain objective showcase selection."])

if __name__=="__main__": main()
