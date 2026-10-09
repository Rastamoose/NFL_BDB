"""Stage 2: compute frame/play geometry, UPT, reaction lag, room, and model tables."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from common import *

warnings.filterwarnings("ignore", category=RuntimeWarning)

def hull_area(xy):
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3: return 0.0
    try: return float(ConvexHull(xy).volume)
    except Exception: return 0.0

def one_play(g, pmeta, pff_play):
    game_id, play_id = int(g.gameId.iloc[0]), int(g.playId.iloc[0])
    snap, end = int(g.snapFrame.iloc[0]), int(g.endFrame.iloc[0])
    w = g[g.inWindow].copy()
    qrows = w[w.pff_role.astype(str).eq("Pass")]
    if qrows.empty: return None
    qid = int(qrows.nflId.iloc[0])
    rush_ids = set(pff_play.loc[pff_play.pff_role.eq("Pass Rush"), "nflId"].astype(int))
    block_ids = set(pff_play.loc[pff_play.pff_role.eq("Pass Block"), "nflId"].astype(int))
    pressure = pressure_label(pff_play)
    hit = int(pff_play.pff_hit.fillna(0).max() > 0)
    hurry = int(pff_play.pff_hurry.fillna(0).max() > 0)
    sack = int(pff_play.pff_sack.fillna(0).max() > 0 or pmeta.passResult == "S")

    frame_rows, rusher_rows, defender_rows = [], [], []
    first_unseen = None; responded = False; rl_frame = end
    upt_frames = 0; max_unseen = 0; first_threat_frame = None
    rooms = {}; vrooms = {}; pockets = {}; strains = []
    snap_def = w[(w.frameId.eq(snap)) & w.is_defense].copy()
    snap_pos = snap_def.set_index("nflId")[["x", "y"]].to_dict("index")

    for frame, fr in w.groupby("frameId", sort=True):
        qb = fr[fr.nflId.eq(qid)]
        if qb.empty: continue
        qb = qb.iloc[0]
        others = fr[fr.nflId.notna() & ~fr.nflId.eq(qid)]
        rush = fr[fr.nflId.isin(rush_ids)].copy()
        unseen_count = 0; nearest = np.nan; max_strain = 0.0
        threat_rows = []
        for _, r in rush.iterrows():
            dx, dy = qb.x - r.x, qb.y - r.y
            dist = float(np.hypot(dx, dy)); ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (0, 0)
            closing = float((r.vx - qb.vx) * ux + (r.vy - qb.vy) * uy)
            rbearing = float(bearing(r.x - qb.x, r.y - qb.y))
            rel_angle = float(abs(angle_diff(rbearing, qb.o_smooth)))
            threat = dist <= THREAT_DISTANCE_YARDS and closing >= THREAT_CLOSING_SPEED_YPS
            unseen = threat and rel_angle > VIEW_CONE_HALF_ANGLE
            strain = max(0.0, closing) / max(dist, 0.25)
            max_strain = max(max_strain, strain)
            nearest = dist if np.isnan(nearest) else min(nearest, dist)
            if threat and first_threat_frame is None: first_threat_frame = int(frame)
            if unseen:
                unseen_count += 1
                threat_rows.append((dist, int(r.nflId), rbearing, rel_angle, r.x, r.y))
            rusher_rows.append({"gameId": game_id, "playId": play_id, "frameId": int(frame),
                "nflId": int(r.nflId), "distance": dist, "closingSpeed": closing,
                "relativeAngle": rel_angle, "threat": int(threat), "unseen": int(unseen),
                "strain": strain, "pressure": pressure})
        strains.append(max_strain)
        if unseen_count: upt_frames += 1
        max_unseen = max(max_unseen, unseen_count)

        # Start reaction clock with nearest unseen threat; keep the same rusher thereafter.
        if first_unseen is None and threat_rows:
            tr = min(threat_rows)
            away_dx, away_dy = qb.x - tr[4], qb.y - tr[5]
            norm = max(np.hypot(away_dx, away_dy), 1e-6)
            first_unseen = {"frame": int(frame), "rusher": tr[1], "initial_error": tr[3],
                            "qx": qb.x, "qy": qb.y, "away_x": away_dx/norm, "away_y": away_dy/norm}
        if first_unseen is not None and not responded and frame >= first_unseen["frame"]:
            rr = fr[fr.nflId.eq(first_unseen["rusher"])]
            if not rr.empty:
                rr = rr.iloc[0]
                rb = bearing(rr.x - qb.x, rr.y - qb.y)
                current_error = abs(angle_diff(rb, qb.o_smooth))
                turned = current_error <= max(0, first_unseen["initial_error"] - REACTION_TURN_DEGREES)
                qmove = (qb.x-first_unseen["qx"])*first_unseen["away_x"] + (qb.y-first_unseen["qy"])*first_unseen["away_y"]
                away_component = qb.vx*first_unseen["away_x"] + qb.vy*first_unseen["away_y"]
                if turned or qmove >= REACTION_MOVE_AWAY_YARDS or away_component >= REACTION_AWAY_SPEED_YPS:
                    responded = True; rl_frame = int(frame)

        xy = others[["x", "y"]].to_numpy()
        room = voronoi_room(qb.x, qb.y, xy)
        projected = np.column_stack([others.x + others.vx*ROOM_PROJECTION_SECONDS,
                                     others.y + others.vy*ROOM_PROJECTION_SECONDS])
        qpx, qpy = qb.x + qb.vx*ROOM_PROJECTION_SECONDS, qb.y + qb.vy*ROOM_PROJECTION_SECONDS
        vroom = voronoi_room(qpx, qpy, projected)
        blocker_xy = fr[fr.nflId.isin(block_ids)][["x", "y"]].to_numpy()
        pocket = hull_area(np.vstack([blocker_xy, [qb.x, qb.y]]))
        rooms[int(frame)] = room; vrooms[int(frame)] = vroom; pockets[int(frame)] = pocket
        frame_rows.append({"gameId": game_id, "playId": play_id, "frameId": int(frame),
            "timeSinceSnap": (frame-snap)/FPS, "nearestRusherDistance": nearest,
            "unseenThreatCount": unseen_count, "uptSoFar": upt_frames/FPS,
            "qbRoom": room, "velocityAwareRoom": vroom, "pocketArea": pocket,
            "maxSTRAIN": max_strain, "event": str(fr.event.iloc[0])})

        # Vectorized frame-level rush-intent features for all 11 defenders.
        ball = fr[fr.team.astype(str).eq("football")]
        by = float(ball.y.iloc[0]) if len(ball) else FIELD_WIDTH/2
        defs = fr[fr.is_defense & fr.nflId.notna()].copy()
        if len(defs):
            dx=defs.x.to_numpy(float); dy=defs.y.to_numpy(float)
            db=bearing(float(qb.x)-dx,float(qb.y)-dy)
            sx=np.array([snap_pos.get(n,{"x":x})["x"] for n,x in zip(defs.nflId,dx)])
            sy=np.array([snap_pos.get(n,{"y":y})["y"] for n,y in zip(defs.nflId,dy)])
            dto=np.cos(np.radians(angle_diff(defs.dir_smooth.to_numpy(float),db)))
            dto=np.nan_to_num(dto,nan=0.0)
            batch=pd.DataFrame({"gameId":game_id,"playId":play_id,"frameId":int(frame),
                "nflId":defs.nflId.astype(int).to_numpy(),
                "actualRush":defs.nflId.astype(int).isin(rush_ids).astype(np.int8).to_numpy(),
                "timeSinceSnap":(frame-snap)/FPS,"xFromLOS":dx-float(pmeta.absoluteYardlineNumber),
                "yFromBall":dy-by,"distanceQB":np.hypot(dx-float(qb.x),dy-float(qb.y)),
                "speed":defs.s.to_numpy(float),"acceleration":defs.a.to_numpy(float),
                "displacement":np.hypot(dx-sx,dy-sy),
                "orientationToQB":np.cos(np.radians(angle_diff(defs.o_smooth.to_numpy(float),db))),
                "dirToQB":dto,"oSin":np.sin(np.radians(defs.o_smooth.to_numpy(float))),
                "oCos":np.cos(np.radians(defs.o_smooth.to_numpy(float))),
                "position":defs.pff_positionLinedUp.astype(str).to_numpy(),"pressure":pressure})
            defender_rows.extend(batch.to_dict("records"))

    if not frame_rows: return None
    start = min(rooms); horizon = min(end, snap + int(PMOE_HORIZON_SECONDS*FPS))
    if horizon not in rooms: horizon = min(rooms, key=lambda x: abs(x-horizon))
    snap_fr = w[w.frameId.eq(snap)]
    qb0 = snap_fr[snap_fr.nflId.eq(qid)].iloc[0]
    rush0 = snap_fr[snap_fr.nflId.isin(rush_ids)]
    block0 = snap_fr[snap_fr.nflId.isin(block_ids)]
    # Defender tables for snap RIS and a true last-pre-snap RIS variant.
    snap_rows = []
    pre_candidates = g.loc[g.frameId.lt(snap), "frameId"]
    phase_frames = [("snap", snap)]
    if len(pre_candidates): phase_frames.append(("preSnap", int(pre_candidates.max())))
    for phase, pf in phase_frames:
        phase_fr = g[g.frameId.eq(pf)]
        ball0 = phase_fr[phase_fr.team.astype(str).eq("football")]
        bx0 = float(ball0.x.iloc[0]) if len(ball0) else float(pmeta.absoluteYardlineNumber)
        by0 = float(ball0.y.iloc[0]) if len(ball0) else FIELD_WIDTH/2
        defenders0 = phase_fr[phase_fr.is_defense & phase_fr.nflId.notna()]
        for _, d in defenders0.iterrows():
            others0 = defenders0[~defenders0.nflId.eq(d.nflId)]
            spacing = float(np.min(np.hypot(others0.x-d.x, others0.y-d.y))) if len(others0) else 0.0
            snap_rows.append({"gameId": game_id, "playId": play_id, "nflId": int(d.nflId), "phase": phase,
                "actualRush": int(int(d.nflId) in rush_ids), "xFromLOS": d.x-pmeta.absoluteYardlineNumber,
                "yFromBall": d.y-by0, "distanceBall": float(np.hypot(d.x-bx0,d.y-by0)),
                "spacingNearest": spacing, "oSin": np.sin(np.radians(d.o_smooth)), "oCos": np.cos(np.radians(d.o_smooth)),
                "position": str(d.pff_positionLinedUp), "formation": str(pmeta.offenseFormation),
                "personnelO": str(pmeta.personnelO), "personnelD": str(pmeta.personnelD),
                "coverage": str(pmeta.pff_passCoverage), "coverageType": str(pmeta.pff_passCoverageType),
                "down": pmeta.down, "yardsToGo": pmeta.yardsToGo, "defendersInBox": pmeta.defendersInBox})

    first_frame = first_unseen["frame"] if first_unseen else np.nan
    rl = (rl_frame-first_unseen["frame"])/FPS if first_unseen else np.nan
    play_row = {"gameId": game_id, "playId": play_id, "qbId": qid, "defense": pmeta.defensiveTeam,
        "offense": pmeta.possessionTeam, "week": pmeta.week, "down": pmeta.down,
        "yardsToGo": pmeta.yardsToGo, "coverage": pmeta.pff_passCoverage,
        "coverageType": pmeta.pff_passCoverageType, "playAction": pmeta.pff_playAction,
        "dropBackType": pmeta.dropBackType, "formation": pmeta.offenseFormation,
        "numRushers": len(rush_ids), "numBlockers": len(block_ids),
        "windowSeconds": (end-snap)/FPS, "timeToThrow": (end-snap)/FPS,
        "timeToPressure": (first_threat_frame-snap)/FPS if first_threat_frame is not None else np.nan,
        "UPT": upt_frames/FPS, "unseenPressureShare": upt_frames/max(1,end-snap+1),
        "maxUnseenThreats": max_unseen, "firstUnseenFrame": first_frame,
        "reactionLag": rl, "reactionObserved": int(first_unseen is not None and responded),
        "reactionCensored": int(first_unseen is not None and not responded),
        "hasUnseenThreat": int(first_unseen is not None), "staticRoomSnap": rooms.get(snap, rooms[start]),
        "velocityRoomSnap": vrooms.get(snap, vrooms[start]), "staticRoomEnd": rooms.get(end, rooms[max(rooms)]),
        "velocityRoomEnd": vrooms.get(end, vrooms[max(vrooms)]),
        "roomGain05": rooms[horizon]-rooms.get(snap,rooms[start]),
        "velocityRoomGain05": vrooms[horizon]-vrooms.get(snap,vrooms[start]),
        "pocketAreaSnap": pockets.get(snap,pockets[start]), "pocketAreaMin": min(pockets.values()),
        "maxSTRAIN": max(strains, default=0), "nearestRusherSnap": float(np.min(np.hypot(rush0.x-qb0.x,rush0.y-qb0.y))) if len(rush0) else np.nan,
        "meanRusherSpeedSnap": rush0.s.mean(), "meanBlockerDistanceSnap": np.hypot(block0.x-qb0.x,block0.y-qb0.y).mean() if len(block0) else np.nan,
        "pressure": pressure, "hit": hit, "hurry": hurry, "sack": sack,
        "penalty": int(pd.notna(pmeta.penaltyYards)), "passResult": pmeta.passResult,
        "description": pmeta.playDescription}
    return play_row, frame_rows, rusher_rows, defender_rows, snap_rows

def process_game(path, meta_game, pff_game):
    """Worker-safe game processor used by joblib's Windows process pool."""
    d = pd.read_parquet(path)
    out = []
    for key, g in d.groupby(["gameId", "playId"], sort=False):
        if key not in meta_game.index or key not in pff_game:
            continue
        result = one_play(g, meta_game.loc[key], pff_game[key])
        if result:
            out.append(result)
    return out

def main():
    from joblib import Parallel, delayed
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = plays.merge(games[["gameId","week"]], on="gameId", how="left")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")
    clean_files = sorted(CLEAN_DIR.glob("clean_*.parquet"))
    if MAX_GAMES: clean_files = clean_files[:MAX_GAMES]
    tasks=[]
    for path in clean_files:
        gid=int(path.stem.replace("clean_",""))
        mg=plays[plays.gameId.eq(gid)].set_index(["gameId","playId"])
        pg={k:g for k,g in pff[pff.gameId.eq(gid)].groupby(["gameId","playId"],sort=False)}
        tasks.append((path,mg,pg))
    game_results=Parallel(n_jobs=min(6,len(tasks)),backend="loky",verbose=10)(
        delayed(process_game)(path,mg,pg) for path,mg,pg in tasks)
    play_rows=[]; frame_rows=[]; rusher_rows=[]; defender_rows=[]; snap_rows=[]
    for results in game_results:
        for pr,fr,rr,dr,sr in results:
            play_rows.append(pr); frame_rows.extend(fr); rusher_rows.extend(rr); defender_rows.extend(dr); snap_rows.extend(sr)
    pm=pd.DataFrame(play_rows); fm=pd.DataFrame(frame_rows); rm=pd.DataFrame(rusher_rows)
    dm=pd.DataFrame(defender_rows); sm=pd.DataFrame(snap_rows)
    pm.to_parquet(TABLE_DIR/"play_metrics_base.parquet",index=False)
    fm.to_parquet(TABLE_DIR/"frame_metrics.parquet",index=False)
    rm.to_parquet(TABLE_DIR/"rusher_frame_metrics.parquet",index=False)
    dm.to_parquet(TABLE_DIR/"defender_frame_features.parquet",index=False)
    sm.to_parquet(TABLE_DIR/"defender_snap_features.parquet",index=False)

    # Sensitivity table: play-level mean UPT under every threshold combination.
    sens=[]
    for cone in SENSITIVITY_CONES:
      for dist in SENSITIVITY_DISTANCES:
       for close in SENSITIVITY_CLOSING_SPEEDS:
        flag=(rm.distance.le(dist)&rm.closingSpeed.ge(close)&rm.relativeAngle.gt(cone))
        tmp=rm.assign(flag=flag).groupby(["gameId","playId","frameId"]).flag.max().groupby(["gameId","playId"]).sum()/FPS
        joined=pm.set_index(["gameId","playId"])[["pressure"]].join(tmp.rename("upt")).fillna({"upt":0})
        sens.append({"cone":cone,"distance":dist,"closing":close,"meanUPT":joined.upt.mean(),
                     "pressureUPT":joined.loc[joined.pressure.eq(1),"upt"].mean(),
                     "noPressureUPT":joined.loc[joined.pressure.eq(0),"upt"].mean(),
                     "difference":joined.loc[joined.pressure.eq(1),"upt"].mean()-joined.loc[joined.pressure.eq(0),"upt"].mean()})
    pd.DataFrame(sens).to_csv(TABLE_DIR/"sensitivity.csv",index=False)
    stage_summary("STAGE 2 COMPLETE",[
        f"Computed UPT, censored reaction lag, room, pocket, and STRAIN for {len(pm):,} plays.",
        f"Saved {len(fm):,} frame rows plus snap/frame defender tables for rush-intent modeling.",
        "Sensitivity results cover all 36 cone/distance/closing-speed definitions."])

if __name__=="__main__": main()
