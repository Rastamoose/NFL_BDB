"""Stage 1: normalize coordinates/angles and cache one Parquet table per game."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import *

KEEP_PLAY = ["gameId", "playId", "absoluteYardlineNumber", "possessionTeam", "defensiveTeam",
             "down", "yardsToGo", "offenseFormation", "personnelO", "defendersInBox",
             "personnelD", "dropBackType", "pff_playAction", "pff_passCoverage",
             "pff_passCoverageType", "passResult", "playResult", "penaltyYards"]

def smooth_grouped(df, col):
    rad = np.radians(df[col].astype(float))
    keys = [df["playId"], df["entity"]]
    ss = pd.Series(np.sin(rad), index=df.index).groupby(keys, sort=False).transform(
        lambda x: x.rolling(ANGLE_SMOOTH_FRAMES, center=True, min_periods=1).mean())
    cc = pd.Series(np.cos(rad), index=df.index).groupby(keys, sort=False).transform(
        lambda x: x.rolling(ANGLE_SMOOTH_FRAMES, center=True, min_periods=1).mean())
    return (np.degrees(np.arctan2(ss, cc)) + 360.0) % 360.0

def clean_one(path, plays, pff):
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in ["nflId", "jerseyNumber", "x", "y", "s", "a", "dis", "o", "dir"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["event"] = raw.event.fillna("None").astype("string")
    raw["entity"] = raw.nflId.round().astype("Int64").astype("string").fillna("football")

    # Normalize every offense to attack toward +x. The stated transformation also mirrors y.
    left = raw.playDirection.eq("left")
    raw.loc[left, "x"] = FIELD_LENGTH - raw.loc[left, "x"]
    raw.loc[left, "y"] = FIELD_WIDTH - raw.loc[left, "y"]
    raw.loc[left & raw.o.notna(), "o"] = (raw.loc[left & raw.o.notna(), "o"] + 180.0) % 360.0
    raw.loc[left & raw.dir.notna(), "dir"] = (raw.loc[left & raw.dir.notna(), "dir"] + 180.0) % 360.0

    raw = raw.sort_values(["playId", "entity", "frameId"]).reset_index(drop=True)
    raw["o_smooth"] = smooth_grouped(raw, "o")
    raw["dir_gated"] = raw.dir.where(raw.s.ge(DIRECTION_SPEED_GATE_YPS))
    raw["dir_smooth"] = smooth_grouped(raw, "dir_gated")
    raw["vx"], raw["vy"] = velocity(raw.s.fillna(0).to_numpy(), raw.dir_smooth.fillna(0).to_numpy())
    raw.loc[raw.dir_smooth.isna(), ["vx", "vy"]] = 0.0

    game_id = int(raw.gameId.iloc[0])
    game_plays = plays[plays.gameId.eq(game_id)][KEEP_PLAY]
    roles = pff[pff.gameId.eq(game_id)][["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"]]
    raw = raw.merge(game_plays, on=["gameId", "playId"], how="left", validate="many_to_one")
    raw = raw.merge(roles, on=["gameId", "playId", "nflId"], how="left", validate="many_to_one")
    raw["pff_role"] = raw.pff_role.fillna("Football").astype("category")
    raw["pff_positionLinedUp"] = raw.pff_positionLinedUp.fillna("BALL").astype("category")
    raw["is_offense"] = raw.team.eq(raw.possessionTeam)
    raw["is_defense"] = raw.team.eq(raw.defensiveTeam)

    windows = []
    for pid, g in raw.groupby("playId", sort=False):
        snap, end, source = find_snap_end(g)
        windows.append({"gameId": game_id, "playId": int(pid), "snapFrame": snap,
                        "endFrame": end, "snapSource": source,
                        "preSnapFrames": snap - int(g.frameId.min()),
                        "windowFrames": end - snap + 1})
    win = pd.DataFrame(windows)
    raw = raw.merge(win[["gameId", "playId", "snapFrame", "endFrame"]], on=["gameId", "playId"], how="left")
    raw["timeSinceSnap"] = (raw.frameId - raw.snapFrame) / FPS
    raw["inWindow"] = raw.frameId.between(raw.snapFrame, raw.endFrame)
    raw["xFromLOS"] = raw.x - raw.absoluteYardlineNumber
    raw["qualityOutOfBounds"] = ~raw.x.between(0, FIELD_LENGTH) | ~raw.y.between(0, FIELD_WIDTH)

    # Downcast high-volume fields before Parquet serialization.
    for col in ["frameId", "snapFrame", "endFrame"]:
        raw[col] = pd.to_numeric(raw[col], downcast="integer")
    for col in ["x", "y", "s", "a", "dis", "o", "dir", "o_smooth", "dir_smooth", "vx", "vy", "timeSinceSnap", "xFromLOS"]:
        raw[col] = pd.to_numeric(raw[col], downcast="float")
    raw["team"] = raw.team.astype("category")
    return raw, win

def main():
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")
    windows = []
    files = tracking_files()
    for i, path in enumerate(files, 1):
        game_id = path.stem.replace("tracking_", "")
        out = CLEAN_DIR / f"clean_{game_id}.parquet"
        win_path = CLEAN_DIR / f"windows_{game_id}.parquet"
        if out.exists() and win_path.exists() and not FORCE_REBUILD:
            windows.append(pd.read_parquet(win_path))
        else:
            clean, win = clean_one(path, plays, pff)
            clean.to_parquet(out, index=False, compression="zstd")
            win.to_parquet(win_path, index=False)
            windows.append(win)
        if i % 10 == 0 or i == len(files): print(f"  cleaned {i}/{len(files)} games")
    all_windows = pd.concat(windows, ignore_index=True)
    all_windows.to_parquet(TABLE_DIR / "play_windows.parquet", index=False)
    stage_summary("STAGE 1 COMPLETE", [
        f"Normalized and cached {len(files)} games as per-game Parquet files.",
        "Leftward plays now attack +x; orientation and motion direction use 3-frame circular smoothing.",
        "Every row is marked pre/post-window so downstream features cannot use information after throw or sack.",
    ])

if __name__ == "__main__": main()
