# Data audit

## Ten-line plain-English summary
1. The dataset contains 122 games and 8,557 passing plays from Weeks 1–8 of the 2021 season.
2. All 122 tracking files opened successfully and contain 8,314,178 rows sampled at 10 frames per second.
3. Every tracked frame has 22 players plus one football; PFF scouting has exactly 22 player rows per play.
4. There are 60 quarterbacks and 32 defenses; the PFF `Pass` role identifies one quarterback per play.
5. A manual `ball_snap` exists for 8,532 plays; 25 need the documented automatic-snap/first-frame fallback.
6. Most plays contain pre-snap tracking: median 5 frames (0.5 seconds) before the snap.
7. Player body orientation is complete and bounded; its 0 invalid/missing values are 0.000% of player rows.
8. Direction follows the NFL convention: 0° is +y and angles increase clockwise; after flipping leftward plays, quarterback dropbacks point near 270°.
9. Outcomes include 543 sack-coded plays, 449 scramble-coded plays, about 19 descriptions mentioning throws away, and 756 plays with penalty yards.
10. Models use only snap-to-throw/sack frames; PFF pressure labels are outcomes and are never included as input features.

## Files and rows
- `games.csv`: 122 rows
- `plays.csv`: 8,557 rows
- `players.csv`: 1,679 rows
- `pffScoutingData.csv`: 188,254 rows
- Tracking files: 122, 8,314,178 rows total; range 48,323–93,334 rows/file.

## Schemas, dtypes, and missing rates
### games

| column | dtype | missing_rate | unique |
|---|---:|---:|---:|
| gameId | int64 | 0.000% | 122 |
| season | int64 | 0.000% | 1 |
| week | int64 | 0.000% | 8 |
| gameDate | object | 0.000% | 24 |
| gameTimeEastern | object | 0.000% | 6 |
| homeTeamAbbr | object | 0.000% | 32 |
| visitorTeamAbbr | object | 0.000% | 32 |

### plays

| column | dtype | missing_rate | unique |
|---|---:|---:|---:|
| gameId | int64 | 0.000% | 122 |
| playId | int64 | 0.000% | 3,762 |
| playDescription | object | 0.000% | 8,555 |
| quarter | int64 | 0.000% | 5 |
| down | int64 | 0.000% | 5 |
| yardsToGo | int64 | 0.000% | 32 |
| possessionTeam | object | 0.000% | 32 |
| defensiveTeam | object | 0.000% | 32 |
| yardlineSide | object | 1.461% | 33 |
| yardlineNumber | int64 | 0.000% | 50 |
| gameClock | object | 0.000% | 898 |
| preSnapHomeScore | int64 | 0.000% | 42 |
| preSnapVisitorScore | int64 | 0.000% | 38 |
| passResult | object | 0.000% | 5 |
| penaltyYards | float64 | 91.165% | 61 |
| prePenaltyPlayResult | int64 | 0.000% | 98 |
| playResult | int64 | 0.000% | 102 |
| foulName1 | object | 91.399% | 30 |
| foulNFLId1 | float64 | 91.399% | 487 |
| foulName2 | object | 99.649% | 16 |
| foulNFLId2 | float64 | 99.649% | 31 |
| foulName3 | object | 99.988% | 2 |
| foulNFLId3 | float64 | 99.988% | 2 |
| absoluteYardlineNumber | float64 | 0.012% | 100 |
| offenseFormation | object | 0.082% | 8 |
| personnelO | object | 0.012% | 31 |
| defendersInBox | float64 | 0.082% | 12 |
| personnelD | object | 0.012% | 30 |
| dropBackType | object | 6.170% | 9 |
| pff_playAction | int64 | 0.000% | 2 |
| pff_passCoverage | object | 0.000% | 12 |
| pff_passCoverageType | object | 0.000% | 3 |

### players

| column | dtype | missing_rate | unique |
|---|---:|---:|---:|
| nflId | int64 | 0.000% | 1,679 |
| height | object | 0.000% | 16 |
| weight | int64 | 0.000% | 176 |
| birthDate | object | 13.818% | 1,175 |
| collegeName | object | 13.341% | 237 |
| officialPosition | object | 0.000% | 19 |
| displayName | object | 0.000% | 1,671 |

### PFF

| column | dtype | missing_rate | unique |
|---|---:|---:|---:|
| gameId | int64 | 0.000% | 122 |
| playId | int64 | 0.000% | 3,762 |
| nflId | int64 | 0.000% | 1,679 |
| pff_role | object | 0.000% | 5 |
| pff_positionLinedUp | object | 0.000% | 56 |
| pff_hit | float64 | 50.000% | 3 |
| pff_hurry | float64 | 50.000% | 3 |
| pff_sack | float64 | 50.000% | 3 |
| pff_beatenByDefender | float64 | 74.456% | 3 |
| pff_hitAllowed | float64 | 74.456% | 3 |
| pff_hurryAllowed | float64 | 74.456% | 3 |
| pff_sackAllowed | float64 | 74.456% | 3 |
| pff_nflIdBlockedPlayer | float64 | 75.286% | 666 |
| pff_blockType | object | 74.554% | 13 |
| pff_backFieldBlock | float64 | 74.554% | 3 |

### tracking

All tracking files share this schema: `gameId, playId, nflId, frameId, time, jerseyNumber, team, playDirection, x, y, s, a, dis, o, dir, event`.

Missing `nflId`, jersey, orientation, and direction correspond to the football (361,486 rows). Player orientation invalid/missing count: 0.

## Requested categorical values

- **passResult:** C=4,620; I=2,755; S=543; R=449; IN=190
- **offenseFormation:** SHOTGUN=5,481; EMPTY=1,396; SINGLEBACK=1,189; I_FORM=298; PISTOL=154; JUMBO=30; <missing>=7; WILDCAT=2
- **dropBackType:** TRADITIONAL=6,542; SCRAMBLE=899; <missing>=528; DESIGNED_ROLLOUT_RIGHT=285; DESIGNED_ROLLOUT_LEFT=149; SCRAMBLE_ROLLOUT_RIGHT=125; SCRAMBLE_ROLLOUT_LEFT=23; DESIGNED_RUN=5; UNKNOWN=1
- **pff_passCoverage:** Cover-3=2,665; Cover-1=2,011; Cover-2=1,085; Quarters=1,033; Cover-6=805; Red Zone=376; Cover-0=270; 2-Man=200; Bracket=47; Prevent=32; Goal Line=25; Miscellaneous=8
- **pff_passCoverageType:** Zone=5,588; Man=2,481; Other=488
- **pff_role:** Coverage=57,765; Pass Block=46,057; Pass Route=39,513; Pass Rush=36,362; Pass=8,557
- **pff_positionLinedUp:** QB=8,557; C=8,557; LT=8,557; LG=8,557; RG=8,557; RT=8,555; RCB=8,224; LCB=8,159; LWR=8,134; RWR=8,033; ROLB=5,325; LOLB=5,070; DRT=4,812; LEO=4,659; DLT=4,371; REO=4,287; LLB=4,202; FSL=3,793; FSR=3,764; RLB=3,627; SLWR=3,546; FS=3,512; RILB=3,490; SRWR=3,468; SCBR=3,422; LILB=3,336; SCBL=3,246; LE=3,204; HB-L=2,870; HB-R=2,676; RE=2,383; TE-R=2,241; SRoWR=1,820; SRiWR=1,820; SCBoL=1,690; TE-L=1,675; HB=1,624; MLB=1,595; SLoWR=1,432; SLiWR=1,432; SCBoR=1,342; NT=1,255; SCBiL=1,125; SCBiR=906; SSL=799; SSR=758; NRT=752; NLT=593; TE-oR=487; TE-iR=487; SS=424; TE-oL=292; TE-iL=292; FB-R=176; FB=147; FB-L=137
- **pff_blockType:** <missing>=140,350; PP=24,697; PA=6,331; PT=5,906; SW=3,025; CL=2,595; CH=1,565; NB=1,352; PU=886; SR=501; BH=443; UP=308; PR=295

- **event frames:** None=333,624; ball_snap=8,532; pass_forward=7,548; autoevent_ballsnap=3,767; autoevent_passforward=3,734; play_action=1,977; run=474; qb_sack=451; pass_arrived=367; autoevent_passinterrupted=201; man_in_motion=177; line_set=140; shift=133; pass_tipped=111; first_contact=80; qb_strip_sack=58; pass_outcome_incomplete=40; pass_outcome_caught=23; fumble=17; fumble_offense_recovered=11; handoff=10; huddle_break_offense=3; tackle=3; penalty_flag=2; lateral=1; dropped_pass=1; out_of_bounds=1

## Critical checks
- **Pre-snap:** yes. Median pre-snap frames=5; manual snap=8,532, auto fallback=1, first-frame fallback=24.
- **Frames/events:** frames per play median=39, range=19–203; labeled event frames/play median=3.
- **Orientation:** `o` is shoulder-pad body-facing direction, not gaze. It is missing for the football only in the full scan; extreme frame-to-frame changes are smoothed circularly over 3 frames.
- **Angle convention validation:** after normalization, median moving-QB direction in the first post-snap second is 277.1°. The expected dropback direction is near 270° under 0°=+y, clockwise.
- **Sacks/scrambles/throwaways/penalties:** `passResult` codes S and R identify sacks and scrambles; `qb_sack`/`qb_strip_sack` end tracking windows; throwaways require description text because there is no dedicated field; penalties use penalty/foul columns and may include no-plays.
- **Time:** timestamps have whole-second precision and occasional football-row disagreement; `frameId` is authoritative.
- **Outliers:** a small number of coordinates leave nominal field bounds and speed/acceleration have extreme values; rows are retained and quality-flagged rather than silently clipped.

## Join keys
- games → plays/tracking: `gameId`
- plays → tracking/PFF: `gameId + playId`
- PFF → tracking: `gameId + playId + nflId`
- players → tracking/PFF: `nflId`
- frame identity: `gameId + playId + frameId`; football is identified by `team == football`.

## Useful columns and use decisions
Used: identifiers; frameId/event; x/y/s/a/dis/o/dir; playDirection; PFF role/alignment/pressure outcomes; LOS, down, yards-to-go, formation/personnel, box count, dropback, play action, coverage; QB/player identity and defense; week and teams.
Outcome-only: PFF hit/hurry/sack and play result are never features. Penalty/foul fields are audit/quality flags. Names, height, weight, position, scores, clock, and date support labels/context and leaderboards but are not core geometry features. nflverse enrichment was skipped because local data are complete and external EPA/air-yards are optional.