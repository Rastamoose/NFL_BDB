# Methods

## Research question

"When a pass rush approaches from outside the quarterback's body-facing view cone, how long does the quarterback take to turn toward it and move, does a defense's disguise (rushers who looked like non-rushers before the snap, or who reveal their rush late) make that lag longer, and does the lag explain hurries, hits, and sacks?"

## Assumptions and data decisions

1. The shoulder-pad orientation sensor measures **body-facing**, not eyes or gaze. The view cone is therefore a reproducible proxy for what is in front of the quarterback, not proof of visual attention.
2. `frameId` is the clock at 10 Hz because source timestamps have only whole-second precision. Frames are keyed by `gameId + playId + frameId`, not timestamp.
3. The manual `ball_snap` event is preferred. If absent, `autoevent_ballsnap` is used; if both are absent, frame 1 is used and flagged. The 25 affected plays remain in the data.
4. The analysis window starts at snap and stops at the first manual `pass_forward`, `qb_sack`, or `qb_strip_sack`; otherwise it stops at the final frame. Raw post-window rows are excluded from metric calculations. The cached fast run used centered three-frame smoothing, so a terminal angle can borrow one adjacent post-window frame; this boundary issue is disclosed below rather than hidden.
5. Leftward plays are transformed exactly as specified: `x'=120-x`, `y'=53.3-y`, `o'=(o+180) mod 360`, `dir'=(dir+180) mod 360`. The line of scrimmage is `absoluteYardlineNumber`.
6. NFL angles use 0° along +y and increase clockwise. Thus a vector `(dx,dy)` has bearing `atan2(dx,dy)`. Direction is ignored below 1 yd/s; body and movement angles use a centered three-frame circular mean.
7. PFF roles define the quarterback (`Pass`), rushers (`Pass Rush`), blockers (`Pass Block`), and coverage defenders (`Coverage`). Exact source capitalization is used.
8. PFF hurry, hit, and sack are outcomes only. They never enter features. Pressure equals any of these three on a rusher row.
9. Penalty plays remain because the physical rush occurred, with a penalty flag available for robustness checks. Results should be rechecked excluding penalties before publication.
10. Missing categorical context is imputed inside training folds. Numeric medians and one-hot categories are also learned inside folds to avoid leakage.
11. The default run uses all games. `NFL_MAX_GAMES` exists only for smoke testing and must not be used for reported results.
12. Optional nflverse enrichment is omitted: all required tracking, role, and pressure targets are local, while external EPA/air-yard data are not necessary for the stated question.

## Metric definitions

Let frames be 0.1 s apart. Let quarterback position and velocity be `q(t), vq(t)`, rusher position and velocity be `r(t), vr(t)`, distance `d(t)=||q-r||`, and unit vector from rusher to QB `u=(q-r)/d`. Closing speed is `c(t)=(vr-vq)·u`. A rusher is a **threat** when `d≤7 yd` and `c≥1.5 yd/s`. Its bearing from the QB is `β=atan2(rx-qx, ry-qy)` under the NGS convention. It is unseen when the circular difference `|β-oQB|` exceeds the 75° view-cone half-angle.

- **Unseen Pressure Time (UPT):** `0.1 × Σt I(any unseen threat at t)` seconds. **Unseen Pressure Share** divides those frames by all observed window frames. **Max simultaneous unseen threats** is the largest unseen count in any frame.
- **Reaction Lag (RL):** time from the first unseen-threat frame to the first of: (a) circular body-bearing error falls by at least 20°, (b) QB displacement projected away from that initial threat reaches 0.75 yd, or (c) QB velocity projected away reaches 1 yd/s. If the window ends first, duration runs to the end and `reactionObserved=0`; this is right-censoring, not deletion.
- **Static QB room:** area of the QB's Voronoi cell clipped to a 6-yard circle. For speed, the clipped polygon is evaluated by 24 radial rays: along ray φ, each other player imposes boundary `d/[2 cos(φ-θ)]` when the cosine is positive; area is `½Σ r(φ)²Δφ`. Animation uses 72 rays.
- **Velocity-aware QB room:** the same Voronoi area after projecting every player by 0.5 s using `x'=x+vx×0.5`, `y'=y+vy×0.5`.
- **Pocket Movement Over Expected (PMOE):** observed static-room change from snap to 0.5 s minus an out-of-fold LightGBM prediction from rusher/blocker geometry, velocities, time/context, and down/distance. QB identity is excluded.
- **Rush Intent Surprise (RIS):** for defenders `j`, `RIS=Σj -log P(Yj=actual role)` using game-grouped out-of-fold snap predictions. `RIS_preSnap` applies the fitted snap model to each defender's last observed pre-snap frame; plays without pre-snap frames are missing, not copied from snap.
- **Rush Reveal Time (RRT):** first post-snap time at which a frame model assigns an eventual rusher `P(rush)≥0.90`. A rusher never crossing the threshold is flagged and assigned the observed window end; play RRT is the mean and maximum across rushers.
- **Disguise Drag (DD):** the adjusted fixed-effect coefficient of RIS or RRT in a mixed-effects model of RL (and separately PMOE), controlling for rusher/blocker counts, coverage type, play action, dropback type, down, and distance, with QB grouping and defense variance components. It is an association, not a causal effect.
- **Disguise Resistance / Yield:** empirically shrunk QB/defense mean reaction lag: `league_mean + n/(n+50)×(raw_mean-league_mean)`, requiring at least 50 qualifying snaps.
- **Baselines:** time to throw; first threat time; STRAIN=`max(0,closing speed)/max(distance,0.25)`; convex-hull area of blockers plus QB; static room; number of rushers; PFF pressure.

## Models and leakage controls

A snap rush-intent logistic baseline is compared with HistGradientBoosting and LightGBM. The frame intent model adds velocity, acceleration, alignment displacement, and orientation/direction toward the QB. PMOE uses LightGBM regression. RL uses a penalized Cox proportional-hazards model with censoring and Kaplan–Meier curves by RIS tertile. Pressure models compare baseline features with each new metric and all metrics together. Mixed-effects models estimate Disguise Drag. Primary validation is three-fold `GroupKFold` by `gameId`; PMOE and pressure also use Weeks 1–5 training versus Weeks 6–8 testing. Frames from one game never appear in both train and validation. Confidence intervals for pressure gains use 50 game-level bootstrap samples.

Angles enter models only through sine/cosine or cosine alignment. PFF outcomes are excluded from predictors. Snap RIS, pre-snap RIS, and frame-level RRT probabilities are cross-fitted by game; a prediction for a game is produced only by a model trained on other games. PMOE excludes QB identity. Model performance is predictive; SHAP describes model attribution, not causality.

## Sensitivity analysis

UPT is recomputed for cone half-angles 45/60/75/90°, threat distances 5/7/9 yd, and closing thresholds 1/1.5/2.5 yd/s (36 combinations). `sensitivity.csv` compares mean UPT in pressured and unpressured plays. A stable sign is more important than exact magnitude.

## Prior work and novelty

This project builds on public player-tracking pressure and disguise work, including **STRAIN**, the **PRSS pocket-collapse metric**, **Continuous Pocket Pressure/OPLE**, **PSOTE**, **Next Gen Stats Pressure Probability**, **Next Gen Stats coverage disguise**, pre-snap **defensive alerts**, and **Trench Chess**. Relevant public starting points are the [NFL Big Data Bowl 2023 competition and community papers](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023), the [NFL Football Operations analytics portal](https://operations.nfl.com/gameday/analytics/), and the [MIT Sloan Sports Analytics Conference research-paper archive](https://www.sloansportsconference.com/research-papers). The combination of body-facing unseen-threat exposure, censored response lag, intent surprise/reveal, and room-over-expected is, **to my knowledge, not previously published**. It is not claimed as “first ever.”

## Limitations

Body-facing is not gaze; a QB may perceive pressure peripherally or through learned timing. PFF roles and pressure are human labels. The Voronoi room is a 24-ray numerical approximation (72 rays in animations) and treats all nearby players symmetrically. Acceleration is a nonnegative magnitude, not signed longitudinal acceleration. A reveal threshold of 0.90 can censor ambiguous rushes. The fast cached preprocessing run used centered three-frame circular smoothing before window filtering, so the terminal frame can incorporate one later angle; 39 plays with only `autoevent_passforward` also use the file end. These small boundary cases should be rebuilt with causal smoothing before publication. Mixed models cannot remove unmeasured play design, receiver separation, quarterback reads, or blocking assignments, so “disguise drag” means controlled association. Only selected passing plays from Weeks 1–8 of 2021 are observed; findings may not generalize across seasons or rushing plays.
