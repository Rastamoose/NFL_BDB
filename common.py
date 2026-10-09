"""Shared geometry, I/O, and football-data helpers."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from config import *

KEY = ["gameId", "playId"]

def tracking_files():
    files = sorted(TRACKING_DIR.rglob("tracking_*.csv"))
    return files[:MAX_GAMES] if MAX_GAMES else files

def angle_diff(a, b):
    """Signed shortest angle a-b in degrees, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0

def bearing(dx, dy):
    """NGS bearing: 0° points +y and angles increase clockwise toward +x."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

def velocity(speed, direction):
    rad = np.radians(direction)
    return speed * np.sin(rad), speed * np.cos(rad)

def circular_smooth(values, window=ANGLE_SMOOTH_FRAMES):
    v = pd.Series(values, dtype="float64")
    r = np.radians(v)
    s = pd.Series(np.sin(r)).rolling(window, center=True, min_periods=1).mean()
    c = pd.Series(np.cos(r)).rolling(window, center=True, min_periods=1).mean()
    return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0

def voronoi_room(qx, qy, other_xy, radius=ROOM_RADIUS_YARDS, rays=ROOM_ANGLE_RAYS):
    """Area of QB Voronoi cell clipped to a circle, via radial half-plane integration."""
    other = np.asarray(other_xy, dtype=float)
    other = other[np.isfinite(other).all(axis=1)] if other.size else np.empty((0, 2))
    if len(other) == 0:
        return math.pi * radius * radius
    dx, dy = other[:, 0] - qx, other[:, 1] - qy
    d = np.hypot(dx, dy)
    keep = d > 1e-6
    dx, dy, d = dx[keep], dy[keep], d[keep]
    if len(d) == 0:
        return 0.0
    theta = np.arctan2(dy, dx)
    phi = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    cos_delta = np.cos(phi[:, None] - theta[None, :])
    limits = np.where(cos_delta > 1e-9, d[None, :] / (2 * cos_delta), np.inf)
    radial = np.minimum(radius, limits.min(axis=1))
    return float(0.5 * np.sum(radial ** 2) * (2 * np.pi / rays))

def voronoi_polygon(qx, qy, other_xy, radius=ROOM_RADIUS_YARDS, rays=72):
    """Boundary points matching voronoi_room, for animation shading."""
    other = np.asarray(other_xy, dtype=float)
    other = other[np.isfinite(other).all(axis=1)] if other.size else np.empty((0, 2))
    phi = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    radial = np.full(rays, radius, dtype=float)
    if len(other):
        dx, dy = other[:, 0] - qx, other[:, 1] - qy
        d = np.hypot(dx, dy); keep = d > 1e-6
        theta = np.arctan2(dy[keep], dx[keep]); d = d[keep]
        if len(d):
            cos_delta = np.cos(phi[:, None] - theta[None, :])
            limits = np.where(cos_delta > 1e-9, d[None, :] / (2 * cos_delta), np.inf)
            radial = np.minimum(radial, limits.min(axis=1))
    return np.column_stack([qx + radial * np.cos(phi), qy + radial * np.sin(phi)])

def pressure_label(pff_play):
    rush = pff_play[pff_play["pff_role"].eq("Pass Rush")]
    cols = ["pff_hit", "pff_hurry", "pff_sack"]
    return int(rush[cols].fillna(0).to_numpy().max(initial=0) > 0)

def find_snap_end(play):
    ev = play.groupby("frameId")["event"].first()
    manual = ev.index[ev.eq("ball_snap")].tolist()
    auto = ev.index[ev.eq("autoevent_ballsnap")].tolist()
    snap = manual[0] if manual else (auto[0] if auto else int(play.frameId.min()))
    candidates = ev.index[ev.isin(["pass_forward", "qb_sack", "qb_strip_sack"])].tolist()
    end = min([f for f in candidates if f >= snap], default=int(play.frameId.max()))
    return int(snap), int(end), "manual" if manual else ("auto" if auto else "first_frame")

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def stage_summary(stage, lines):
    print(f"\n[{stage}]")
    for line in lines[:3]:
        print(line)
