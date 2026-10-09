"""Central configuration for the NFL unseen-pressure project."""
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

def _find_data_dir() -> Path:
    candidates = [REPO_DIR, REPO_DIR / "data", PROJECT_DIR]
    for base in candidates:
        if (base / "games.csv").exists() and list(base.rglob("tracking_*.csv")):
            return base
    for games in REPO_DIR.rglob("games.csv"):
        if list(games.parent.rglob("tracking_*.csv")):
            return games.parent
    raise FileNotFoundError("Could not locate games.csv and tracking_*.csv under repository")

DATA_DIR = Path(os.getenv("NFL_DATA_DIR", _find_data_dir())).resolve()
TRACKING_DIR = DATA_DIR / "tracking" if (DATA_DIR / "tracking").exists() else DATA_DIR
CACHE_DIR = PROJECT_DIR / "cache"
CLEAN_DIR = CACHE_DIR / "clean_games"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
ANIMATION_DIR = OUTPUT_DIR / "animations"
for path in [CACHE_DIR, CLEAN_DIR, TABLE_DIR, FIGURE_DIR, ANIMATION_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 2026
FPS = 10.0
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3

# Core definitions; edit here to run a different research specification.
VIEW_CONE_HALF_ANGLE = 75.0
THREAT_DISTANCE_YARDS = 7.0
THREAT_CLOSING_SPEED_YPS = 1.5
REACTION_TURN_DEGREES = 20.0
REACTION_MOVE_AWAY_YARDS = 0.75
REACTION_AWAY_SPEED_YPS = 1.0
DIRECTION_SPEED_GATE_YPS = 1.0
ANGLE_SMOOTH_FRAMES = 3
ROOM_RADIUS_YARDS = 6.0
ROOM_PROJECTION_SECONDS = 0.5
ROOM_ANGLE_RAYS = 24
PMOE_HORIZON_SECONDS = 0.5
RUSH_REVEAL_PROBABILITY = 0.90
MIN_LEADERBOARD_SNAPS = 50

SENSITIVITY_CONES = [45, 60, 75, 90]
SENSITIVITY_DISTANCES = [5, 7, 9]
SENSITIVITY_CLOSING_SPEEDS = [1.0, 1.5, 2.5]
GROUP_FOLDS = 3
BOOTSTRAP_REPS = 50
FRAME_MODEL_MAX_ROWS = 100_000
ANIMATION_MAX_FRAMES = 80
ANIMATION_FPS = 8
SHOWCASE_COUNT = 6

# Set NFL_MAX_GAMES for a smoke run; default processes the full dataset.
MAX_GAMES = int(os.getenv("NFL_MAX_GAMES", "0")) or None
FORCE_REBUILD = os.getenv("NFL_FORCE_REBUILD", "0") == "1"
