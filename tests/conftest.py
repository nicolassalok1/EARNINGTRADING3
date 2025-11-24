import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / "sources"

# Mirror the sys.path injection used in the app so rl4f modules can be imported.
RL4F_DIRS = [
    SOURCES_DIR / "rl4f" / "trading_dql",
    SOURCES_DIR / "rl4f" / "hedging_dql",
    SOURCES_DIR / "rl4f" / "allocation_3ac",
]
for path in RL4F_DIRS:
    if path.exists():
        sys.path.insert(0, str(path))

# Add project root for importing streamlit_app helpers.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
