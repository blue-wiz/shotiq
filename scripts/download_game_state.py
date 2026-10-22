"""Step 3 (research extension): rebuild the score and red cards before every shot."""
import sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotiq.game_state import download_states

if __name__ == "__main__":
    t = time.time()
    s = download_states(ROOT / "data" / "raw" / "game_states.parquet")
    print(f"{len(s):,} shot states from {s.match_id.nunique():,} matches in {time.time()-t:.0f}s")
