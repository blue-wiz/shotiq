"""Step 1: download real StatsBomb shots for the 2015/16 big-five leagues."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shotiq.data import download_shots

if __name__ == "__main__":
    t = time.time()
    shots = download_shots(Path(__file__).resolve().parents[1] / "data" / "raw")
    print(f"{len(shots):,} shots from {shots.match_id.nunique():,} matches in {time.time()-t:.0f}s")
    print(shots.groupby("competition").size())
