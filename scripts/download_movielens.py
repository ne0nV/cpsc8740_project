# Convenience script to download MovieLens 100k locally.
# Requires internet access on your machine.
from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import zipfile, io, requests, pandas as pd
from pathlib import Path
from src.utils.config import DATA_DIR

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "ml-100k"
    if out.exists():
        print("MovieLens 100k already present.")
        return
    print("Downloading MovieLens 100k ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)
    print(f"Extracted to {out.parent}")

if __name__ == "__main__":
    main()
