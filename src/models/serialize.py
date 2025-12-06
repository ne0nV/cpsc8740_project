from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

def save_lightfm(model, user_map: dict[int, int], item_map: dict[int, int], model_path: str | Path, maps_path: str | Path):
    if joblib is None:
        raise ImportError("joblib not available. Install with `pip install joblib`.")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    maps = {
        "user_map": {int(k): int(v) for k, v in user_map.items()},
        "item_map": {int(k): int(v) for k, v in item_map.items()},
    }
    Path(maps_path).write_text(json.dumps(maps))

def load_lightfm(model_path: str | Path, maps_path: str | Path):
    if joblib is None:
        raise ImportError("joblib not available. Install with `pip install joblib`.")
    model = joblib.load(model_path)
    maps = json.loads(Path(maps_path).read_text())
    return model, {int(k): int(v) for k, v in maps["user_map"].items()}, {int(k): int(v) for k, v in maps["item_map"].items()}
