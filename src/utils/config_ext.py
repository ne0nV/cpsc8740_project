# Extends the base config with LightFM artifact paths.
from pathlib import Path
from .config import ARTIFACTS_DIR

LIGHTFM_MODEL_PATH = ARTIFACTS_DIR / "lightfm_model.joblib"
LIGHTFM_MAPS_PATH = ARTIFACTS_DIR / "lightfm_maps.json"
