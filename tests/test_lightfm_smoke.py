def test_lightfm_paths_import():
    from src.utils.config_ext import LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH
    assert str(LIGHTFM_MODEL_PATH).endswith('lightfm_model.joblib')
    assert str(LIGHTFM_MAPS_PATH).endswith('lightfm_maps.json')
