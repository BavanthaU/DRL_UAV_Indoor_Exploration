import os
import importlib
import inspect
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def load_custom_feature_extractors():
    """
    Dynamically import all classes that inherit from BaseFeaturesExtractor
    in the RL_drone/models directory.
    """
    feature_extractor_map = {}
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    models_dir = os.path.abspath(models_dir)
    package_name = f"{__package__}.models"

    for file in os.listdir(models_dir):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = f"{package_name}.{file[:-3]}"
            try:
                module = importlib.import_module(module_name)

                # Register classes that inherit from BaseFeaturesExtractor
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseFeaturesExtractor) and obj is not BaseFeaturesExtractor:
                        feature_extractor_map[name] = obj
            except Exception as e:
                print(f"Could not import {module_name}: {e}")

    print(f"Loaded custom feature extractors: {list(feature_extractor_map.keys())}")
    return feature_extractor_map


def check_custom_feature_extractor_in_sb3_cfg(cfg: dict) -> dict:
    """
    Replace string-based feature extractor class names in SB3 config dict
    with actual Python class references.
    """
    feature_extractor_map = load_custom_feature_extractors()

    features_extractor_class = cfg["policy_kwargs"]["features_extractor_class"]

    if features_extractor_class in feature_extractor_map:
        cfg["policy_kwargs"]["features_extractor_class"] = feature_extractor_map[features_extractor_class]
    else:
        print(f" Custom feature extractor '{features_extractor_class}' not found in RL_drone/models/")
        print(f"   Available: {list(feature_extractor_map.keys())}")

    return cfg
