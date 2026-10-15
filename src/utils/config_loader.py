import os
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = os.path.join(base_dir, "config", "settings.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
