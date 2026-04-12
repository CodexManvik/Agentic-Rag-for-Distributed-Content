"""Configuration management."""

from .adaptive import AdaptiveConfig, AdaptiveConfigGenerator

# Re-export settings from the config module at the parent level
# This resolves the ambiguity between app/config.py and app/config/ package
import sys
import importlib.util
from pathlib import Path

# Load the config.py module directly
_config_py_path = Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_app_config_module", _config_py_path)
_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_module)

# Export settings from the loaded module
settings = _config_module.settings
Settings = _config_module.Settings

__all__ = ["AdaptiveConfig", "AdaptiveConfigGenerator", "settings", "Settings"]
