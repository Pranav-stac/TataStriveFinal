"""
Configuration Manager for TataStrive Analytics.
Handles persistent settings storage in JSON format.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """Manages application configuration with JSON persistence."""
    
    DEFAULT_CONFIG = {
        "center_id": "",           # Set on first launch via CenterDialog
        "last_video_path": "",
        "last_video_folder": "",
        "last_output_dir": "",
        "last_db_path": "",
        "bigquery": {
            "auto_sync": True,         # Trigger daily sync automatically
            "sync_hour": 0,            # Hour of day (UTC) to auto-sync (0 = midnight)
            "last_sync_date": "",      # ISO date of last successful sync
        },
        "classroom": {
            "probe_duration": 300,
            "probe_interval": 3600,
            "frame_skip": 3,
            "similarity_threshold": 0.75,
            "max_time_gap": 600,
            "max_pixel_dist": 200
        },
        "crossday": {
            "t_strict_merge": 0.55,
            "t_new_id": 0.35,
            "t_ratio_margin": 0.10,
            "min_samples": 8,
            "max_exemplars": 5,
            "t_outlier": 0.6,
            "visitor_upgrade_days": 3,
            "save_output_video": True
        },
        "inference": {
            "use_openvino": True,
            "yolo_imgsz": 416,
            "face_det_size": 416,
            "frame_skip": 1,
            "preview_mode": "cv2"
        },
        "preview_enabled": False,
        "window": {
            "width": 1200,
            "height": 800,
            "x": 100,
            "y": 100
        }
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional custom config directory. Defaults to ~/.tatastrive/
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".tatastrive"
        
        self.config_file = self.config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from file or create default."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Merge with defaults to handle new keys
                self._config = self._deep_merge(self.DEFAULT_CONFIG.copy(), loaded)
            else:
                self._config = self.DEFAULT_CONFIG.copy()
                self._save()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config: {e}. Using defaults.")
            self._config = self.DEFAULT_CONFIG.copy()
    
    def _save(self) -> None:
        """Save configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
        except IOError as e:
            print(f"Warning: Could not save config: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "classroom.probe_duration")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "classroom.probe_duration")
            value: Value to set
            save: Whether to save immediately (default True)
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        if save:
            self._save()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {}).copy()
    
    def set_section(self, section: str, values: Dict[str, Any], save: bool = True) -> None:
        """Set an entire configuration section."""
        self._config[section] = values
        if save:
            self._save()
    
    def save(self) -> None:
        """Manually save configuration."""
        self._save()
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = self.DEFAULT_CONFIG.copy()
        self._save()
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return self._config.copy()


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
