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
        "last_classroom_output_dir": "",
        "last_classroom_video_folder": "",
        "last_crossday_output_dir": "",
        "last_crossday_video_folder": "",
        # Folder listener: videos successfully processed (survives app restart / update)
        "crossday_completed_videos": {"folder": "", "paths": []},
        "classroom_completed_videos": {"folder": "", "paths": []},
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
            "max_pixel_dist": 200,
            "delete_video_after_processing": False
        },
        "crossday": {
            # Face gallery: slightly strict to separate identities; margin only when 2nd is also strong
            "t_strict_merge": 0.36,
            "t_new_id": 0.22,
            "t_ratio_margin": 0.05,
            # With face pipeline on, wait this many frames before NF_* (gives InsightFace time to crop)
            "nf_min_frames_before_label": 12,
            "min_samples": 2,  # Stop collecting face crops after this many (centroid stability)
            "min_embeds_for_match": 1,  # Try gallery match with this many embeddings (1 = first InsightFace vec)
            "min_post_samples": 2,  # Post-video pass: min embeddings to resolve track
            "max_exemplars": 5,
            "t_outlier": 0.6,
            "t_match_student": 0.40,
            "visitor_upgrade_days": 3,
            "save_output_video": True,
            "delete_video_after_processing": False,
            "enable_motion_detection": False,
            "student_db_path": "",
            "enable_ocr_timestamp": True,
            "ocr_interval": 30,
            "timestamp_coords": [0, 15, 600, 90]
        },
        "inference": {
            "use_openvino": False,  # PyTorch by default = same as unique_and_recognition.py
            "force_cpu": False,
            "yolo_imgsz": 640,
            "face_det_size": 640,
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
                self._migrate_crossday_face_defaults()
                # Migrate to match unique_and_recognition.py defaults
                inf = self._config.get("inference") or {}
                if inf.get("yolo_imgsz") == 416 or inf.get("face_det_size") == 416:
                    if "inference" not in self._config:
                        self._config["inference"] = {}
                    if self._config["inference"].get("yolo_imgsz") == 416:
                        self._config["inference"]["yolo_imgsz"] = 640
                    if self._config["inference"].get("face_det_size") == 416:
                        self._config["inference"]["face_det_size"] = 640
                    self._save()
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

    def _migrate_crossday_face_defaults(self) -> None:
        """
        Persisted config.json overrides DEFAULT_CONFIG, so old strict Settings values
        (e.g. t_strict_merge=0.9, min_samples=9) never picked up new lenient defaults.
        One-time migration when face_match_defaults_revision < 2.
        """
        cd = self._config.setdefault("crossday", {})
        try:
            old_rev = int(cd.get("face_match_defaults_revision", 0))
        except (TypeError, ValueError):
            old_rev = 0
        if old_rev >= 2:
            return

        d = self.DEFAULT_CONFIG["crossday"]
        changed = False

        try:
            strict = float(cd.get("t_strict_merge", d["t_strict_merge"]))
        except (TypeError, ValueError):
            strict = float(d["t_strict_merge"])
        if strict >= 0.70:
            cd["t_strict_merge"] = d["t_strict_merge"]
            changed = True

        try:
            nid = float(cd.get("t_new_id", d["t_new_id"]))
        except (TypeError, ValueError):
            nid = float(d["t_new_id"])
        if nid <= 0.12:
            cd["t_new_id"] = d["t_new_id"]
            changed = True

        try:
            ms = int(cd.get("min_samples", d["min_samples"]))
        except (TypeError, ValueError):
            ms = int(d["min_samples"])
        if ms >= 6:
            cd["min_samples"] = d["min_samples"]
            changed = True

        if changed:
            cd["t_ratio_margin"] = d["t_ratio_margin"]
            if "min_embeds_for_match" in d:
                cd["min_embeds_for_match"] = d["min_embeds_for_match"]
            if "min_post_samples" in d:
                cd["min_post_samples"] = d["min_post_samples"]

        cd["face_match_defaults_revision"] = 2
        if changed or old_rev < 2:
            self._save()
    
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
