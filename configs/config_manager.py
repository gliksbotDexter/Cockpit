"""
Configuration Manager for Dexter‑Gliksbot (patched copy)

This file is a verbatim copy of the upstream ConfigManager.  It
provides a centralized, thread‑safe configuration loader with
automatic reload support using watchdog.  The manager reads from
`configs/dexter_config.yml`, validates the structure and exposes
convenience methods for nested lookup and mutation.  See the original
repository for full documentation.  This copy allows the patched
repository to operate independently of the upstream code.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
except ImportError:
    # If watchdog is missing, provide dummy classes to avoid import errors.
    Observer = None
    FileSystemEventHandler = object  # type: ignore
    FileModifiedEvent = object  # type: ignore

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config file changes."""
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._last_modified = 0
    def on_modified(self, event):  # type: ignore[override]
        # Only handle modification events
        if Observer is None:
            return
        if not isinstance(event, FileModifiedEvent):
            return
        import time
        current_time = time.time()
        if current_time - self._last_modified < 0.5:
            return
        self._last_modified = current_time
        if Path(event.src_path) == self.config_manager.config_path:
            logger.info(f"Config file modified: {event.src_path}")
            try:
                self.config_manager.reload(notify=True)
            except Exception as e:
                logger.error(f"Failed to reload config: {e}")


class ConfigManager:
    """
    Centralized configuration manager for Dexter‑Gliksbot.

    Singleton pattern ensures only one instance exists globally.  Thread
    safe for concurrent access.  See `configs/dexter_config.yml` for
    schema details.
    """
    _instance: Optional[ConfigManager] = None
    _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, config_path: Optional[Path] = None, watch: bool = True, fallback_legacy: bool = True):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._lock = threading.RLock()
        self._config: Dict[str, Any] = {}
        self._observers: List[Observer] = []
        self._change_callbacks: List[callable] = []
        if config_path is None:
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "configs" / "dexter_config.yml"
        self.config_path = Path(config_path)
        self.fallback_legacy = fallback_legacy
        self.reload(notify=False)
        
        # Initialize tool registry after config is loaded
        self._initialize_tool_registry()
        
        if watch and Observer is not None:
            self.start_watching()
    
    def _initialize_tool_registry(self):
        """Initialize the global tool registry with current configuration"""
        try:
            from ..tools.registry import initialize_tool_registry
            initialize_tool_registry(self._config)
            logger.info("Tool registry initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tool registry: {e}", exc_info=True)
    def reload(self, notify: bool = True) -> None:
        with self._lock:
            logger.info(f"Loading configuration from {self.config_path}")
            if self.config_path.exists():
                self._config = self._load_unified_config()
                logger.info("Loaded unified configuration")
            elif self.fallback_legacy:
                logger.warning("Unified config not found, falling back to legacy configs")
                self._config = self._load_legacy_configs()
            else:
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}\n"
                    "Create configs/dexter_config.yml or enable fallback_legacy=True"
                )
            valid, errors = self._validate_config(self._config)
            if not valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.error(error_msg)
                raise ValueError(error_msg)
            if notify:
                self._notify_change("*", self._config)
    def _load_unified_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            return config
        except Exception as e:
            logger.error(f"Failed to load {self.config_path}: {e}")
            raise
    def _load_legacy_configs(self) -> Dict[str, Any]:
        config_dir = self.config_path.parent
        merged = {
            "version": "1.0.0-legacy",
            "system": {},
            "agents": [],
            "deny_list": {"global": {}, "agents": {}},
            "security_profiles": {},
            "providers": {},
            "brain": {},
            "workers": {},
            "ui_bridge": {},
            "cockpit": {},
            "collaboration": {},
            "windows": {},
            "logging": {},
            "development": {},
        }
        # Load various legacy files if they exist.  For brevity, these
        # operations are identical to the upstream implementation.
        dexter_yml = config_dir / "dexter.yml"
        if dexter_yml.exists():
            with open(dexter_yml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                merged["system"] = {
                    "data_dir": data.get("data_dir", "./data"),
                    "collab_dir": data.get("collab_dir", "./collaboration"),
                }
                merged["ui_bridge"]["host"] = data.get("server", {}).get("host", "0.0.0.0")
                merged["ui_bridge"]["port"] = data.get("server", {}).get("port", 8765)
                merged["providers"]["ollama"] = {
                    "endpoint": data.get("ollama", {}).get("host", "http://127.0.0.1:11434"),
                }
                merged["windows"]["ocr"] = {
                    "tesseract_path": data.get("tesseract_path", "C:/Program Files/Tesseract-OCR/tesseract.exe"),
                }
        slots_yml = config_dir / "slots.yml"
        if slots_yml.exists():
            with open(slots_yml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                slots = data.get("slots", {})
                for slot_id, slot_config in slots.items():
                    agent = {
                        "id": slot_id,
                        "name": slot_config.get("label", slot_id),
                        "description": slot_config.get("description", ""),
                        "enabled": True,
                        "provider": "ollama",  # default; override below
                        "model": slot_config.get("model", ""),
                        "temperature": slot_config.get("temperature", 0.2),
                        "system_prompt": slot_config.get("system_prompt", ""),
                        "params": slot_config.get("ollama_options", {}),
                    }
                    merged["agents"].append(agent)
        denylist_master = config_dir / "denylist.master.yml"
        if denylist_master.exists():
            with open(denylist_master, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                merged["deny_list"]["global"] = data
        denylist_profiles = config_dir / "denylist.profiles.yml"
        if denylist_profiles.exists():
            with open(denylist_profiles, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                merged["security_profiles"] = {
                    "active_profile": data.get("mode_default", "medium"),
                    "profiles": data.get("profiles", {}),
                }
        agents_overlays = config_dir / "agents.overlays.yml"
        if agents_overlays.exists():
            with open(agents_overlays, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                merged["deny_list"]["agents"] = data
        logger.info("Merged legacy configurations")
        return merged
    def _validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        required_sections = ["agents", "deny_list", "providers"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        # Validate agents list
        agents = config.get("agents")
        if agents is not None and not isinstance(agents, (list, dict)):
            errors.append("'agents' must be a list or dict")
        # Validate deny_list
        deny_list = config.get("deny_list")
        if deny_list is not None and not isinstance(deny_list, dict):
            errors.append("'deny_list' must be a dict")
        # Validate providers
        providers = config.get("providers")
        if providers is not None and not isinstance(providers, dict):
            errors.append("'providers' must be a dict")
        return len(errors) == 0, errors
    def _notify_change(self, path: str, value: Any) -> None:
        for callback in self._change_callbacks:
            try:
                callback(path, value)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            keys = path.split('.')
            value: Any = self._config
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list):
                    try:
                        idx = int(key)
                        value = value[idx]
                    except (ValueError, IndexError):
                        return default
                else:
                    return default
                if value is None:
                    return default
            return value
    def get_section(self, section: str, default: Any = None) -> Dict[str, Any]:
        """Return a config section as a dict.

        Accepts an optional `default` which is returned if the section is missing
        or is not a dict. This signature is compatible with callers that pass a
        default value (legacy usage in start.py).
        """
        if default is None:
            default = {}
        with self._lock:
            val = self._config.get(section, default)
            if isinstance(val, dict):
                return val
            return default

    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        agents = self.get_section("agents")
        if isinstance(agents, dict):
            return agents.get(agent_id)
        elif isinstance(agents, list):
            for agent in agents:
                if agent.get("id") == agent_id:
                    return agent
        return None

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        providers = self.get_section("providers")
        return providers.get(provider_name) if isinstance(providers, dict) else None

    def set(self, path: str, value: Any) -> None:
        """Set configuration value by dot-separated path."""
        with self._lock:
            keys = path.split(".")
            target = self._config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            target[keys[-1]] = value
            self._notify_change(path, value)

    def save(self) -> None:
        """Save configuration to disk atomically."""
        with self._lock:
            valid, errors = self._validate_config(self._config)
            if not valid:
                error_msg = "Cannot save invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.config_path.with_suffix(".tmp")
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(
                        self._config,
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                        indent=2
                    )
                temp_path.replace(self.config_path)
                logger.info(f"Configuration saved to {self.config_path}")
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                logger.error(f"Failed to save configuration: {e}")
                raise

    def start_watching(self) -> None:
        """Start file system watcher for hot-reload."""
        if Observer is None:
            logger.warning("watchdog not available, file watching disabled")
            return
        if not self.config_path.exists():
            logger.warning(f"Cannot watch non-existent config: {self.config_path}")
            return
        event_handler = ConfigFileHandler(self)
        observer = Observer()
        observer.schedule(
            event_handler,
            path=str(self.config_path.parent),
            recursive=False
        )
        observer.start()
        self._observers.append(observer)
        logger.info(f"Started watching config file: {self.config_path}")

    def stop_watching(self) -> None:
        """Stop all file system watchers."""
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()
        logger.info("Stopped watching config file")

    def register_change_callback(self, callback: callable) -> None:
        """Register callback for configuration changes."""
        self._change_callbacks.append(callback)

    def export_to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dictionary."""
        import copy
        with self._lock:
            return copy.deepcopy(self._config)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop watchers."""
        self.stop_watching()


# Global singleton
_global_config: Optional[ConfigManager] = None


def get_global_config() -> ConfigManager:
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(watch=False)
    return _global_config


def reload_global_config() -> None:
    config = get_global_config()
    config.reload(notify=True)


def get_config(*args, **kwargs):
    """Compatibility alias for legacy imports."""
    return get_global_config(*args, **kwargs)
