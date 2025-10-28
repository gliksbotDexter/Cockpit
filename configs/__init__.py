"""
Dexter Configuration Module

Provides centralized configuration management for Dexter-Gliksbot.

Usage:
    from dexter_autonomy.configs import get_global_config
    
    config = get_global_config()
    port = config.get("ui_bridge.port", 8765)
    agents = config.get_section("agents")
"""
from .config_manager import ConfigManager, get_global_config, reload_global_config

__all__ = [
    "ConfigManager",
    "get_global_config",
    "reload_global_config",
]
