"""
Tool Registry for Dexter-Gliksbot
Provides centralized tool management and configuration for all agents.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

from .web_search import RobustWebSearchTool, AdvancedWebTool
from .shell import AdvancedPowerShellTool, AdvancedShellTool
from .file_ops import AdvancedFileReadTool
from .windows.gui import AdvancedGUIAutomationTool, ProGUIAutomationTool

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration for a tool instance"""
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class ToolRegistry:
    """
    Central registry for all available tools in the Dexter system.
    Agents can request tools by name or category.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tool registry with configuration.
        
        Args:
            config: Global configuration dictionary from dexter_config.yml
        """
        self.config = config
        self.tools_config = config.get("tools", {})
        self._tool_instances: Dict[str, Any] = {}
        self._tool_classes: Dict[str, Type] = {
            # Web tools
            "web_search": RobustWebSearchTool,
            "advanced_web": AdvancedWebTool,
            
            # Shell/PowerShell tools
            "powershell": AdvancedPowerShellTool,
            "shell": AdvancedShellTool,
            
            # File operations
            "file_ops": AdvancedFileReadTool,
            
            # GUI automation
            "gui_automation": AdvancedGUIAutomationTool,
            "pro_gui": ProGUIAutomationTool,
        }
        
        # Initialize default tools
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """Initialize commonly used tools on startup"""
        default_tools = ["web_search", "powershell", "file_ops"]
        
        for tool_name in default_tools:
            try:
                self.get_tool(tool_name)
                logger.info(f"Initialized default tool: {tool_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize default tool '{tool_name}': {e}")
    
    def get_tool(self, tool_name: str, force_new: bool = False) -> Any:
        """
        Get a tool instance by name. Returns cached instance unless force_new=True.
        
        Args:
            tool_name: Name of the tool to retrieve
            force_new: If True, creates a new instance instead of returning cached
            
        Returns:
            Tool instance
            
        Raises:
            ValueError: If tool name is not registered
        """
        if tool_name not in self._tool_classes:
            available = ", ".join(self._tool_classes.keys())
            raise ValueError(f"Unknown tool '{tool_name}'. Available: {available}")
        
        # Return cached instance if available and not forcing new
        if not force_new and tool_name in self._tool_instances:
            return self._tool_instances[tool_name]
        
        # Get tool-specific config
        tool_cfg = self._get_tool_config(tool_name)
        
        # Create new instance
        tool_class = self._tool_classes[tool_name]
        try:
            instance = tool_class(tool_cfg)
            
            # Cache the instance
            if not force_new:
                self._tool_instances[tool_name] = instance
            
            logger.debug(f"Created tool instance: {tool_name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create tool '{tool_name}': {e}", exc_info=True)
            raise
    
    def _get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        # Start with global config
        cfg = dict(self.config)
        
        # Merge tool-specific config
        if tool_name in self.tools_config:
            tool_specific = self.tools_config[tool_name]
            cfg.update(tool_specific)
        
        # Add security defaults
        cfg.setdefault("security", {
            "safe_mode": True,
            "denylist": [],
            "allowlist": [],
        })
        
        # Add common defaults
        cfg.setdefault("timeout", 60)
        cfg.setdefault("max_retries", 3)
        cfg.setdefault("working_directory", ".")
        
        return cfg
    
    def get_tools_by_category(self, category: str) -> List[Any]:
        """
        Get all tools in a category.
        
        Args:
            category: Tool category ("web", "shell", "file", "gui", "all")
            
        Returns:
            List of tool instances
        """
        category_map = {
            "web": ["web_search", "advanced_web"],
            "shell": ["powershell", "shell"],
            "file": ["file_ops"],
            "gui": ["gui_automation", "pro_gui"],
            "all": list(self._tool_classes.keys()),
        }
        
        if category not in category_map:
            raise ValueError(f"Unknown category '{category}'. Available: {', '.join(category_map.keys())}")
        
        tool_names = category_map[category]
        tools = []
        
        for name in tool_names:
            try:
                tool = self.get_tool(name)
                tools.append(tool)
            except Exception as e:
                logger.warning(f"Failed to get tool '{name}' in category '{category}': {e}")
        
        return tools
    
    def list_available_tools(self) -> Dict[str, Dict[str, str]]:
        """
        List all available tools with their descriptions.
        
        Returns:
            Dictionary mapping tool names to metadata
        """
        tools_info = {}
        
        for tool_name, tool_class in self._tool_classes.items():
            tools_info[tool_name] = {
                "name": getattr(tool_class, "name", tool_name),
                "description": getattr(tool_class, "description", "No description available"),
                "initialized": tool_name in self._tool_instances,
            }
        
        return tools_info
    
    def register_tool(self, tool_name: str, tool_class: Type):
        """
        Register a new tool class dynamically.
        
        Args:
            tool_name: Unique name for the tool
            tool_class: Tool class to register
        """
        if tool_name in self._tool_classes:
            logger.warning(f"Overwriting existing tool registration: {tool_name}")
        
        self._tool_classes[tool_name] = tool_class
        logger.info(f"Registered new tool: {tool_name}")
    
    def clear_cache(self):
        """Clear all cached tool instances"""
        self._tool_instances.clear()
        logger.info("Cleared tool instance cache")


# Global tool registry instance (initialized by ConfigManager)
_global_registry: Optional[ToolRegistry] = None


def initialize_tool_registry(config: Dict[str, Any]) -> ToolRegistry:
    """
    Initialize the global tool registry.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ToolRegistry instance
    """
    global _global_registry
    _global_registry = ToolRegistry(config)
    logger.info("Tool registry initialized")
    return _global_registry


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        ToolRegistry instance
        
    Raises:
        RuntimeError: If registry not initialized
    """
    if _global_registry is None:
        raise RuntimeError("Tool registry not initialized. Call initialize_tool_registry() first.")
    return _global_registry


# Convenience functions for agents
def get_tool(tool_name: str, force_new: bool = False) -> Any:
    """Convenience function to get a tool from the global registry"""
    return get_tool_registry().get_tool(tool_name, force_new)


def get_web_tools() -> List[Any]:
    """Get all web-related tools"""
    return get_tool_registry().get_tools_by_category("web")


def get_shell_tools() -> List[Any]:
    """Get all shell-related tools"""
    return get_tool_registry().get_tools_by_category("shell")


def get_file_tools() -> List[Any]:
    """Get all file operation tools"""
    return get_tool_registry().get_tools_by_category("file")


def get_gui_tools() -> List[Any]:
    """Get all GUI automation tools"""
    return get_tool_registry().get_tools_by_category("gui")
