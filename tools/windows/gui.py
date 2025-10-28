from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class GUIAction:
    action: str
    target: str
    timestamp: float
    result: str
    duration: float

class AdvancedGUIAutomationTool:
    name = "gui_automation"
    description = """Advanced Windows GUI automation with safety, recording, and intelligent targeting.
    Features:
    - Multiple backend support (pywinauto, pyautogui)
    - Action recording and playback
    - Visual element recognition
    - Safety checks and undo capability
    - Screenshot integration
    - Coordinate system management
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.automation_cfg = cfg.get("automation", {})
        self.backend = self.automation_cfg.get("gui_backend", "pywinauto")
        self.action_history: List[GUIAction] = []
        self.max_history = cfg.get("max_gui_history", 100)
        self.safety_mode = self.automation_cfg.get("safety_mode", True)
        self.screenshot_dir = Path(self.automation_cfg.get("screenshot_dir", "./screenshots"))
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Initialize backends
        self.pyautogui_available = self._init_pyautogui()
        self.pywinauto_available = self._init_pywinauto()
        
        # Safety boundaries
        self.screen_width = 1920  # Default, will update dynamically
        self.screen_height = 1080
        if self.pyautogui_available:
            try:
                import pyautogui
                self.screen_width, self.screen_height = pyautogui.size()
            except:
                pass
    
    def _init_pyautogui(self) -> bool:
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move to corner to abort
            return True
        except ImportError:
            return False
    
    def _init_pywinauto(self) -> bool:
        try:
            from pywinauto.application import Application
            return True
        except ImportError:
            return False
    
    def _is_coordinate_safe(self, x: int, y: int) -> bool:
        """Check if coordinates are within safe boundaries"""
        if not self.safety_mode:
            return True
        
        # Check screen boundaries
        if x < 0 or x > self.screen_width or y < 0 or y > self.screen_height:
            return False
        
        # Avoid dangerous areas (customize as needed)
        dangerous_areas = [
            (0, 0, 10, 10),  # Top-left corner (failsafe)
            (self.screen_width - 10, 0, 10, 10),  # Top-right corner
        ]
        
        for area_x, area_y, area_w, area_h in dangerous_areas:
            if area_x <= x <= area_x + area_w and area_y <= y <= area_y + area_h:
                return False
        
        return True
    
    def _record_action(self, action: str, target: str, result: str, duration: float):
        """Record GUI action for history/playback"""
        gui_action = GUIAction(
            action=action,
            target=target,
            timestamp=time.time(),
            result=result,
            duration=duration
        )
        
        self.action_history.append(gui_action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def click(self, x: int, y: int, delay: float = 0.1, clicks: int = 1) -> str:
        """Click at coordinates with safety checks"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        if not self._is_coordinate_safe(x, y):
            return f"🚫 Unsafe coordinates: ({x}, {y}). Outside screen bounds or in danger zone."
        
        try:
            start_time = time.time()
            import pyautogui
            
            pyautogui.click(x, y, clicks=clicks, interval=delay)
            duration = time.time() - start_time
            
            result = f"✅ Clicked {clicks} time(s) at ({x}, {y})"
            self._record_action("click", f"{x},{y}", result, duration)
            
            return result
            
        except Exception as e:
            return f"💥 Click failed: {str(e)}"
    
    def type_text(self, text: str, interval: float = 0.05) -> str:
        """Type text with controlled timing"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        try:
            start_time = time.time()
            import pyautogui
            
            # Safety check for malicious text
            if self.safety_mode and any(dangerous in text.lower() for dangerous in 
                                      ["del ", "rm ", "format ", "shutdown"]):
                return "🚫 Dangerous text blocked by safety mode"
            
            pyautogui.typewrite(text, interval=interval)
            duration = time.time() - start_time
            
            result = f"✅ Typed {len(text)} characters"
            self._record_action("type", text[:50], result, duration)
            
            return result
            
        except Exception as e:
            return f"💥 Typing failed: {str(e)}"
    
    def take_screenshot(self, filename: str = None, region: Tuple[int, int, int, int] = None) -> str:
        """Take screenshot with optional region"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        try:
            import pyautogui
            from PIL import Image
            
            timestamp = int(time.time())
            if not filename:
                filename = f"screenshot_{timestamp}.png"
            
            filepath = self.screenshot_dir / filename
            
            if region:
                screenshot = pyautogui.screenshot(region=region)
                result_msg = f"📸 Screenshot taken (region: {region})"
            else:
                screenshot = pyautogui.screenshot()
                result_msg = "📸 Full screen screenshot taken"
            
            screenshot.save(filepath)
            
            return f"{result_msg}\n💾 Saved to: {filepath}\n📏 Size: {screenshot.width}x{screenshot.height}"
            
        except Exception as e:
            return f"💥 Screenshot failed: {str(e)}"
    
    def find_image_on_screen(self, image_path: str, confidence: float = 0.8) -> str:
        """Find image/template on screen"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        try:
            import pyautogui
            
            if not os.path.exists(image_path):
                return f"❌ Image not found: {image_path}"
            
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            
            if location:
                center = pyautogui.center(location)
                return f"🎯 Found image at: ({center.x}, {center.y})\n📍 Box: {location}"
            else:
                return "📭 Image not found on screen"
                
        except Exception as e:
            return f"💥 Image search failed: {str(e)}"
    
    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> str:
        """Move mouse smoothly to coordinates"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        if not self._is_coordinate_safe(x, y):
            return f"🚫 Unsafe coordinates: ({x}, {y})"
        
        try:
            start_time = time.time()
            import pyautogui
            
            pyautogui.moveTo(x, y, duration=duration)
            duration_taken = time.time() - start_time
            
            result = f"🖱️ Moved mouse to ({x}, {y})"
            self._record_action("move", f"{x},{y}", result, duration_taken)
            
            return result
            
        except Exception as e:
            return f"💥 Mouse move failed: {str(e)}"
    
    def drag_mouse(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                   duration: float = 1.0) -> str:
        """Drag mouse from start to end position"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available. Install with: pip install pyautogui"
        
        if not self._is_coordinate_safe(start_x, start_y) or not self._is_coordinate_safe(end_x, end_y):
            return "🚫 Unsafe coordinates provided"
        
        try:
            start_time = time.time()
            import pyautogui
            
            pyautogui.drag(start_x, start_y, end_x, end_y, duration=duration, button='left')
            duration_taken = time.time() - start_time
            
            result = f"✋ Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            self._record_action("drag", f"{start_x},{start_y}->{end_x},{end_y}", result, duration_taken)
            
            return result
            
        except Exception as e:
            return f"💥 Drag failed: {str(e)}"
    
    def get_screen_info(self) -> str:
        """Get screen resolution and mouse position"""
        if not self.pyautogui_available:
            return "❌ pyautogui not available"
        
        try:
            import pyautogui
            
            screen_width, screen_height = pyautogui.size()
            mouse_x, mouse_y = pyautogui.position()
            
            return f"""🖥️ Screen Information:
Resolution: {screen_width}x{screen_height}
Mouse Position: ({mouse_x}, {mouse_y})
Safety Mode: {"ON" if self.safety_mode else "OFF"}
Screenshot Directory: {self.screenshot_dir}"""
            
        except Exception as e:
            return f"💥 Screen info failed: {str(e)}"
    
    def undo_last_action(self) -> str:
        """Undo the last GUI action (limited support)"""
        if not self.action_history:
            return "📭 No actions to undo"
        
        last_action = self.action_history[-1]
        return f"↩️ Last action was: {last_action.action} at {last_action.target}\n(GUI undo is limited - manual reversal may be needed)"
    
    def get_action_history(self, limit: int = 10) -> str:
        """Get recent GUI action history"""
        if not self.action_history:
            return "📭 No GUI actions recorded"
        
        lines = ["📋 Recent GUI Actions:"]
        for action in self.action_history[-limit:]:
            lines.append(f"- {time.strftime('%H:%M:%S', time.localtime(action.timestamp))} "
                        f"{action.action} {action.target[:30]} → {action.result[:50]}")
        
        return "\n".join(lines)
    
    def run(self, action: str, **kwargs) -> str:
        """Main entry point for GUI automation"""
        actions = {
            "click": lambda: self.click(
                kwargs.get("x", 0), 
                kwargs.get("y", 0),
                kwargs.get("delay", 0.1),
                kwargs.get("clicks", 1)
            ),
            "type": lambda: self.type_text(
                kwargs.get("text", ""), 
                kwargs.get("interval", 0.05)
            ),
            "screenshot": lambda: self.take_screenshot(
                kwargs.get("filename"),
                kwargs.get("region")
            ),
            "find_image": lambda: self.find_image_on_screen(
                kwargs.get("image_path", ""),
                kwargs.get("confidence", 0.8)
            ),
            "move_mouse": lambda: self.move_mouse(
                kwargs.get("x", 0),
                kwargs.get("y", 0),
                kwargs.get("duration", 0.5)
            ),
            "drag_mouse": lambda: self.drag_mouse(
                kwargs.get("start_x", 0),
                kwargs.get("start_y", 0),
                kwargs.get("end_x", 100),
                kwargs.get("end_y", 100),
                kwargs.get("duration", 1.0)
            ),
            "screen_info": self.get_screen_info,
            "undo": self.undo_last_action,
            "history": lambda: self.get_action_history(kwargs.get("limit", 10))
        }
        
        if action in actions:
            return actions[action]()
        else:
            return f"❓ Unknown action: {action}. Available: {', '.join(actions.keys())}"

# Enhanced version with window/application control
class ProGUIAutomationTool(AdvancedGUIAutomationTool):
    name = "pro_gui_automation"
    description = """Professional GUI automation with application control and advanced features."""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.applications = {}
    
    def launch_application(self, app_path: str, wait_for_window: bool = True) -> str:
        """Launch application and optionally wait for its window"""
        if not self.pywinauto_available:
            return "❌ pywinauto not available. Install with: pip install pywinauto"
        
        try:
            from pywinauto.application import Application
            
            if not os.path.exists(app_path):
                return f"❌ Application not found: {app_path}"
            
            app = Application(backend="uia").start(app_path)
            app_name = os.path.basename(app_path)
            self.applications[app_name] = app
            
            if wait_for_window:
                time.sleep(2)  # Give app time to load
                try:
                    window = app.top_window()
                    window.set_focus()
                    return f"🚀 Launched {app_name} and focused window: {window.window_text()}"
                except:
                    return f"🚀 Launched {app_name} (window detection pending)"
            else:
                return f"🚀 Launched {app_name}"
                
        except Exception as e:
            return f"💥 Launch failed: {str(e)}"
    
    def find_window(self, title_contains: str = None, class_name: str = None) -> str:
        """Find and interact with windows"""
        if not self.pywinauto_available:
            return "❌ pywinauto not available"
        
        try:
            from pywinauto import Desktop
            
            desktop = Desktop(backend="uia")
            
            if title_contains:
                windows = [w for w in desktop.windows() if title_contains.lower() in w.window_text().lower()]
            elif class_name:
                windows = [w for w in desktop.windows() if class_name.lower() in str(w.class_name()).lower()]
            else:
                windows = list(desktop.windows())[:10]  # First 10 windows
            
            if not windows:
                return "📭 No matching windows found"
            
            lines = [f"🔍 Found {len(windows)} windows:"]
            for i, window in enumerate(windows[:10]):
                lines.append(f"{i+1}. Title: {window.window_text()[:50]}")
                lines.append(f"   Class: {window.class_name()}")
                lines.append(f"   Visible: {window.is_visible()}")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"💥 Window search failed: {str(e)}"
    
    def run_pro(self, action: str, **kwargs) -> str:
        """Extended actions for professional automation"""
        pro_actions = {
            "launch_app": lambda: self.launch_application(
                kwargs.get("app_path", ""),
                kwargs.get("wait_for_window", True)
            ),
            "find_window": lambda: self.find_window(
                kwargs.get("title_contains"),
                kwargs.get("class_name")
            )
        }
        
        if action in pro_actions:
            return pro_actions[action]()
        else:
            # Fall back to parent class
            return super().run(action, **kwargs)

# Update the factory function
def create_all_tools(cfg: Dict[str, Any]) -> List[Any]:
    """Create all tools with configuration"""
    tools = [
        WebSearchTool(cfg),
        PowerShellTool(cfg),
        FileOperationsTool(cfg)
    ]
    
    # Add GUI tools if dependencies are available
    try:
        tools.append(AdvancedGUIAutomationTool(cfg))
    except Exception as e:
        print(f"GUI automation tool not available: {e}")
    
    return tools

