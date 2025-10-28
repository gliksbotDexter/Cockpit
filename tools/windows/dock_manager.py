"""
DockManager: Manages the state of external windows docked in the Cockpit.

This class is responsible for:
- Scanning for scannable/dockable windows on the system.
- Maintaining a list of currently docked windows (by HWND).
- Providing methods to add (dock) and remove (undock) windows from the list.
- Publishing events to the TripleBus when windows are docked or undocked.
"""
import asyncio
import logging
import ctypes
from ctypes import wintypes
from typing import List, Dict, Any, Optional

from ...core.triple_bus import TripleBusSystem, MainTopic

logger = logging.getLogger(__name__)

try:
    import pygetwindow as gw
    WINDOWS_TOOLS_AVAILABLE = True
except ImportError:
    gw = None
    WINDOWS_TOOLS_AVAILABLE = False


USER32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


def _is_valid_window(hwnd: int) -> bool:
    if USER32 is None:
        return False
    return bool(USER32.IsWindow(ctypes.c_void_p(hwnd)))


def _read_window_title(hwnd: int) -> Optional[str]:
    if USER32 is None:
        return None
    length = USER32.GetWindowTextLengthW(ctypes.c_void_p(hwnd))
    if length <= 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    USER32.GetWindowTextW(ctypes.c_void_p(hwnd), buffer, length + 1)
    return buffer.value.strip() or None


class DockedWindow:
    """A simple data class representing a docked window."""
    def __init__(self, hwnd: int, title: str, process: Optional[str] = None):
        self.hwnd = hwnd
        self.title = title
        self.process = process

    def to_dict(self) -> Dict[str, Any]:
        return {"hwnd": self.hwnd, "title": self.title, "process": self.process}


class DockManager:
    """Manages scanning and tracking of docked windows."""

    def __init__(self, buses: TripleBusSystem):
        if not WINDOWS_TOOLS_AVAILABLE:
            logger.warning("pygetwindow not installed. Docking features will be disabled. Run: pip install pygetwindow")
        self.buses = buses
        self._docked_windows: Dict[int, DockedWindow] = {}
        self._lock = asyncio.Lock()

    async def scan_for_windows(self) -> List[Dict[str, Any]]:
        """Scans for all visible, non-minimized windows suitable for docking."""
        windows = []
        if WINDOWS_TOOLS_AVAILABLE:
            try:
                for win in gw.getAllWindows():
                    if win.title and win.visible and not win.isMinimized:
                        windows.append({
                            "hwnd": win._hWnd,
                            "title": win.title,
                            "is_docked": win._hWnd in self._docked_windows
                        })
            except Exception as e:
                logger.error(f"Failed to scan for windows: {e}")
        elif USER32 is not None:
            # Minimal fallback: enumerate from current dock list only
            for hwnd, docked in self._docked_windows.items():
                if _is_valid_window(hwnd):
                    windows.append({
                        "hwnd": hwnd,
                        "title": docked.title or f"HWND {hwnd}",
                        "is_docked": True
                    })
        return windows

    async def add_dock(self, hwnd: int, title: str) -> bool:
        """Adds a window to the list of docked windows."""
        async with self._lock:
            if hwnd in self._docked_windows:
                return True # Already docked

            if not WINDOWS_TOOLS_AVAILABLE:
                raise RuntimeError("Cannot dock window: pygetwindow is not installed.")

            try:
                verified = False
                if WINDOWS_TOOLS_AVAILABLE:
                    win = gw.getWindowsWithHandle(hwnd)
                    if win:
                        if not title:
                            title = win[0].title
                        verified = True
                if not verified and _is_valid_window(hwnd):
                    title = title or _read_window_title(hwnd) or f"HWND {hwnd}"
                    verified = True
                if not verified:
                    logger.warning(f"Attempted to dock a non-existent window with HWND: {hwnd}")
                    return False
                self._docked_windows[hwnd] = DockedWindow(hwnd=hwnd, title=title)
                logger.info(f"Window docked: '{title}' (HWND: {hwnd})")
                await self.buses.main.publish(MainTopic.TRACE, {
                    "from": "dock_manager",
                    "event": "window_docked",
                    "window": self._docked_windows[hwnd].to_dict()
                })
                return True
            except Exception as e:
                logger.error(f"Failed to dock window HWND {hwnd}: {e}")
                return False

    async def remove_dock(self, hwnd: int) -> bool:
        """Removes a window from the list of docked windows."""
        async with self._lock:
            if hwnd not in self._docked_windows:
                return False # Not currently docked

            window = self._docked_windows.pop(hwnd)
            logger.info(f"Window undocked: '{window.title}' (HWND: {hwnd})")
            await self.buses.main.publish(MainTopic.TRACE, {
                "from": "dock_manager",
                "event": "window_undocked",
                "window": window.to_dict()
            })
            return True

    def get_docked_windows(self) -> List[Dict[str, Any]]:
        """Returns a list of all currently docked windows."""
        return [win.to_dict() for win in self._docked_windows.values()]