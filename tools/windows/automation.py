"""
Windows automation helpers for Dexter‑Gliksbot.

This module wraps the `pyautogui` library so that calls to click,
keyboard shortcuts and text entry only attempt to import `pyautogui`
at runtime.  If the module is unavailable (e.g. on non‑Windows
platforms or before dependencies are installed) the functions will
raise a clear and actionable error instead of silently doing nothing.

The original repository shipped stub implementations that returned
``True`` without performing any actions.  Those stubs blocked all
automation because nothing actually happened.  The functions below
implement the documented behaviour: they delegate to pyautogui and
log any errors encountered.  Small sleeps are inserted after
actions to avoid races with the Windows message loop.  See the
WHATS‑LEFT‑TODO.md file for a detailed discussion of the stub issue.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)

try:
    import ctypes
    from ctypes import wintypes
    USER32 = ctypes.windll.user32
    WINDOWS_AVAILABLE = True
except Exception:
    USER32 = None
    WINDOWS_AVAILABLE = False


def activate_window(hwnd: int) -> bool:
    """Activate and bring a window to the foreground.
    
    Args:
        hwnd: Window handle to activate.
        
    Returns:
        True if the window was activated successfully, False otherwise.
    """
    if not WINDOWS_AVAILABLE or USER32 is None:
        logger.error("Windows API not available for window activation")
        return False
    
    try:
        # First, restore the window if it's minimized
        USER32.ShowWindow(ctypes.c_void_p(hwnd), 9)  # SW_RESTORE = 9
        time.sleep(0.05)
        
        # Then bring it to the foreground
        result = USER32.SetForegroundWindow(ctypes.c_void_p(hwnd))
        time.sleep(0.1)
        
        # Verify it has focus
        focused_hwnd = USER32.GetForegroundWindow()
        success = (focused_hwnd == hwnd)
        
        if success:
            logger.debug(f"Successfully activated window HWND {hwnd}")
        else:
            logger.warning(f"Failed to activate window HWND {hwnd}. Focused HWND: {focused_hwnd}")
            
        return success
    except Exception as exc:
        logger.exception(f"Failed to activate window HWND {hwnd}")
        return False


def get_focused_window() -> Optional[int]:
    """Get the handle of the currently focused window.
    
    Returns:
        Window handle (HWND) of the focused window, or None if unavailable.
    """
    if not WINDOWS_AVAILABLE or USER32 is None:
        return None
    
    try:
        hwnd = USER32.GetForegroundWindow()
        return hwnd if hwnd else None
    except Exception as exc:
        logger.exception("Failed to get focused window")
        return None


def _require_pyautogui() -> Any:
    """Import pyautogui lazily.

    Returns the pyautogui module if available.  Raises a RuntimeError
    with installation instructions if the import fails.  This helper
    centralises error handling so that individual functions remain
    concise.
    """
    try:
        import pyautogui  # type: ignore
        # Use failsafe to avoid infinite loops on unexpected screens.
        pyautogui.FAILSAFE = True
        return pyautogui
    except Exception as exc:  # broad catch because import may raise OSError
        logger.error("pyautogui unavailable: %s", exc)
        raise RuntimeError(
            "Windows automation is unavailable because the 'pyautogui' "
            "package is missing or failed to load.  Run `pip install pyautogui` "
            "and ensure you are running on a Windows desktop session."
        ) from exc


def click(x: int, y: int, button: str = "left") -> bool:
    """Click at the given screen coordinates.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        button: Mouse button to click (default: "left").

    Returns:
        True if the click succeeded, False otherwise.
    """
    pyautogui = _require_pyautogui()
    try:
        pyautogui.click(x, y, button=button)
        # Small delay to give the UI time to process the click
        time.sleep(0.1)
        return True
    except Exception as exc:
        logger.exception("Click at (%s, %s) with button '%s' failed", x, y, button)
        return False


def hotkey(*keys: str) -> bool:
    """Press one or more hotkeys simultaneously.

    Example: ``hotkey('ctrl', 'c')`` will send Ctrl+C.  Keys should
    be passed as individual strings.

    Returns True if the key sequence was sent successfully, False otherwise.
    """
    if not keys:
        raise ValueError("At least one key must be provided to hotkey()")
    pyautogui = _require_pyautogui()
    try:
        pyautogui.hotkey(*keys)
        time.sleep(0.1)
        return True
    except Exception as exc:
        logger.exception("Hotkey %s failed", keys)
        return False


def type_text(text: str, interval: float = 0.05) -> bool:
    """Type a string of text using the keyboard.

    Args:
        text: The text to type.
        interval: Delay between key presses (seconds).  Default is 50 ms.

    Returns:
        True if the text was typed successfully, False otherwise.
    """
    pyautogui = _require_pyautogui()
    try:
        pyautogui.typewrite(text, interval=interval)
        return True
    except Exception as exc:
        logger.exception("Typing text failed: %s", text)
        return False


def scroll(amount: int) -> bool:
    """Scroll the mouse wheel.

    Positive values scroll up, negative values scroll down.  This helper
    is provided to round out the API and may be useful in future recipes.
    """
    pyautogui = _require_pyautogui()
    try:
        pyautogui.scroll(amount)
        return True
    except Exception:
        logger.exception("Scrolling by %s failed", amount)
        return False