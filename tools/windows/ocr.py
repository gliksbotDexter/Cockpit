"""
Optical character recognition (OCR) helper for Dexter‑Gliksbot.

This module wraps the `pytesseract` and `PIL` (Pillow) libraries to
provide a single function for extracting text from a portion of the
screen or from a specific window handle.  Similar to the automation
helpers, the imports are performed lazily so that the rest of the
platform can start up on systems where OCR is not available.  When
OCR dependencies are missing, the functions raise clear errors with
installation instructions instead of silently returning dummy text.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def _require_pillow() -> Tuple[Any, Any]:
    """Import Pillow modules lazily.

    Returns a tuple of (ImageGrab, Image) classes.  Raises a
    RuntimeError if Pillow is not installed.  We import inside a helper
    to centralise error handling.
    """
    try:
        from PIL import Image, ImageGrab  # type: ignore
        return ImageGrab, Image
    except Exception as exc:
        logger.error("Pillow unavailable: %s", exc)
        raise RuntimeError(
            "OCR requires the 'pillow' package.  Install it with `pip install Pillow`."
        ) from exc


def _require_pytesseract() -> Any:
    """Import pytesseract lazily.

    Returns the pytesseract module.  Raises RuntimeError if import fails.
    """
    try:
        import pytesseract  # type: ignore
        return pytesseract
    except Exception as exc:
        logger.error("pytesseract unavailable: %s", exc)
        raise RuntimeError(
            "OCR requires the 'pytesseract' package and Tesseract binary.  "
            "Install the Python package with `pip install pytesseract` and ensure "
            "the Tesseract executable is on your PATH."
        ) from exc


def ocr_capture(
    x: Optional[int] = None,
    y: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """Capture a region of the screen and return its OCR text.

    If no coordinates are provided, the entire screen is captured.  On
    multi‑monitor setups, coordinates are relative to the primary
    monitor.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        width: Width of the region.
        height: Height of the region.

    Returns:
        The text extracted from the captured image.  Returns an empty
        string on failure.
    """
    ImageGrab, _Image = _require_pillow()
    pytesseract = _require_pytesseract()
    try:
        if x is not None and y is not None and width is not None and height is not None:
            bbox = (x, y, x + width, y + height)
            img = ImageGrab.grab(bbox=bbox)
        else:
            img = ImageGrab.grab()
        # Convert to RGB to avoid issues on some platforms
        img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return text
    except Exception as exc:
        logger.exception("OCR capture failed: %s", exc)
        return ""


def ocr_hwnd(hwnd: int, *, tesseract_path: Optional[str] = None) -> str:
    """OCR capture for a specific window handle (hwnd).

    This is a placeholder implementation that raises a NotImplementedError
    because capturing a window by handle is platform‑specific and
    requires Win32 API calls.  A proper implementation would use
    `pywinauto` or `win32gui` to capture the window into a bitmap and
    then pass it through pytesseract.  The function remains here to
    preserve the original API surface and to provide a clear error
    message if called on unsupported systems.
    """
    raise NotImplementedError(
        "ocr_hwnd is not implemented in this environment.  Use ocr_capture() "
        "with screen coordinates or contribute a platform‑specific implementation."
    )