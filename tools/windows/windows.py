"""
Advanced Windows-specific automation tools.

This module provides functions for interacting with specific windows using
their handles (HWND), such as capturing their content for OCR.
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)