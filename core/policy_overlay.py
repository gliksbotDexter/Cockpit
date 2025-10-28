"""Composite deny policy with configurable guardrails."""
from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_PROCESS_DENY = {
    "rm -rf *",
    "format*",
    "shutdown*",
    "poweroff*",
    "del /f /s *",
    "erase *",
    "rd /s *",
    "cipher /w:*",
    "mkfs*",
}

_DEFAULT_PATH_DENY = {
    "C:/Windows/*",
    "C:/Program Files/*",
    "/etc/*",
    "/var/lib/*",
}

_DEFAULT_URL_DENY_SCHEMES = {"file", "ftp", "sftp"}


def _compile_regex(patterns: Iterable[str]) -> Iterable[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logger.warning("Invalid deny regex skipped: %s", pattern)
    return compiled


def _casefold(value: str) -> str:
    return value.casefold() if hasattr(value, "casefold") else value.lower()


class CompositeDenyPolicy:
    """Configurable deny policy that merges profile and overlay rules."""

    def __init__(self, profile: Dict[str, Any] | None = None, overlay: Mapping[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self.profile = profile or {}
        self.overlay = overlay or {}
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "overlay": self.overlay,
        }

    # ------------------------------------------------------------------
    def allow_process(self, command: str) -> Tuple[bool, str]:
        command_cf = _casefold(command or "")
        rules = self._rules("process")
        patterns = set(_DEFAULT_PROCESS_DENY)
        patterns.update(rules.get("deny_cmd_patterns", []))
        for pattern in patterns:
            if fnmatch.fnmatch(command_cf, _casefold(pattern)):
                return False, f"Command '{command}' blocked by pattern '{pattern}'"
        for forbidden in rules.get("deny_exact", []):
            if command_cf.strip() == _casefold(forbidden):
                return False, f"Command '{command}' explicitly denied"
        return True, ""

    def allow_path(self, path: str, mode: str = "read") -> Tuple[bool, str]:
        resolved = Path(path).resolve().as_posix()
        rules = self._rules("files")
        deny_globs = set(_DEFAULT_PATH_DENY)
        deny_globs.update(rules.get("deny_write_globs" if mode != "read" else "deny_read_globs", []))
        deny_globs.update(rules.get("deny_globs", []))
        deny_dirs = rules.get("deny_dirs", [])

        for directory in deny_dirs:
            directory_norm = Path(directory).as_posix().rstrip("/")
            if resolved.startswith(directory_norm):
                return False, f"Access to '{directory}' is blocked"
        for pattern in deny_globs:
            if fnmatch.fnmatch(resolved, _casefold(pattern)):
                return False, f"Path '{resolved}' denied by pattern '{pattern}'"
        return True, ""

    def allow_hotkey(self, chord: str) -> Tuple[bool, str]:
        rules = self._rules("hotkeys")
        denial = { _casefold(item) for item in rules.get("deny", []) }
        if chord and _casefold(chord) in denial:
            return False, f"Hotkey '{chord}' is disabled by policy"
        return True, ""

    def allow_input(self, text: str) -> Tuple[bool, str]:
        rules = self._rules("input")
        limit = rules.get("max_chars")
        if limit is not None and len(text) > int(limit):
            return False, f"Input exceeds maximum length of {limit}"
        regexes = _compile_regex(rules.get("deny_regex", []))
        for pattern in regexes:
            if pattern.search(text or ""):
                return False, f"Input matches denied pattern '{pattern.pattern}'"
        return True, ""

    def allow_url(self, url: str) -> Tuple[bool, str]:
        rules = self._rules("network")
        parsed = urlparse(url or "")
        if not parsed.scheme:
            return False, "URL missing scheme"
        scheme_cf = _casefold(parsed.scheme)
        deny_schemes = {_casefold(s) for s in rules.get("deny_schemes", [])} | _DEFAULT_URL_DENY_SCHEMES
        if scheme_cf in deny_schemes:
            return False, f"Scheme '{parsed.scheme}' is not permitted"
        host = parsed.hostname or ""
        deny_hosts = {_casefold(h) for h in rules.get("deny_hosts", [])}
        if any(fnmatch.fnmatch(_casefold(host), pattern) for pattern in deny_hosts):
            return False, f"Host '{host}' is blocked"
        deny_regex = _compile_regex(rules.get("deny_url_regex", []))
        for pattern in deny_regex:
            if pattern.search(url):
                return False, f"URL blocked by pattern '{pattern.pattern}'"
        return True, ""

    def allow_rate(self, event: str, *, count: int, window_seconds: float) -> Tuple[bool, str]:
        rules = self._rules("rate_limits")
        key = f"max_{event}"
        limit = rules.get(key)
        if limit is None:
            return True, ""
        if count > int(limit):
            return False, f"Rate limit exceeded for {event}: {count}>{limit} in {window_seconds}s"
        return True, ""

    # ------------------------------------------------------------------
    def _rules(self, category: str) -> Dict[str, Any]:
        with self._lock:
            if category in self._cache:
                return self._cache[category]
            base = self.profile.get(category, {}) if isinstance(self.profile, Mapping) else {}
            over = self.overlay.get(category, {}) if isinstance(self.overlay, Mapping) else {}
            merged = self._merge_dicts(base, over)
            self._cache[category] = merged
            return merged

    @staticmethod
    def _merge_dicts(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        keys = set(primary.keys()) | set(secondary.keys())
        for key in keys:
            left = primary.get(key)
            right = secondary.get(key)
            if isinstance(left, Mapping) and isinstance(right, Mapping):
                result[key] = CompositeDenyPolicy._merge_dicts(left, right)
            elif isinstance(left, list) or isinstance(right, list):
                merged = []
                if isinstance(left, list):
                    merged.extend(left)
                if isinstance(right, list):
                    merged.extend(right)
                result[key] = merged
            else:
                result[key] = right if right is not None else left
        return result

    # ------------------------------------------------------------------
    def is_allowed(self, subject: str, action: str) -> bool:
        """Compatibility helper that delegates to the appropriate check."""
        subject = (subject or "").lower()
        action = action or ""
        if subject == "process":
            return self.allow_process(action)[0]
        if subject == "file":
            return self.allow_path(action)[0]
        if subject == "url":
            return self.allow_url(action)[0]
        if subject == "hotkey":
            return self.allow_hotkey(action)[0]
        if subject == "input":
            return self.allow_input(action)[0]
        return True

    # ------------------------------------------------------------------
    def refresh(self, profile: Dict[str, Any] | None = None, overlay: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if profile is not None:
                self.profile = profile
            if overlay is not None:
                self.overlay = overlay
            self._cache.clear()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"CompositeDenyPolicy(profile={self.profile!r}, overlay={self.overlay!r})"

