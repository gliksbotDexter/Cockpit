"""Policy-aware adapters for external tool modules."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from ..core.policy_overlay import CompositeDenyPolicy

logger = logging.getLogger(__name__)

GuardResult = Tuple[bool, str]
Guard = Callable[[CompositeDenyPolicy, Dict[str, Any]], GuardResult]


@dataclass
class ToolAdapter:
    """Wraps an external module and enforces policy before execution."""

    name: str
    description: str
    run_callable: Callable[..., Any]
    guard: Guard
    origin: str
    risk: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, policy: CompositeDenyPolicy, args: Dict[str, Any]) -> Dict[str, Any]:
        allowed, reason = self.guard(policy, args)
        if not allowed:
            return {"status": "denied", "reason": reason, "data": {}}
        try:
            response = await asyncio.to_thread(self.run_callable, **args)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Tool %s raised an exception", self.name)
            return {
                "status": "error",
                "reason": f"{self.name} failed: {exc}",
                "data": {"exception": str(exc)},
            }
        return self._normalise_response(response)

    def _normalise_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            success = bool(response.get("success", True))
            status = "ok" if success else "error"
            reason = "" if success else response.get("content", "tool error")
            return {"status": status, "reason": reason, "data": response}
        if isinstance(response, str):
            return {"status": "ok", "reason": "", "data": {"content": response}}
        return {"status": "ok", "reason": "", "data": {"result": response}}


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------


def _safe_paths(args: Dict[str, Any]) -> Iterable[str]:
    for key in ("path", "directory", "start", "destination", "output", "image_path", "filename"):
        value = args.get(key)
        if isinstance(value, str) and value:
            yield value


def _file_ops_guard(policy: CompositeDenyPolicy, args: Dict[str, Any]) -> GuardResult:
    action = (args.get("action") or "read").lower()
    paths = list(_safe_paths(args))
    mode = "write" if action in {"write"} or args.get("append") else "read"
    for path in paths:
        allowed, reason = policy.allow_path(path, mode=mode)
        if not allowed:
            return allowed, reason
    if action == "write" and "content" in args:
        allowed, reason = policy.allow_input(str(args.get("content", "")))
        if not allowed:
            return allowed, reason
    return True, ""


def _shell_guard(policy: CompositeDenyPolicy, args: Dict[str, Any]) -> GuardResult:
    command = args.get("command") or ""
    return policy.allow_process(command)


def _gui_guard(policy: CompositeDenyPolicy, args: Dict[str, Any]) -> GuardResult:
    text = args.get("text")
    if text:
        allowed, reason = policy.allow_input(str(text))
        if not allowed:
            return allowed, reason
    chord = args.get("chord")
    if chord:
        allowed, reason = policy.allow_hotkey(str(chord))
        if not allowed:
            return allowed, reason
    return True, ""


def _web_guard(policy: CompositeDenyPolicy, args: Dict[str, Any]) -> GuardResult:
    url = args.get("url")
    if url:
        return policy.allow_url(str(url))
    return True, ""


def _allow_all_guard(policy: CompositeDenyPolicy, args: Dict[str, Any]) -> GuardResult:  # noqa: ARG001
    return True, ""


_GUARD_MAP: Dict[str, Guard] = {
    "file_operations": _file_ops_guard,
    "file_ops": _file_ops_guard,
    "shell": _shell_guard,
    "powershell": _shell_guard,
    "gui": _gui_guard,
    "gui_automation": _gui_guard,
    "pro_gui_automation": _gui_guard,
    "web_search": _web_guard,
    "sys_check": _allow_all_guard,
}


def _guard_for(tool_name: str) -> Guard:
    return _GUARD_MAP.get(tool_name, _allow_all_guard)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _import_module(path: Path) -> Optional[Any]:
    module_name = f"dexter_external.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:  # pragma: no cover - defensive
        logger.warning("Skipping tool %s: import spec unavailable", path)
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        sys.modules.pop(module_name, None)
        logger.exception("Failed to import tool %s", path)
        return None
    return module


def load_external_tools(root: Optional[Path] = None) -> Dict[str, ToolAdapter]:
    """Load external tools from the configured directory."""
    adapters: Dict[str, ToolAdapter] = {}
    tool_root = root or _discover_tool_root()
    if tool_root is None or not tool_root.exists():
        logger.info("External tool directory not found; skipping tool adapters")
        return adapters

    for path in sorted(tool_root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = _import_module(path)
        if module is None:
            continue
        name = getattr(module, "NAME", path.stem)
        run_callable = getattr(module, "run", None)
        description = getattr(module, "DESCRIPTION", name)
        if not callable(run_callable):
            logger.warning("Tool %s missing callable run(); skipping", name)
            continue
        guard = _guard_for(name)
        adapter = ToolAdapter(
            name=name,
            description=description,
            run_callable=run_callable,
            guard=guard,
            origin=str(path),
            metadata={"params": getattr(module, "PARAMS", {})},
        )
        adapters[name] = adapter
    return adapters


def _discover_tool_root() -> Optional[Path]:
    env = os.environ.get("DEXTER_TOOL_ROOT")
    if env:
        return Path(env).expanduser()
    default = Path("M:/1/tools")
    if default.exists():
        return default
    return None

