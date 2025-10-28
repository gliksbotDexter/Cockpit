from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_profiles(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"mode_default": "medium", "profiles": {}}
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("mode_default", "medium")
    data.setdefault("profiles", {})
    return data


def save_profiles(path: str, profiles: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profiles, handle, sort_keys=False, allow_unicode=False)


def list_profiles(profiles: dict) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, entry in profiles.get("profiles", {}).items():
        label = entry.get("label") or name.title()
        description = entry.get("description") or ""
        policy = entry.get("policy") if isinstance(entry, dict) else {}
        if not policy and isinstance(entry, dict):
            # Backward compatibility: profile stored directly as policy.
            policy = {k: v for k, v in entry.items() if k not in {"label", "description"}}
        out[name] = {"label": label, "description": description, "policy": policy}
    return out


def select_profile(profiles: dict, mode: str) -> Dict[str, Any]:
    profiles_map = profiles.get("profiles", {})
    if not profiles_map:
        return {}
    mode_default = profiles.get("mode_default") or next(iter(profiles_map))
    selected = profiles_map.get(mode or mode_default)
    if not selected:
        raise KeyError(f"Unknown profile '{mode}'")
    if isinstance(selected, dict) and "policy" in selected:
        return selected["policy"] or {}
    return {k: v for k, v in selected.items() if k not in {"label", "description"}}


def update_profile_policy(profiles: dict, name: str, policy: Dict[str, Any]) -> Tuple[dict, Dict[str, Any]]:
    profiles_map = profiles.setdefault("profiles", {})
    entry = profiles_map.setdefault(name, {})
    if isinstance(entry, dict) and "policy" in entry:
        entry["policy"] = policy
    else:
        entry = {"policy": policy}
        profiles_map[name] = entry
    return profiles, entry
