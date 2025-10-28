from __future__ import annotations
import json, time
from pathlib import Path
class TraceSink:
    def __init__(self, base: str = "./collaboration"):
        self.path = Path(base) / "trace.jsonl"; self.path.parent.mkdir(parents=True, exist_ok=True)
    async def handle(self, msg: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **msg})+"\n")
