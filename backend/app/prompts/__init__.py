from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@dataclass
class Prompt:
    name: str
    version: str
    template: str


def load_prompt(version: str = "default") -> Prompt:
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        path = PROMPTS_DIR / "default.md"
    return Prompt(name=path.stem, version=version, template=path.read_text(encoding="utf-8"))
