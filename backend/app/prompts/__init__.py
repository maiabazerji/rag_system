from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.prompts.loader import get_prompt_manager, render_prompt

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent


@dataclass
class Prompt:
    name: str
    version: str
    template: str


def load_prompt(version: str = "default") -> Prompt:
    """Load a prompt template (legacy interface).

    Loads the raw template text without rendering variables.
    For safe template rendering with variables, use render_prompt() instead.
    """
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        path = PROMPTS_DIR / "default.md"
    return Prompt(name=path.stem, version=version, template=path.read_text(encoding="utf-8"))


__all__ = ["Prompt", "load_prompt", "render_prompt", "get_prompt_manager"]
