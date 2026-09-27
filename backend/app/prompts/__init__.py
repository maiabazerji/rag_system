from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.prompts.loader import (
    UnknownPromptVersionError,
    get_prompt_manager,
    prompt_version_exists,
    render_prompt,
)

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

    Raises:
        UnknownPromptVersionError: If no template exists for ``version``.
    """
    if not prompt_version_exists(version):
        raise UnknownPromptVersionError(f"Unknown prompt_version '{version}'")
    path = PROMPTS_DIR / f"{version}.md"
    return Prompt(name=path.stem, version=version, template=path.read_text(encoding="utf-8"))


__all__ = [
    "Prompt",
    "UnknownPromptVersionError",
    "get_prompt_manager",
    "load_prompt",
    "prompt_version_exists",
    "render_prompt",
]
