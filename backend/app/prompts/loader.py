"""Prompt template manager with Jinja2 support and caching."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent

_VERSION_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class UnknownPromptVersionError(ValueError):
    """Raised for a prompt_version with no template. A client error (HTTP 400).

    Unknown versions used to fall back to ``default`` silently, so an eval run
    labelled with a mistyped version recorded default-prompt scores under the
    wrong name.
    """


def prompt_version_exists(version: str, templates_dir: Path | None = None) -> bool:
    """Whether a template exists for ``version`` (and the name is well formed)."""
    if not _VERSION_RE.match(version) or ".." in version:
        return False
    return ((templates_dir or PROMPTS_DIR) / f"{version}.md").is_file()


def available_prompt_versions(templates_dir: Path | None = None) -> list[str]:
    """Names of every prompt template, sorted."""
    return sorted(p.stem for p in (templates_dir or PROMPTS_DIR).glob("*.md"))


class PromptManager:
    """Loads and renders Jinja2 prompt templates.

    Autoescaping is deliberately OFF. These templates render Markdown that is
    sent to a language model, not HTML sent to a browser, so HTML-entity
    encoding would corrupt every apostrophe, ampersand and angle bracket in the
    question and the retrieved context. Defence against instructions embedded in
    retrieved text belongs in the prompt itself (see `default.md`), not here.
    """

    def __init__(self, templates_dir: Path | None = None):
        """Initialize PromptManager with a templates directory.

        Args:
            templates_dir: Directory containing prompt templates. Defaults to PROMPTS_DIR.
        """
        self.templates_dir = templates_dir or PROMPTS_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def load_template(self, version: str = "default") -> str:
        """Load a prompt template by version name.

        Args:
            version: Template version/name (without extension).

        Returns:
            Rendered template as string.

        Raises:
            UnknownPromptVersionError: If no template exists for ``version``.
        """
        return self._get(version).render()

    def _get(self, version: str):
        """Fetch a template, refusing unknown or malformed version names."""
        if not prompt_version_exists(version, self.templates_dir):
            raise UnknownPromptVersionError(
                f"Unknown prompt_version '{version}'. Available: "
                f"{', '.join(available_prompt_versions(self.templates_dir))}"
            )
        try:
            return self.env.get_template(f"{version}.md")
        except TemplateNotFound as e:  # pragma: no cover - raced with deletion
            raise UnknownPromptVersionError(f"Unknown prompt_version '{version}'") from e

    def render_prompt(self, version: str = "default", **context) -> str:
        """Render a prompt template with context variables.

        Args:
            version: Template version/name (without extension).
            **context: Variables to pass to the template (e.g., question, context).

        Returns:
            Rendered prompt with variables substituted safely.

        Raises:
            UnknownPromptVersionError: If no template exists for ``version``.
            UndefinedError: If required variable is missing.
        """
        return self._get(version).render(**context)


# Global instance for prompt management
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get or initialize the global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(version: str = "default", **context) -> str:
    """Render a prompt with the given version and context.

    Args:
        version: Template version/name.
        **context: Variables for template (question, context, etc.).

    Returns:
        Rendered prompt string.
    """
    return get_prompt_manager().render_prompt(version, **context)
