"""Prompt template manager with Jinja2 support and caching."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, StrictUndefined

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent


class PromptManager:
    """Manages prompt templates with Jinja2 templating and auto-escaping for security."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize PromptManager with a templates directory.

        Args:
            templates_dir: Directory containing prompt templates. Defaults to PROMPTS_DIR.
        """
        self.templates_dir = templates_dir or PROMPTS_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            undefined=StrictUndefined,
        )

    def load_template(self, version: str = "default") -> str:
        """Load a prompt template by version name.

        Args:
            version: Template version/name (without extension).

        Returns:
            Rendered template as string.

        Raises:
            TemplateNotFound: If template doesn't exist and no default fallback.
        """
        template_name = f"{version}.md"
        try:
            template = self.env.get_template(template_name)
            return template.render()
        except TemplateNotFound:
            logger.warning(f"Template '{template_name}' not found, falling back to 'default.md'")
            if version != "default":
                template = self.env.get_template("default.md")
                return template.render()
            raise

    def render_prompt(self, version: str = "default", **context) -> str:
        """Render a prompt template with context variables.

        Args:
            version: Template version/name (without extension).
            **context: Variables to pass to the template (e.g., question, context).

        Returns:
            Rendered prompt with variables substituted safely.

        Raises:
            TemplateNotFound: If template doesn't exist.
            UndefinedError: If required variable is missing.
        """
        template_name = f"{version}.md"
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            logger.warning(f"Template '{template_name}' not found, falling back to 'default.md'")
            if version != "default":
                template = self.env.get_template("default.md")
            else:
                raise

        return template.render(**context)


# Global instance for prompt management
_prompt_manager: Optional[PromptManager] = None


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
