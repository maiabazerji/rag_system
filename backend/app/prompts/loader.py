"""Prompt template manager with Jinja2 support and caching."""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent


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
