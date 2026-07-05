"""Clean up and simplify LLM responses for user-friendliness."""
from __future__ import annotations

import json
import re


def clean_response(text: str) -> str:
    """Remove JSON, code blocks, chunk IDs, and excessive formatting."""
    if not text:
        return text

    # If response is JSON, try to extract the answer field
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "answer" in data:
                return clean_response(str(data["answer"]))
            if "summary" in data:
                return clean_response(str(data["summary"]))
    except (json.JSONDecodeError, TypeError):
        pass

    # Remove chunk ID citations like [abc123def:0]
    text = re.sub(r"\[([a-f0-9]+:[0-9]+)\]", "", text)

    # Remove markdown code blocks
    text = re.sub(r"```[\w]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

    # Remove markdown headers (##, ###, etc)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove "Rated X/5" and rating-related lines
    text = re.sub(r"Rated \d+/5.*?(?:\n|$)", "", text)
    text = re.sub(r"Saved to.*?(?:\n|$)", "", text)
    text = re.sub(r"Thanks for rating.*?(?:\n|$)", "", text)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text
