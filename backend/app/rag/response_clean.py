"""Clean up and simplify LLM responses for user-friendliness."""
from __future__ import annotations

import json
import re


def clean_response(text: str) -> str:
    """Remove JSON, code blocks, and excessive formatting from responses."""
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

    # Remove markdown code blocks
    text = re.sub(r"```[\w]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

    # Remove excessive headers/formatting
    text = re.sub(r"#+\s+", "", text)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text
