"""Email .eml parser (stdlib email).

The body prefers the text/plain alternative and falls back to HTML through
the safe converter in `html`. Attachments are always listed by name; those of
a supported type are also parsed and appended, under a depth limit (an email
attached to an email attached to...) and a size limit.
"""
from __future__ import annotations

import email
import email.policy
import re
from email.message import EmailMessage, Message, MIMEPart
from email.utils import parsedate_to_datetime

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.parsers.base import DocumentMetadata, ParsedDocument, ParseError, clean_str
from app.rag.parsers.html import html_to_text

logger = get_structured_logger(__name__)

# An email inside an email inside an email is parsed; one level deeper is only listed.
MAX_ATTACHMENT_DEPTH = 2
MAX_PARSED_ATTACHMENTS = 20

_HEADERS = ("From", "To", "Cc", "Date", "Subject")
_MD_HEADING = re.compile(r"^(#{1,6})(\s)", re.MULTILINE)


def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Parse an email: headers, body, then its attachments."""
    try:
        message = email.message_from_bytes(content, policy=email.policy.default)
    except Exception as e:
        raise ParseError(f"Could not read '{filename}' as an email ({type(e).__name__}).") from e
    if not isinstance(message, EmailMessage) or not any(message.get(h) for h in _HEADERS):
        raise ParseError(f"'{filename}' has no email headers; it is not an .eml file.")

    subject = clean_str(message.get("Subject"))
    lines = [f"# {subject}"] if subject else []
    lines += [f"{h}: {message[h]}" for h in _HEADERS[:-1] if message.get(h)]
    blocks = ["\n".join(lines), _body(message)]
    blocks += _attachments(message, depth)

    return ParsedDocument(
        text="\n\n".join(b for b in blocks if b.strip()),
        metadata=_metadata(message, subject),
    )


def _body(message: EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = _part_text(part)
    return html_to_text(text) if part.get_content_type() == "text/html" else text.strip()


def _part_text(part: MIMEPart) -> str:
    try:
        return str(part.get_content())
    except (LookupError, UnicodeDecodeError, AssertionError):
        # An unknown or lying charset: decode what is there rather than fail.
        payload = part.get_payload(decode=True)
        return payload.decode("utf-8", "replace") if isinstance(payload, bytes) else ""


def _attachments(message: EmailMessage, depth: int) -> list[str]:
    from app.rag.parsers.registry import extract

    listing: list[str] = []
    parsed: list[str] = []
    for index, part in enumerate(message.iter_attachments(), start=1):
        name, payload = _attachment_payload(part, index)
        listing.append(f"- {name} ({len(payload) // 1024 or 1} KB)")

        if depth + 1 > MAX_ATTACHMENT_DEPTH or len(parsed) >= MAX_PARSED_ATTACHMENTS:
            continue
        if len(payload) > settings.max_upload_bytes:
            continue
        try:
            document = extract(name, payload, depth=depth + 1)
        except Exception as e:  # one bad attachment must not lose the email
            logger.info(
                "Attachment not indexed",
                extra_fields={"attachment": name, "reason": str(e)},
            )
            continue
        if document.text.strip():
            parsed.append(f"## Attachment: {name}\n\n{_demote_headings(document.text)}")

    if not listing:
        return []
    return ["## Attachments\n\n" + "\n".join(listing), *parsed]


def _attachment_payload(part: MIMEPart, index: int) -> tuple[str, bytes]:
    """An attachment's name and raw bytes; a forwarded message is re-serialised."""
    if part.get_content_type() == "message/rfc822":
        inner = part.get_payload(0) if part.is_multipart() else None
        message = inner.as_bytes() if isinstance(inner, Message) else b""
        return part.get_filename() or f"attached-message-{index}.eml", message
    decoded = part.get_payload(decode=True)
    data = decoded if isinstance(decoded, bytes) else b""
    return part.get_filename() or f"attachment-{index}", data


def _demote_headings(text: str) -> str:
    """Push an attachment's headings under its "## Attachment" heading."""
    return _MD_HEADING.sub(lambda m: "#" * min(len(m.group(1)) + 2, 6) + m.group(2), text)


def _metadata(message: EmailMessage, subject: str | None) -> DocumentMetadata:
    created = None
    if message.get("Date"):
        try:
            created = parsedate_to_datetime(str(message["Date"])).isoformat()
        except (TypeError, ValueError):
            created = clean_str(message["Date"])
    extra: dict[str, str | int] = {
        key.lower(): str(message[key]) for key in ("To", "Cc") if message.get(key)
    }
    return DocumentMetadata(
        source_format="eml",
        title=subject,
        author=clean_str(message.get("From")),
        created=created,
        language=clean_str(message.get("Content-Language")),
        extra=extra,
    )
