"""Zip-bomb and encryption checks for zip-based formats (docx, pptx, odt, ods).

The parsers for these formats inflate every XML member they touch. A few
kilobytes of crafted zip can declare gigabytes of content, so the declared
sizes are checked before any parser sees the file. The declared sizes are a
safe bound: `zipfile` never inflates a member past its declared size, and a
member that lies about it fails its CRC check instead.
"""
from __future__ import annotations

import zipfile
from io import BytesIO

from app.config import settings
from app.rag.parsers.base import ArchiveTooLargeError, EncryptedDocumentError, ParseError

MAX_MEMBERS = 10_000
# Text-heavy XML compresses around 10-30x. A member far beyond that, and big
# enough to matter, is a bomb even when the total stays under the ceiling.
MAX_RATIO = 250
_RATIO_CHECK_MIN_BYTES = 5 * 1024 * 1024

# An encrypted OOXML file is not a zip at all but an OLE2 container holding an
# "EncryptedPackage" stream; so is a legacy .doc renamed to .docx.
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_OLE_ENCRYPTED_STREAM = "EncryptedPackage".encode("utf-16-le")


def open_zip(filename: str, content: bytes) -> zipfile.ZipFile:
    """Open a zip-based document after checking it is safe to inflate.

    Args:
        filename: Used in error messages.
        content: The raw file bytes.

    Returns:
        The open archive, for parsers that want to read members directly.

    Raises:
        EncryptedDocumentError: If the document is password-protected.
        ArchiveTooLargeError: If it would inflate past MAX_UNCOMPRESSED_MB,
            has too many members, or has a bomb-like compression ratio.
        ParseError: If it is not a valid zip at all.
    """
    if content.startswith(_OLE_MAGIC):
        if _OLE_ENCRYPTED_STREAM in content:
            raise EncryptedDocumentError(
                f"'{filename}' is password-protected. Remove the password and retry."
            )
        raise ParseError(
            f"'{filename}' is a legacy binary Office file, not a {_suffix(filename)} file. "
            "Re-save it in the modern format and retry."
        )
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except (zipfile.BadZipFile, OSError) as e:
        raise ParseError(f"'{filename}' is corrupt or not a {_suffix(filename)} file.") from e

    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ArchiveTooLargeError(
            f"'{filename}' contains {len(infos)} archive members (limit {MAX_MEMBERS})."
        )
    if any(info.flag_bits & 0x1 for info in infos):
        raise EncryptedDocumentError(
            f"'{filename}' is password-protected. Remove the password and retry."
        )

    limit = settings.max_uncompressed_bytes
    total = sum(info.file_size for info in infos)
    if total > limit:
        raise ArchiveTooLargeError(
            f"'{filename}' would expand to {total // (1024 * 1024)} MB, over the "
            f"{settings.max_uncompressed_mb} MB limit (MAX_UNCOMPRESSED_MB)."
        )
    for info in infos:
        if (
            info.file_size > _RATIO_CHECK_MIN_BYTES
            and info.file_size > MAX_RATIO * max(info.compress_size, 1)
        ):
            raise ArchiveTooLargeError(
                f"'{filename}' has a member compressed more than {MAX_RATIO}:1 "
                f"({info.filename}); refusing it as a likely zip bomb."
            )
    return archive


def _suffix(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else "zip-based"
