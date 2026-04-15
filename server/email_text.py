"""Extract normalized text from RFC822 messages for training and inference."""

from __future__ import annotations

import email
import re
from email.policy import default as email_policy_default

HTML_TAG_RE = re.compile(r"<[^>]+>")


def _decode_part(part: email.message.Message) -> str | None:
    payload = part.get_payload(decode=True)
    if not payload:
        return None
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract_text_from_message(msg: email.message.Message) -> str:
    subject = msg.get("Subject", "") or ""
    from_ = msg.get("From", "") or ""
    bodies: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() != "multipart":
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    text = _decode_part(part)
                    if text:
                        bodies.append(text)
                elif ctype == "text/html":
                    text = _decode_part(part)
                    if text:
                        bodies.append(HTML_TAG_RE.sub(" ", text))
    else:
        text = _decode_part(msg)
        if text:
            if msg.get_content_type() == "text/html":
                text = HTML_TAG_RE.sub(" ", text)
            bodies.append(text)

    body = "\n".join(bodies)
    return normalize_text(f"Subject: {subject}\nFrom: {from_}\n\n{body}")


def extract_text_from_bytes(data: bytes) -> str:
    msg = email.message_from_bytes(data, policy=email_policy_default)
    return extract_text_from_message(msg)


def normalize_text(s: str) -> str:
    return " ".join(s.split()).strip()
