"""Email notifier task used by the training flow on success / failure."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from ..config import get_settings
from ..observability.logging import get_logger

log = get_logger(__name__)


def send_email_notification(
    subject: str,
    body: str,
    to_addr: str | None = None,
) -> bool:
    """Send a plain-text email via configured SMTP server.

    Returns True on success, False if configuration is missing or sending
    fails. Errors are logged but never raised — this is a notifier, not a
    pipeline-blocking step.
    """
    settings = get_settings()
    host = settings.smtp_host
    user = settings.smtp_user
    password = settings.smtp_pass
    from_addr = settings.notify_from or settings.smtp_user
    to = to_addr or settings.notify_to

    if not host or not user or not password or not to:
        log.warning(
            "email_notification_skipped",
            reason="missing_smtp_config",
            host=bool(host),
            user=bool(user),
            password=bool(password),
            to=bool(to),
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, settings.smtp_port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        log.info("email_notification_sent", subject=subject, to=to)
        return True
    except Exception as exc:
        log.error("email_notification_failed", error=str(exc), subject=subject)
        return False
