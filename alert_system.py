"""
User-facing alert channel: desktop notification, optional alarm sound, and console log.

Designed to run from Flask request handlers via background threads so HTTP latency stays low.
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


def _play_alarm_sound() -> None:
    """
    Play a short system beep or WAV where supported.

    On Windows uses winsound; elsewhere falls back to printing BEL to the terminal.
    """
    if not config.ENABLE_SOUND_ALARM:
        return
    try:
        if sys.platform == "win32":
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", flush=True)
    except Exception as exc:
        logger.debug("Alarm sound skipped: %s", exc)


def _show_popup(title: str, message: str) -> None:
    """
    Show a native-style popup from a worker thread.

    On Windows uses MessageBoxW (no Tk main loop — safe with Flask).
    Else tries plyer desktop notification, then logs only.
    """
    if not config.ENABLE_POPUP:
        return

    if sys.platform == "win32":
        try:
            import ctypes

            MB_ICONWARNING = 0x30
            ctypes.windll.user32.MessageBoxW(0, message, title, MB_ICONWARNING)
            return
        except Exception as exc:
            logger.debug("Win32 MessageBox failed: %s", exc)

    try:
        from plyer import notification

        notification.notify(title=title, message=message, timeout=10)
    except Exception as exc:
        logger.warning("Popup/desktop notify unavailable: %s", exc)


def trigger_high_alert(
    criminal_name: str,
    crime_type: str,
    confidence: float,
    detection_time: datetime | None = None,
) -> None:
    """
    Fire HIGH ALERT UX: sound + popup + structured log line.

    detection_time defaults to current UTC if omitted.
    """
    ts = detection_time or datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        f"Criminal match: {criminal_name}\n"
        f"Crime type: {crime_type}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Time: {ts_str}"
    )
    logger.error("HIGH ALERT — %s", msg.replace("\n", " | "))

    def _run() -> None:
        _play_alarm_sound()
        _show_popup("HIGH ALERT — Criminal match", msg)

    threading.Thread(target=_run, daemon=True).start()


def notify_no_match(confidence: float, distance: float) -> None:
    """Optional quiet path for monitoring (disabled by default to avoid noise)."""
    dist_txt = f"{distance:.4f}" if distance < float("inf") else "n/a"
    logger.info(
        "No match above threshold (best confidence=%.4f distance=%s)",
        confidence,
        dist_txt,
    )
