# monitoring/logger.py
"""
Centralised logging for GODBOT v3.0.

Design:
  • One root logger  "GOD_BOT"  with two handlers:
      – StreamHandler  → stdout   (INFO+)
      – TimedRotatingFileHandler  → logs/forex_YYYYMMDD.log  (DEBUG+)
  • Child loggers (e.g. get_logger("signals.engine")) inherit both
    handlers via Python's logger hierarchy and add NO handlers of
    their own, preventing duplicate output and multiple open file
    handles on Windows.
  • All timestamps displayed in Europe/Madrid local time (CET/CEST).
  • Millisecond precision retained for order-timing diagnostics.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import pytz

# Fix 5 – parents=True so nested paths (e.g. logs/godbot/) are created safely.
LOG_DIR = Path("logs/")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Timezone-aware formatter ──────────────────────────────────────────────────
class MadridFormatter(logging.Formatter):
    """
    Custom formatter that displays all log timestamps in
    Europe/Madrid local time (CET/CEST) instead of UTC.
    Milliseconds are included for order-timing diagnostics.
    """
    TZ = pytz.timezone("Europe/Madrid")

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        utc_dt   = datetime.fromtimestamp(record.created, tz=pytz.utc)
        local_dt = utc_dt.astimezone(self.TZ)
        if datefmt:
            return local_dt.strftime(datefmt)
        # Fix 6 – Include milliseconds for high-frequency timing diagnostics.
        base = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        return f"{base},{record.msecs:03.0f}"


# ── Root logger factory ───────────────────────────────────────────────────────
def _build_root_logger() -> logging.Logger:
    """
    Build and return the 'GOD_BOT' root logger with a console handler
    and a daily-rotating file handler.  Called once at module import.
    """
    root = logging.getLogger("GOD_BOT")

    # Guard: if already configured (e.g. module reloaded) do not add
    # duplicate handlers.
    if root.handlers:
        return root

    root.setLevel(logging.DEBUG)

    # Fix 3 – Prevent log records from bubbling up to the Python root
    # logger and being printed twice by third-party basicConfig calls.
    root.propagate = False

    fmt = MadridFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    )

    # ── Console handler (INFO+) ───────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # ── File handler — daily rotation, keeps 30 days of logs ─────────────────
    # Fix 1 & 2 – Use TimedRotatingFileHandler with when="midnight" so the
    # file rolls over automatically at midnight and the date suffix advances
    # correctly. This replaces the baked-in date filename + RotatingFileHandler
    # which would have written to a stale file after midnight.
    log_file = LOG_DIR / "forex.log"
    fh = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,          # keep one month of daily logs
        encoding="utf-8",
        utc=False,               # rotate at local midnight (Madrid time)
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Suffix format used by TimedRotatingFileHandler for rolled files,
    # e.g. forex.log.2026-03-09
    fh.suffix = "%Y-%m-%d"
    root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'GOD_BOT' hierarchy.

    Fix 4 – Child loggers inherit both handlers from the 'GOD_BOT' root
    logger and add NO handlers of their own.  This prevents:
      • Duplicate console output
      • Multiple file handles open on the same file (PermissionError
        on Windows, interleaved writes on Linux)

    Usage:
        from monitoring.logger import get_logger
        logger = get_logger(__name__)
    """
    # Ensure the root logger exists and is fully configured before
    # creating any child.
    _build_root_logger()

    child = logging.getLogger(f"GOD_BOT.{name}")
    # Child loggers must NOT add their own handlers — they propagate
    # up to GOD_BOT which owns the console + file handlers.
    child.propagate = True
    return child


# ── Module-level shared instance ──────────────────────────────────────────────
# All modules that do `from monitoring.logger import logger` get this
# single pre-configured instance.  Modules that need a named child logger
# can still call get_logger(__name__) instead.
logger: logging.Logger = _build_root_logger()
