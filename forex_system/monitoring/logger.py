# Logging framework
# monitoring/logger.py
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pytz

LOG_DIR = Path("logs/")
LOG_DIR.mkdir(exist_ok=True)

# ── Timezone-aware formatter ──────────────────────────────────────────────────
class MadridFormatter(logging.Formatter):
    """
    Custom formatter that displays all log timestamps in
    Europe/Madrid local time (CET/CEST) instead of UTC.
    """
    TZ = pytz.timezone("Europe/Madrid")

    def formatTime(self, record, datefmt=None):
        # Convert the log record's UTC timestamp to Madrid local time
        utc_dt    = datetime.fromtimestamp(record.created, tz=pytz.utc)
        local_dt  = utc_dt.astimezone(self.TZ)
        if datefmt:
            return local_dt.strftime(datefmt)
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger          # already configured

    logger.setLevel(logging.DEBUG)

    fmt = MadridFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler — rotating 5 MB × 5 backups
    today = datetime.now(pytz.timezone("Europe/Madrid")).strftime("%Y%m%d")
    fh = RotatingFileHandler(
        LOG_DIR / f"forex_{today}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
