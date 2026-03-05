# research/calendar_scanner.py
import feedparser
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
from monitoring.logger import get_logger

logger = get_logger("CalendarScanner")


class CalendarScanner:
    """Economic calendar using RSS feeds only (MT5 calendar API unavailable)."""

    RSS_FEEDS = {
        "fxstreet":  "https://www.fxstreet.com/rss/news",
        "investing":  "https://www.investing.com/rss/news_14.rss",
        "forexlive":  "https://www.forexlive.com/feed/news",
        "dailyfx":    "https://www.dailyfx.com/feeds/all",
    }

    HIGH_IMPACT_KEYWORDS = [
        "non-farm payroll", "nonfarm payroll", "nfp",
        "fed rate", "federal reserve", "fomc", "fed decision",
        "interest rate decision", "rate decision", "rate hike", "rate cut",
        "cpi", "inflation", "consumer price",
        "gdp", "gross domestic product",
        "unemployment", "jobless claims", "jobs report",
        "retail sales", "trade balance",
        "pmi", "ism manufacturing", "ism services",
        "ecb", "boe", "bank of england", "bank of japan", "boj",
        "powell", "lagarde", "bailey",
        "payroll", "employment change",
        "core inflation", "pce", "personal consumption",
        "durable goods", "housing starts",
        "ats", "ats report",
    ]

    # ── Only block articles that are genuinely future‑event previews ─────────
    PREVIEW_KEYWORDS = [
        "tomorrow",
        "next week",
        "next month",
        "upcoming week",
        "week ahead",
        "preview:",          # colon makes it a section header, not a result
        "what to watch",
        "events to watch",
        "trading week ahead",
        "economic week ahead",
        "scheduled for",
        "due out",
        "due on",
        "due next",
        "will be released",
        "set to release",
        "looking ahead",
        "ahead of next",
        "markets brace",
        "traders await",
        "all eyes on",
    ]

    SYMBOL_CURRENCIES = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "XAUUSD": ["XAU", "USD"],
    }

    KEYWORD_CURRENCY = {
        "fed":       "USD", "fomc":     "USD", "powell":   "USD",
        "payroll":   "USD", "nfp":      "USD", "jobless":  "USD",
        "ism":       "USD", "pce":      "USD", "durable":  "USD",
        "housing":   "USD", "retail":   "USD",
        "ecb":       "EUR", "lagarde":  "EUR", "euro":     "EUR",
        "boe":       "GBP", "bailey":   "GBP", "sterling": "GBP",
        "boj":       "JPY", "japan":    "JPY", "yen":      "JPY",
        "rba":       "AUD", "australia":"AUD", "aussie":   "AUD",
        "boc":       "CAD", "canada":   "CAD", "loonie":   "CAD",
        "gold":      "XAU", "xau":      "XAU",
    }

    CACHE_MINUTES = 60

    def __init__(self):
        self._cache_df:   Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime]      = None

    # ── public API ────────────────────────────────────────────────────────────

    def is_safe_to_trade(
        self,
        symbol: str,
        minutes_before: int = 30,
        minutes_after:  int = 15,
    ) -> dict:
        """Return {'safe': bool, 'reason': str, 'events': list}."""
        try:
            events_df = self._get_rss_events()
            if events_df is None or events_df.empty:
                return {"safe": True, "reason": "No events found", "events": []}

            currencies = self.SYMBOL_CURRENCIES.get(symbol.upper(), [])
            now_utc    = datetime.now(pytz.utc)
            blocking   = []

            for _, row in events_df.iterrows():
                evt_time = row.get("datetime")
                if evt_time is None:
                    continue
                if not evt_time.tzinfo:
                    evt_time = pytz.utc.localize(evt_time)

                delta_mins = (evt_time - now_utc).total_seconds() / 60
                in_window  = -minutes_after <= delta_mins <= minutes_before

                if in_window and row.get("currency", "UNKNOWN") in currencies:
                    blocking.append(row.to_dict())

            if blocking:
                titles = ", ".join(e.get("event", "?") for e in blocking[:2])
                return {
                    "safe":   False,
                    "reason": f"High-impact event near: {titles}",
                    "events": blocking,
                }

            return {"safe": True, "reason": "No blocking events", "events": []}

        except Exception as e:
            logger.warning(f"CalendarScanner error: {e}")
            return {"safe": True, "reason": f"Error: {e}", "events": []}

    def get_todays_events(self) -> list:
        """Return today's high-impact events as a list of dicts."""
        try:
            df = self._get_rss_events()
            if df is None or df.empty:
                return []
            today_utc = datetime.now(pytz.utc).date()
            result = []
            for _, row in df.iterrows():
                evt_time = row.get("datetime")
                if evt_time is None:
                    continue
                if not evt_time.tzinfo:
                    evt_time = pytz.utc.localize(evt_time)
                if evt_time.date() == today_utc:
                    result.append(row.to_dict())
            return result
        except Exception:
            return []

    def print_todays_events(self) -> None:
        events = self.get_todays_events()
        if not events:
            logger.info("📅 No high-impact events today.")
            return
        logger.info(f"📅 Today's high-impact events ({len(events)}):")
        for e in events:
            t = e.get("datetime", "?")
            if hasattr(t, "strftime"):
                t = t.strftime("%H:%M UTC")
            logger.info(f"   {t} | {e.get('currency','?'):4s} | {e.get('event','?')}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_rss_events(self) -> Optional[pd.DataFrame]:
        if self._is_cache_valid():
            return self._cache_df
        self._cache_df   = self._fetch_all_rss_feeds()
        self._cache_time = datetime.now(pytz.utc)
        return self._cache_df

    def _fetch_all_rss_feeds(self) -> pd.DataFrame:
        rows = []
        for source, url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                count = 0
                for entry in feed.entries[:30]:
                    title   = getattr(entry, "title",   "") or ""
                    summary = getattr(entry, "summary", "") or ""
                    text    = f"{title} {summary}".lower()

                    # ── skip genuine preview/reminder articles ─────────────
                    if self._is_preview_article(text):
                        logger.debug(f"📰 Skipping preview article: {title[:80]}")
                        continue

                    # ── keep only high-impact items ────────────────────────
                    if not any(kw in text for kw in self.HIGH_IMPACT_KEYWORDS):
                        continue

                    currency = self._detect_currency(text)
                    pub_dt   = self._parse_pub_date(entry)

                    rows.append({
                        "event":    title[:120],
                        "currency": currency,
                        "datetime": pub_dt,
                        "impact":   "HIGH",
                        "source":   source,
                    })
                    count += 1

                if count:
                    logger.debug(f"CalendarScanner: {count} high-impact items from {source}")

            except Exception as e:
                logger.debug(f"CalendarScanner RSS error ({source}): {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).drop_duplicates(subset=["event"])
        logger.info(f"✅ RSS calendar: {len(df)} high impact events cached")
        return df

    def _is_preview_article(self, text: str) -> bool:
        """Return True only when the text is a forward-looking preview, not a result."""
        return any(kw in text for kw in self.PREVIEW_KEYWORDS)

    def _detect_currency(self, text: str) -> str:
        for kw, currency in self.KEYWORD_CURRENCY.items():
            if kw in text:
                return currency
        return "UNKNOWN"

    @staticmethod
    def _parse_pub_date(entry) -> Optional[datetime]:
        """Parse feedparser entry date to timezone-aware UTC datetime."""
        try:
            import time as time_mod
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                ts = time_mod.mktime(entry.published_parsed)
                return datetime.fromtimestamp(ts, tz=pytz.utc)
        except Exception:
            pass
        return datetime.now(pytz.utc)

    def _is_cache_valid(self) -> bool:
        if self._cache_time is None or self._cache_df is None:
            return False
        age = (datetime.now(pytz.utc) - self._cache_time).total_seconds() / 60
        return age < self.CACHE_MINUTES

    # ── disabled MT5 method (kept to avoid ImportError elsewhere) ────────────
    @staticmethod
    def _get_mt5_events(*_, **__):
        return None
