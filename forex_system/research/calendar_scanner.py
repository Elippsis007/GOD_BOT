# research/calendar_scanner.py
import MetaTrader5 as mt5
import feedparser
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
from monitoring.logger import get_logger

logger = get_logger("CalendarScanner")


class CalendarScanner:
    """
    Economic calendar using two reliable FREE sources:

    PRIMARY:  RSS feeds from FXStreet, Investing.com, ForexLive, DailyFX
              → Legitimate RSS, never blocked
              → They want you to use these feeds
              → Zero scraping

    NOTE:     MT5 built-in calendar API (calendar_value_history) does NOT
              exist in the MetaTrader5 Python package. The MT5 calendar
              is only available via the desktop terminal UI, not the Python
              API. All calendar data is sourced from RSS feeds only.
    """

    # ── RSS Feeds ─────────────────────────────────────────────────────────────
    CALENDAR_FEEDS = {
        "fxstreet":  "https://www.fxstreet.com/rss/news",
        "investing": "https://www.investing.com/rss/news_14.rss",
        "forexlive": "https://www.forexlive.com/feed/news",
        "dailyfx":   "https://www.dailyfx.com/feeds/all",
    }

    # ── High Impact Keywords ──────────────────────────────────────────────────
    HIGH_IMPACT_KEYWORDS = [
        "non-farm payroll", "nfp", "fed rate", "fomc",
        "federal reserve", "cpi", "inflation", "gdp",
        "unemployment", "interest rate decision",
        "powell", "jobs report",
        "ecb rate", "ecb decision", "lagarde",
        "eurozone cpi", "eurozone gdp",
        "boe rate", "bank of england", "bailey",
        "uk cpi", "uk gdp",
        "boj rate", "bank of japan", "ueda",
        "japan cpi", "japan gdp",
        "rba rate", "reserve bank australia",
        "australia cpi",
        "boc rate", "bank of canada",
        "canada cpi",
        "rate decision", "rate hike", "rate cut",
        "emergency meeting", "surprise cut",
        "surprise hike", "recession",
    ]

    # ── Symbol → Currencies ───────────────────────────────────────────────────
    SYMBOL_CURRENCIES = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "XAUUSD": ["XAU", "USD", "GOLD"],
    }

    # ── Keyword → Currency ────────────────────────────────────────────────────
    KEYWORD_CURRENCIES = {
        "USD": ["fed", "fomc", "powell", "nfp", "non-farm",
                "us cpi", "us gdp", "dollar", "treasury",
                "federal reserve", "us jobs"],
        "EUR": ["ecb", "lagarde", "eurozone", "euro",
                "european central bank"],
        "GBP": ["boe", "bailey", "uk cpi", "uk gdp",
                "bank of england", "sterling", "pound"],
        "JPY": ["boj", "ueda", "japan", "yen",
                "bank of japan"],
        "AUD": ["rba", "australia", "aussie",
                "reserve bank australia"],
        "CAD": ["boc", "canada", "loonie",
                "bank of canada"],
        "XAU": ["gold", "bullion", "precious metals"],
    }

    def __init__(self):
        self._cache_data: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime]     = None
        self.CACHE_MINUTES = 60

    # ── Main Safety Check ─────────────────────────────────────────────────────
    def is_safe_to_trade(
        self,
        symbol:         str,
        minutes_before: int = 30,
        minutes_after:  int = 15,
    ) -> dict:
        """
        Primary method called before every trade.
        Returns whether it is currently safe to trade the given symbol.
        """
        # RSS is the only working source — MT5 calendar API does not exist
        events = self._get_rss_events(symbol)

        if events is None or events.empty:
            return {
                "safe":   True,
                "reason": "No high impact events found",
                "events": [],
            }

        now     = datetime.now(pytz.utc)
        dangers = []

        for _, event in events.iterrows():
            event_time = event.get("datetime")
            if event_time is None or pd.isna(event_time):
                continue

            # Ensure timezone-aware
            if hasattr(event_time, "tzinfo") and event_time.tzinfo is None:
                event_time = pytz.utc.localize(event_time)

            mins_to    = (event_time - now).total_seconds() / 60
            mins_since = (now - event_time).total_seconds() / 60

            # >= 0 ensures the exact release minute is always blocked
            too_close_before = 0 <= mins_to    <= minutes_before
            too_close_after  = 0 <= mins_since <= minutes_after

            if too_close_before or too_close_after:
                dangers.append({
                    "event":    event.get("event", "Unknown"),
                    "currency": event.get("currency", ""),
                    "impact":   event.get("impact", "High"),
                    "time":     str(event_time),
                    "minutes":  round(mins_to, 1),
                })

        if dangers:
            return {
                "safe":   False,
                "reason": "High impact news event nearby",
                "events": dangers,
            }

        return {
            "safe":   True,
            "reason": "Clear of all high impact events",
            "events": [],
        }

    # ── MT5 Calendar — DISABLED ───────────────────────────────────────────────
    def _get_mt5_events(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        FIX: mt5.calendar_value_history() does not exist in the
        MetaTrader5 Python package. Calling it raises AttributeError
        on every scan cycle, spamming the log with:
            'module MetaTrader5 has no attribute calendar_value_history'

        The MT5 economic calendar is only accessible via the desktop
        terminal UI — it is not exposed through the Python API.
        This method now returns None immediately so the bot falls
        through to the RSS feed without any error being logged.
        """
        return None

    # ── RSS Calendar — PRIMARY (only working source) ──────────────────────────
    def _get_rss_events(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Reads legitimate RSS feeds.
        Checks cache first — only fetches if stale.
        """
        try:
            if self._is_cache_valid():
                df = self._cache_data.copy()
            else:
                df = self._fetch_all_rss_feeds()

            if df is None or df.empty:
                return None

            # Filter by positively identified currencies only —
            # excludes "UNKNOWN" so unidentified headlines don't
            # wrongly block all USD pairs.
            if symbol and symbol in self.SYMBOL_CURRENCIES:
                currencies = self.SYMBOL_CURRENCIES[symbol]
                mask = df["currency"].apply(
                    lambda c: c != "UNKNOWN" and any(
                        cur.upper() in str(c).upper()
                        for cur in currencies
                    )
                )
                df = df[mask]

            df = df[df["impact"] == "High"]
            return df.reset_index(drop=True) if not df.empty else None

        except Exception as e:
            logger.error(f"RSS calendar error: {e}")
            return None

    def _fetch_all_rss_feeds(self) -> Optional[pd.DataFrame]:
        """Fetches and parses all RSS calendar feeds. Updates the cache."""
        all_events = []

        for source, url in self.CALENDAR_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text    = f"{title} {summary}".lower()

                    is_high = any(kw in text for kw in self.HIGH_IMPACT_KEYWORDS)
                    if not is_high:
                        continue

                    currency = self._detect_currency(text)

                    published = entry.get("published_parsed")
                    if published:
                        ev_time = datetime(*published[:6], tzinfo=pytz.utc)
                    else:
                        ev_time = datetime.now(pytz.utc)

                    all_events.append({
                        "datetime": ev_time,
                        "currency": currency,
                        "impact":   "High",
                        "event":    title,
                        "source":   source,
                    })

            except Exception as e:
                logger.debug(f"RSS {source} error: {e}")
                continue

        if not all_events:
            self._cache_data = pd.DataFrame()
            self._cache_time = datetime.now()
            return None

        df = pd.DataFrame(all_events)
        self._cache_data = df.copy()
        self._cache_time = datetime.now()
        logger.info(f"✅ RSS calendar: {len(df)} high impact events cached")
        return df

    def _detect_currency(self, text: str) -> str:
        """
        Detects which currency an article is about.
        Returns "UNKNOWN" for unmatched headlines so they
        do not incorrectly block USD pairs.
        """
        for currency, keywords in self.KEYWORD_CURRENCIES.items():
            if any(kw.lower() in text for kw in keywords):
                return currency
        return "UNKNOWN"

    # ── Today's Events Summary ────────────────────────────────────────────────
    def get_todays_events(self) -> pd.DataFrame:
        """Returns all high impact events for today."""
        df = self._get_rss_events()

        if df is None or df.empty:
            return pd.DataFrame()

        today = datetime.now(pytz.utc).date()
        mask  = df["datetime"].apply(
            lambda x: x.date() == today if hasattr(x, "date") else False
        )
        return df[mask].reset_index(drop=True)

    def print_todays_events(self) -> None:
        """Pretty-prints today's high impact events to the console."""
        df = self.get_todays_events()
        print("\n📅 TODAY'S HIGH IMPACT EVENTS")
        print("=" * 55)
        if df.empty:
            print("  ✅ No high impact events today — clear to trade")
        else:
            for _, row in df.iterrows():
                dt = row["datetime"]
                time_str = (
                    dt.strftime("%H:%M")
                    if hasattr(dt, "strftime")
                    else "??:??"
                )
                print(
                    f"  🔴 {row['currency']:<5} | "
                    f"{time_str} UTC | "
                    f"{row['event'][:45]}"
                )
        print("=" * 55 + "\n")

    # ── Cache Validity ────────────────────────────────────────────────────────
    def _is_cache_valid(self) -> bool:
        if self._cache_data is None or self._cache_time is None:
            return False
        # total_seconds() gives full elapsed duration, not just the
        # seconds component of the timedelta
        age_minutes = (datetime.now() - self._cache_time).total_seconds() / 60
        return age_minutes < self.CACHE_MINUTES
