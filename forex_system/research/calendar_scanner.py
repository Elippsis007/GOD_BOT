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

    # ── Preview/Reminder Filter — these are NOT live events ──────────────────
    # Headlines containing these phrases are articles written ABOUT
    # upcoming events, not the actual releases. They should never
    # trigger a news block.
    PREVIEW_KEYWORDS = [
        "tomorrow",
        "next week",
        "reminder",
        "preview",
        "ahead of",
        "scheduled for",
        "looking ahead",
        "what to expect",
        "week ahead",
        "what to watch",
        "market preview",
        "economic preview",
        "forecast",
        "expectations for",
        "what we know",
        "coming up",
        "prepare for",
        "traders await",
        "markets await",
        "eyes on",
        "watch out for",
        "on the horizon",
        "due tomorrow",
        "due next",
        "due friday",
        "due monday",
        "due tuesday",
        "due wednesday",
        "due thursday",
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

    # ── Preview Detection ─────────────────────────────────────────────────────
    @classmethod
    def _is_preview_article(cls, text: str) -> bool:
        """
        Returns True if the headline/summary is a preview or reminder
        article about a future event rather than an actual live release.
        These should NOT trigger a news block.
        """
        text_lower = text.lower()
        return any(kw in text_lower for kw in cls.PREVIEW_KEYWORDS)

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
        return None

    # ── RSS Calendar — PRIMARY ────────────────────────────────────────────────
    def _get_rss_events(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            if self._is_cache_valid():
                df = self._cache_data.copy()
            else:
                df = self._fetch_all_rss_feeds()

            if df is None or df.empty:
                return None

            # Filter by positively identified currencies only
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

                    # ── Skip preview / reminder articles ──────────────────
                    # These headlines mention future events but are not
                    # actual releases — they must not trigger a news block.
                    if self._is_preview_article(text):
                        logger.debug(
                            f"📰 Skipping preview article: {title[:60]}"
                        )
                        continue

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
        Returns UNKNOWN for unmatched headlines so they
        do not incorrectly block USD pairs.
        """
        for currency, keywords in self.KEYWORD_CURRENCIES.items():
            if any(kw.lower() in text for kw in keywords):
                return currency
        return "UNKNOWN"

    # ── Today's Events Summary ────────────────────────────────────────────────
    def get_todays_events(self) -> pd.DataFrame:
        df = self._get_rss_events()
        if df is None or df.empty:
            return pd.DataFrame()
        today = datetime.now(pytz.utc).date()
        mask  = df["datetime"].apply(
            lambda x: x.date() == today if hasattr(x, "date") else False
        )
        return df[mask].reset_index(drop=True)

    def print_todays_events(self) -> None:
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
        age_minutes = (datetime.now() - self._cache_time).total_seconds() / 60
        return age_minutes < self.CACHE_MINUTES
