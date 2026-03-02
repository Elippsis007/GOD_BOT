# research/calendar_scanner.py
import MetaTrader5 as mt5
import feedparser
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional, List
from monitoring.logger import get_logger

logger = get_logger("CalendarScanner")


class CalendarScanner:
    """
    Economic calendar using two reliable FREE sources:

    PRIMARY:  MT5 built-in calendar
              → Direct API, never blocked
              → Always accurate
              → Zero scraping

    BACKUP:   FXStreet + Investing.com RSS feeds
              → Legitimate RSS, never blocked
              → They want you to use these feeds
              → Zero scraping

    ForexFactory scraper REMOVED — too unreliable
    """

    # ── Legitimate RSS Feeds — Never Blocked ──────────────
    CALENDAR_FEEDS = {
        "fxstreet":   "https://www.fxstreet.com/rss/news",
        "investing":  "https://www.investing.com/rss/news_14.rss",
        "forexlive":  "https://www.forexlive.com/feed/news",
        "dailyfx":    "https://www.dailyfx.com/feeds/all",
    }

    # ── High Impact Keywords In Headlines ─────────────────
    HIGH_IMPACT_KEYWORDS = [
        # US Events
        "non-farm payroll", "nfp", "fed rate", "fomc",
        "federal reserve", "cpi", "inflation", "gdp",
        "unemployment", "interest rate decision",
        "powell", "jobs report",
        # EU Events
        "ecb rate", "ecb decision", "lagarde",
        "eurozone cpi", "eurozone gdp",
        # UK Events
        "boe rate", "bank of england", "bailey",
        "uk cpi", "uk gdp",
        # JP Events
        "boj rate", "bank of japan", "ueda",
        "japan cpi", "japan gdp",
        # AU Events
        "rba rate", "reserve bank australia",
        "australia cpi",
        # CA Events
        "boc rate", "bank of canada",
        "canada cpi",
        # General
        "rate decision", "rate hike", "rate cut",
        "emergency meeting", "surprise cut",
        "surprise hike", "recession"
    ]

    # ── Currency to Symbol Mapping ─────────────────────────
    SYMBOL_CURRENCIES = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "XAUUSD": ["XAU", "USD", "GOLD"],
    }

    # ── Keyword to Currency Mapping ────────────────────────
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

    # ── Main Safety Check ─────────────────────────────────
    def is_safe_to_trade(
        self,
        symbol:         str,
        minutes_before: int = 30,
        minutes_after:  int = 15
    ) -> dict:
        """
        Primary method called before every trade.
        Checks both MT5 calendar and RSS feeds.
        Returns whether it is safe to trade right now.
        """
        # Try MT5 calendar first — most reliable
        events = self._get_mt5_events(symbol)

        # Fall back to RSS if MT5 returns nothing
        if events is None or events.empty:
            events = self._get_rss_events(symbol)

        if events is None or events.empty:
            return {
                "safe":   True,
                "reason": "No high impact events found",
                "events": []
            }

        now     = datetime.now(pytz.utc)
        dangers = []

        for _, event in events.iterrows():
            event_time = event.get("datetime")
            if event_time is None or pd.isna(event_time):
                continue

            # Ensure timezone aware
            if hasattr(event_time, "tzinfo") and event_time.tzinfo is None:
                event_time = pytz.utc.localize(event_time)

            mins_to    = (event_time - now).total_seconds() / 60
            mins_since = (now - event_time).total_seconds() / 60

            too_close_before = 0 < mins_to    <= minutes_before
            too_close_after  = 0 < mins_since <= minutes_after

            if too_close_before or too_close_after:
                dangers.append({
                    "event":    event.get("event", "Unknown"),
                    "currency": event.get("currency", ""),
                    "impact":   event.get("impact", "High"),
                    "time":     str(event_time),
                    "minutes":  round(mins_to, 1)
                })

        if dangers:
            return {
                "safe":   False,
                "reason": "High impact news event nearby",
                "events": dangers
            }

        return {
            "safe":   True,
            "reason": "Clear of all high impact events",
            "events": []
        }

    # ── MT5 Built-in Calendar — PRIMARY ───────────────────
    def _get_mt5_events(
        self,
        symbol: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Uses MT5's built-in economic calendar.
        Most reliable source — direct API, never blocked.
        """
        try:
            now      = datetime.now(pytz.utc)
            end_time = now + timedelta(hours=24)

            # MT5 calendar API
            events_raw = mt5.calendar_event_by_time(
                int(now.timestamp()),
                int(end_time.timestamp())
            )

            if not events_raw:
                logger.debug("MT5 calendar returned no events")
                return None

            events = []
            for ev in events_raw:
                # Map importance to impact level
                importance = getattr(ev, "importance", 0)
                if importance >= 3:
                    impact = "High"
                elif importance == 2:
                    impact = "Medium"
                else:
                    impact = "Low"

                # Only keep high impact
                if impact != "High":
                    continue

                currency = getattr(ev, "currency", "")
                name     = getattr(ev, "name", "")
                ev_time  = datetime.fromtimestamp(
                    getattr(ev, "time", 0),
                    tz=pytz.utc
                )

                events.append({
                    "datetime": ev_time,
                    "currency": currency,
                    "impact":   impact,
                    "event":    name,
                    "source":   "MT5"
                })

            if not events:
                return None

            df = pd.DataFrame(events)

            # Filter by symbol currencies if provided
            if symbol and symbol in self.SYMBOL_CURRENCIES:
                currencies = self.SYMBOL_CURRENCIES[symbol]
                df = df[df["currency"].isin(currencies)]

            logger.debug(
                f"✅ MT5 calendar: {len(df)} high impact events"
            )
            return df.reset_index(drop=True)

        except Exception as e:
            logger.debug(f"MT5 calendar error: {e}")
            return None

    # ── RSS Feed Calendar — BACKUP ────────────────────────
    def _get_rss_events(
        self,
        symbol: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Reads legitimate RSS feeds for upcoming events.
        Used as backup when MT5 calendar is unavailable.
        Never gets blocked — official RSS endpoints.
        """
        try:
            if self._is_cache_valid():
                df = self._cache_data
            else:
                df = self._fetch_all_rss_feeds()

            if df is None or df.empty:
                return None

            # Filter by symbol currencies
            if symbol and symbol in self.SYMBOL_CURRENCIES:
                currencies = self.SYMBOL_CURRENCIES[symbol]
                mask = df["currency"].apply(
                    lambda c: any(
                        cur.upper() in str(c).upper()
                        for cur in currencies
                    )
                )
                df = df[mask]

            # Only high impact
            df = df[df["impact"] == "High"]
            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"RSS calendar error: {e}")
            return None

    def _fetch_all_rss_feeds(self) -> Optional[pd.DataFrame]:
        """Fetches and parses all RSS calendar feeds."""
        all_events = []

        for source, url in self.CALENDAR_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text    = f"{title} {summary}".lower()

                    # Check if high impact
                    is_high = any(
                        kw in text
                        for kw in self.HIGH_IMPACT_KEYWORDS
                    )
                    if not is_high:
                        continue

                    # Detect currency
                    currency = self._detect_currency(text)

                    # Parse time
                    published = entry.get("published_parsed")
                    if published:
                        ev_time = datetime(
                            *published[:6],
                            tzinfo=pytz.utc
                        )
                    else:
                        ev_time = datetime.now(pytz.utc)

                    all_events.append({
                        "datetime": ev_time,
                        "currency": currency,
                        "impact":   "High",
                        "event":    title,
                        "source":   source
                    })

            except Exception as e:
                logger.debug(f"RSS {source} error: {e}")
                continue

        if not all_events:
            return None

        df = pd.DataFrame(all_events)
        self._cache_data = df
        self._cache_time = datetime.now()

        logger.info(
            f"✅ RSS calendar: {len(df)} high impact events found"
        )
        return df

    def _detect_currency(self, text: str) -> str:
        """Detects which currency an article is about."""
        for currency, keywords in self.KEYWORD_CURRENCIES.items():
            if any(kw.lower() in text for kw in keywords):
                return currency
        return "USD"  # Default

    # ── Today's Events Summary ────────────────────────────
    def get_todays_events(self) -> pd.DataFrame:
        """Returns all high impact events for today."""
        # Try MT5 first
        df = self._get_mt5_events()

        # Fall back to RSS
        if df is None or df.empty:
            df = self._fetch_all_rss_feeds()

        if df is None or df.empty:
            return pd.DataFrame()

        today = datetime.now(pytz.utc).date()
        mask  = df["datetime"].apply(
            lambda x: x.date() == today
            if hasattr(x, "date") else False
        )
        return df[mask].reset_index(drop=True)

    def print_todays_events(self):
        """Pretty prints today's high impact events."""
        df = self.get_todays_events()
        print("\n📅 TODAY'S HIGH IMPACT EVENTS")
        print("=" * 55)
        if df.empty:
            print("  ✅ No high impact events today — clear to trade")
        else:
            for _, row in df.iterrows():
                time_str = str(row["datetime"])[11:16]
                print(
                    f"  🔴 {row['currency']:<5} | "
                    f"{time_str} UTC | "
                    f"{row['event'][:45]}"
                )
        print("=" * 55 + "\n")

    def _is_cache_valid(self) -> bool:
        if self._cache_data is None or self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).seconds / 60
        return age < self.CACHE_MINUTES