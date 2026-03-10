# research/calendar_scanner.py
import re
import feedparser
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
from monitoring.logger import logger


class CalendarScanner:
    """
    Economic calendar using RSS feeds only (MT5 calendar API unavailable).

    FIX 6 — Calendar gate wrong timestamps:
        Uses _parse_event_time() → _parse_pub_date() → _estimated_event_time()
        priority chain to resolve the true economic event time rather than
        the article publication time.

    FILTER IMPROVEMENTS:
        - HIGH_IMPACT_KEYWORDS tightened to scheduled data releases only.
        - NOISE_KEYWORDS blocklist added to reject commentary/geopolitical
          articles that mention economic terms in passing.
        - UNKNOWN currency events are discarded (they can never block a
          known symbol and only add log noise).
        - Minimum title length check prevents very short feed entries.

    GODBOT v3.0 UPDATES:
        - CACHE_MINUTES reduced from 60 → 15 to match M5 scan cycle and
          prevent stale cache missing a news event within the hour.
        - Logger updated to use shared monitoring.logger instance directly,
          consistent with all other updated GOD_BOT modules.
        - minutes_before default now reads from CONFIG.NEWS_BLACKOUT_MINUTES
          (default 30) so it can be tuned from settings.py without touching
          this file.
    """

    RSS_FEEDS = {
        "fxstreet":  "https://www.fxstreet.com/rss/news",
        "investing":  "https://www.investing.com/rss/news_14.rss",
        "forexlive":  "https://www.forexlive.com/feed/news",
        "dailyfx":    "https://www.dailyfx.com/feeds/all",
    }

    # ── Tightened keyword list — scheduled releases and CB decisions only ─────
    # Each keyword here should only appear in articles about an actual
    # scheduled economic data release or a central bank policy decision.
    # Removed: "rate hike", "rate cut", "inflation" — too broad, match
    # commentary. Kept CB-specific terms (fomc, ecb, boe, boj) which are
    # almost always decision/statement articles.
    HIGH_IMPACT_KEYWORDS = [
        # US labour / employment
        "non-farm payroll", "nonfarm payroll", "nfp",
        "jobless claims", "initial claims", "continuing claims",
        "adp employment", "adp nonfarm",
        "employment change", "employment situation",
        "unemployment rate",

        # US inflation / spending
        "consumer price index", "cpi report", "cpi data", "cpi reading",
        "core cpi", "pce deflator", "core pce",
        "personal consumption expenditure",
        "producer price index", "ppi report",

        # US growth / activity
        "gdp report", "gdp reading", "gdp growth", "gdp data",
        "gross domestic product report",
        "retail sales report", "retail sales data",
        "durable goods orders", "durable goods report",
        "housing starts", "building permits",
        "ism manufacturing", "ism services", "ism pmi",
        "chicago pmi", "empire state",

        # Trade / current account
        "trade balance report", "trade deficit report",
        "current account report",

        # CB decisions / statements
        "fomc decision", "fomc statement", "fomc minutes",
        "fed rate decision", "federal reserve decision",
        "federal reserve statement", "federal reserve minutes",
        "ecb decision", "ecb rate decision", "ecb statement",
        "ecb minutes", "ecb press conference",
        "boe decision", "boe rate decision", "boe statement",
        "bank of england decision", "bank of england rate",
        "boj decision", "boj rate decision", "boj statement",
        "bank of japan decision",
        "rba decision", "rba rate decision", "rba statement",
        "boc decision", "boc rate decision", "boc statement",
        "bank of canada decision",
        "snb decision", "snb rate decision",

        # CB speaker events (formal testimony only)
        "powell testimony", "powell speaks", "powell statement",
        "lagarde testimony", "lagarde speaks", "lagarde statement",
        "bailey testimony", "bailey speaks", "bailey statement",

        # PMI data releases
        "flash pmi", "composite pmi", "manufacturing pmi",
        "services pmi", "pmi report", "pmi data",

        # Other tier-1 releases
        "nonfarm productivity", "unit labor costs",
        "michigan sentiment", "consumer confidence report",
        "jolts", "job openings",
    ]

    # ── Noise blocklist — reject articles matching any of these ───────────────
    # These patterns appear in commentary, geopolitical, and market-wrap
    # articles that mention economic terms in passing but are NOT scheduled
    # data releases. Any article matching a noise keyword is discarded
    # even if it also matches a HIGH_IMPACT_KEYWORD.
    NOISE_KEYWORDS = [
        # Geopolitical / war / sanctions
        "war", "military", "strike", "missile", "drone attack",
        "sanctions", "invasion", "conflict", "troops", "nato",
        "middle east", "iran", "russia", "ukraine", "taiwan",
        "bahrain", "oil refinery", "oil supply", "oil surge",
        "wti", "brent crude", "crude oil",

        # Market commentary / wraps
        "market wrap", "market news wrap", "markets wrap",
        "weekly wrap", "daily wrap", "asia-pacific wrap",
        "europe wrap", "americas wrap",
        "week ahead", "week in review",
        "what happened", "what to watch",
        "top stories", "morning briefing", "afternoon briefing",
        "daily briefing", "market briefing",
        "technical analysis", "chart analysis",
        "support and resistance",

        # Opinion / analysis pieces
        "opinion:", "analysis:", "commentary:",
        "what are the main events",
        "everything that trump",
        "hard to say how",
        "complicates", "brace for", "bracing for",
        "eyes on", "all eyes",
        "traders await", "markets await",
        "could affect", "may affect",

        # Price/commodity commentary (not releases)
        "jumps above", "surges above", "falls below",
        "largest one-day", "biggest one-day",
        "record high", "record low",
        "emergency reserve",
        "bond markets plunge", "bond markets surge",
        "stock markets", "equity markets",
    ]

    # ── Only block articles that are genuinely future-event previews ──────────
    PREVIEW_KEYWORDS = [
        "tomorrow",
        "next week",
        "next month",
        "upcoming week",
        "week ahead",
        "preview:",
        "events to watch",
        "trading week ahead",
        "economic week ahead",
        "scheduled for",
        "will be released",
        "set to release",
        "looking ahead",
        "ahead of next",
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
        # USD
        "fed":        "USD", "fomc":       "USD", "powell":    "USD",
        "payroll":    "USD", "nfp":         "USD", "jobless":   "USD",
        "ism":        "USD", "pce":         "USD", "durable":   "USD",
        "housing":    "USD", "retail sale": "USD", "jolts":     "USD",
        "michigan":   "USD", "u.s.":        "USD", "u.s ":      "USD",
        "us cpi":     "USD", "us gdp":      "USD", "us pmi":    "USD",
        "us jobs":    "USD", "us trade":    "USD",
        # EUR
        "ecb":        "EUR", "lagarde":     "EUR", "euro":      "EUR",
        "eurozone":   "EUR", "euro zone":   "EUR", "euro area":  "EUR",
        "german":     "EUR", "germany":     "EUR", "france":    "EUR",
        # GBP
        "boe":        "GBP", "bailey":      "GBP", "sterling":  "GBP",
        "u.k.":       "GBP", "uk cpi":      "GBP", "uk gdp":    "GBP",
        "uk jobs":    "GBP", "uk trade":    "GBP",
        # JPY
        "boj":        "JPY", "japan":       "JPY", "yen":       "JPY",
        "japanese":   "JPY",
        # AUD
        "rba":        "AUD", "australia":   "AUD", "aussie":    "AUD",
        "australian": "AUD",
        # CAD
        "boc":        "CAD", "canada":      "CAD", "loonie":    "CAD",
        "canadian":   "CAD",
        # XAU
        "gold":       "XAU", "xau":         "XAU",
    }

    # Minimum title word count — rejects very short feed entries
    MIN_TITLE_WORDS = 4

    # [FIX] Reduced from 60 → 15 minutes to match the M5 scan cycle.
    # A 60-minute stale cache could miss a news event published within
    # the hour, causing the bot to trade through high-impact releases.
    CACHE_MINUTES = 15

    EVENT_TIME_OFFSET_HOURS = 1

    _EVENT_TIME_PATTERNS = [
        re.compile(
            r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\s*(et|est|edt|gmt|utc|cet|cest|bst|jst)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bat\s+(\d{1,2}):(\d{2})\s*(am|pm)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdue\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:released?|out)\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)?\b",
            re.IGNORECASE,
        ),
    ]

    _TZ_OFFSETS: dict = {
        "et":   -5,
        "est":  -5,
        "edt":  -4,
        "gmt":   0,
        "utc":   0,
        "cet":   1,
        "cest":  2,
        "bst":   1,
        "jst":   9,
    }

    def __init__(self):
        self._cache_df:   Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime]      = None

    # ── Public API ────────────────────────────────────────────────────────────

    def is_safe_to_trade(
        self,
        symbol:         str,
        minutes_before: int = None,
        minutes_after:  int = 15,
    ) -> dict:
        """
        Return {'safe': bool, 'reason': str, 'events': list}.

        minutes_before defaults to CONFIG.NEWS_BLACKOUT_MINUTES (30) so it
        can be tuned from settings.py without touching this file.
        """
        # [FIX] Read blackout window from CONFIG with fallback to 30 minutes.
        if minutes_before is None:
            try:
                from config.settings import CONFIG
                minutes_before = int(getattr(CONFIG, "NEWS_BLACKOUT_MINUTES", 30))
            except Exception:
                minutes_before = 30

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
            result    = []
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
            logger.info("📅 No scheduled high-impact economic events today.")
            return
        logger.info(f"📅 Today's scheduled high-impact events ({len(events)}):")
        for e in events:
            t = e.get("datetime", "?")
            if hasattr(t, "strftime"):
                t = t.strftime("%H:%M UTC")
            time_src = e.get("time_source", "unknown")
            logger.info(
                f"   {t} | {e.get('currency','?'):4s} | "
                f"{e.get('event','?')}  [{time_src}]"
            )

    # ── Internal ──────────────────────────────────────────────────────────────

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
                feed  = feedparser.parse(url)
                count = 0
                for entry in feed.entries[:30]:
                    title   = getattr(entry, "title",   "") or ""
                    summary = getattr(entry, "summary", "") or ""
                    text    = f"{title} {summary}".lower()

                    # ── Reject very short titles ───────────────────────────
                    if len(title.split()) < self.MIN_TITLE_WORDS:
                        continue

                    # ── Reject noise / commentary articles ────────────────
                    if self._is_noise_article(text):
                        logger.debug(f"📰 Skipping noise: {title[:80]}")
                        continue

                    # ── Reject genuine preview / reminder articles ─────────
                    if self._is_preview_article(text):
                        logger.debug(f"📰 Skipping preview: {title[:80]}")
                        continue

                    # ── Keep only high-impact scheduled release articles ───
                    if not any(kw in text for kw in self.HIGH_IMPACT_KEYWORDS):
                        continue

                    # ── Detect currency — discard UNKNOWN events ───────────
                    # UNKNOWN events can never block a known symbol pair and
                    # only add log noise — discard them here.
                    currency = self._detect_currency(text)
                    if currency == "UNKNOWN":
                        logger.debug(
                            f"📰 Skipping unknown currency: {title[:80]}"
                        )
                        continue

                    # ── Resolve event time ────────────────────────────────
                    event_dt, time_source = self._resolve_event_time(
                        entry, title, summary
                    )

                    rows.append({
                        "event":       title[:120],
                        "currency":    currency,
                        "datetime":    event_dt,
                        "impact":      "HIGH",
                        "source":      source,
                        "time_source": time_source,
                    })
                    count += 1

                if count:
                    logger.debug(
                        f"CalendarScanner: {count} high-impact items from {source}"
                    )

            except Exception as e:
                logger.debug(f"CalendarScanner RSS error ({source}): {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).drop_duplicates(subset=["event"])
        logger.info(f"✅ RSS calendar: {len(df)} high-impact events cached")
        return df

    # ── Article classification helpers ────────────────────────────────────────

    def _is_noise_article(self, text: str) -> bool:
        """
        Return True if the article is commentary, geopolitical news, a
        market wrap, or any other non-scheduled-release content.
        These articles mention economic terms in passing but are not
        actual data release alerts — blocking trades based on them
        would produce false positives.
        """
        return any(kw in text for kw in self.NOISE_KEYWORDS)

    def _is_preview_article(self, text: str) -> bool:
        """Return True only when the text is a forward-looking preview."""
        return any(kw in text for kw in self.PREVIEW_KEYWORDS)

    def _detect_currency(self, text: str) -> str:
        for kw, currency in self.KEYWORD_CURRENCY.items():
            if kw in text:
                return currency
        return "UNKNOWN"

    # ── FIX 6: event-time resolution ─────────────────────────────────────────

    def _resolve_event_time(
        self,
        entry,
        title:   str,
        summary: str,
    ) -> tuple:
        combined_text = f"{title} {summary}"

        extracted = self._parse_event_time(combined_text, entry)
        if extracted is not None:
            return extracted, "text_extracted"

        pub_dt = self._parse_pub_date(entry)
        if pub_dt is not None:
            return pub_dt, "published_parsed"

        return self._estimated_event_time(), "estimated"

    def _parse_event_time(
        self,
        text:  str,
        entry,
    ) -> Optional[datetime]:
        pub_dt = self._parse_pub_date(entry)
        anchor = pub_dt if pub_dt is not None else datetime.now(pytz.utc)

        for pattern in self._EVENT_TIME_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue

            groups = match.groups()
            try:
                hour   = int(groups[0])
                minute = int(groups[1])
                ampm   = groups[2].lower() if len(groups) > 2 and groups[2] else None
                tz_str = groups[3].lower() if len(groups) > 3 and groups[3] else None

                if ampm == "pm" and hour != 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0

                if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                    continue

                naive_dt = anchor.replace(
                    hour=hour, minute=minute, second=0, microsecond=0,
                    tzinfo=None,
                )
                utc_dt = self._normalize_to_utc(naive_dt, tz_str)
                logger.debug(
                    f"CalendarScanner: extracted event time "
                    f"{utc_dt.strftime('%H:%M UTC')} (tz='{tz_str or 'none'}')"
                )
                return utc_dt

            except (ValueError, TypeError, AttributeError):
                continue

        return None

    @staticmethod
    def _normalize_to_utc(
        naive_dt: datetime,
        tz_str:   Optional[str],
    ) -> datetime:
        offset_hours = CalendarScanner._TZ_OFFSETS.get(
            (tz_str or "utc").lower(), 0
        )
        utc_dt = naive_dt - timedelta(hours=offset_hours)
        return utc_dt.replace(tzinfo=pytz.utc)

    def _estimated_event_time(self) -> datetime:
        return datetime.now(pytz.utc) + timedelta(hours=self.EVENT_TIME_OFFSET_HOURS)

    @staticmethod
    def _parse_pub_date(entry) -> Optional[datetime]:
        try:
            import time as time_mod
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                ts = time_mod.mktime(entry.published_parsed)
                return datetime.fromtimestamp(ts, tz=pytz.utc)
        except Exception:
            pass
        return None

    def _is_cache_valid(self) -> bool:
        if self._cache_time is None or self._cache_df is None:
            return False
        age = (datetime.now(pytz.utc) - self._cache_time).total_seconds() / 60
        return age < self.CACHE_MINUTES

    @staticmethod
    def _get_mt5_events(*_, **__):
        return None
