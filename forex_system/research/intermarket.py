# research/intermarket.py
"""
Intermarket analysis for GODBOT v3.0.

Pulls correlated market data via yfinance (free, no API key needed).
EURUSD-optimised — tracks DXY, VIX, SPX, GOLD.

Cache strategy:
  • In-memory cache (UTC-aware timestamps) — checked first.
  • Disk cache at data/intermarket_cache.json — checked on cold start.
  • Cache TTL = 6 hours (daily bars only update once per day;
    1-hour TTL caused unnecessary Yahoo re-fetches).
"""

import os
import json
import random
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from monitoring.logger import logger


# Fix 7 – Raise TTL from 1 hour to 6 hours; daily yfinance bars do not
# change intraday so hourly re-fetches are wasteful and trigger rate limits.
_CACHE_TTL_SECS = 21_600   # 6 hours


class IntermarketAnalyzer:
    """
    Pulls correlated market data via yfinance (free).
    EURUSD-optimised — only DXY, VIX, SPX, GOLD.
    """

    CACHE_FILE    = "data/intermarket_cache.json"
    FETCH_TIMEOUT = 8
    MAX_WORKERS   = 4

    TICKERS = {
        "DXY":  ["DX-Y.NYB", "UUP"],
        "VIX":  ["^VIX",     "VIXY"],
        "SPX":  ["^GSPC",    "SPY"],
        "GOLD": ["GC=F",     "GLD"],
    }

    CORRELATIONS = {
        "EURUSD": {
            "DXY":  -0.9,
            "VIX":  -0.3,
            "SPX":  +0.2,
            "GOLD": +0.3,
        },
    }

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._cache: dict               = {}
        # Fix 1 & 2 – Store cache timestamp as UTC-aware datetime throughout
        # so cross-module comparisons never raise offset-naive TypeError.
        self._cache_time: datetime | None = None

    # ── Disk cache ────────────────────────────────────────────────────────────
    def _load_disk_cache(self) -> dict | None:
        """
        Load market data from disk cache if it exists and is fresh.
        Returns the markets dict or None if cache is absent/stale/corrupt.
        """
        try:
            if not os.path.exists(self.CACHE_FILE):
                return None
            with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Fix 1 – Parse timestamp as UTC-aware so age subtraction is safe.
            raw_ts    = cached.get("timestamp", "")
            timestamp = datetime.fromisoformat(raw_ts)
            if timestamp.tzinfo is None:
                # Legacy cache written without timezone — treat as UTC.
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if age < _CACHE_TTL_SECS:
                logger.info(
                    f"📊 Intermarket: using disk cache "
                    f"({int(age / 60)} min old)"
                )
                return cached["markets"]
        except Exception as e:
            logger.debug(f"Disk cache read failed: {e}")
        return None

    def _save_disk_cache(self, markets: dict) -> None:
        """Persist market data to disk with a UTC-aware ISO timestamp."""
        try:
            with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        # Fix 1 – Save UTC-aware ISO string so future reads
                        # can parse it unambiguously.
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "markets":   markets,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.debug(f"Disk cache save failed: {e}")

    # ── Single ticker fetch ───────────────────────────────────────────────────
    def _fetch_single_ticker(self, ticker: str) -> tuple[str, object]:
        """
        Download the last 5 daily bars for `ticker` via yfinance.
        Returns (ticker, close_series) or (ticker, None) on failure.
        """
        import time
        # Jitter to reduce Yahoo rate-limit risk across parallel workers.
        time.sleep(random.uniform(0.1, 0.4))

        try:
            data = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=self.FETCH_TIMEOUT,
            )

            if data is None or len(data) < 2:
                return ticker, None

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"][ticker].dropna()
            else:
                close = data["Close"].dropna()

            if len(close) < 2:
                return ticker, None

            return ticker, close

        except Exception as e:
            if "RateLimit" in str(e) or "Too Many" in str(e):
                logger.warning(f"⚠️ Yahoo rate-limited on {ticker} — skipping")
            else:
                logger.debug(f"⚠️ {ticker} fetch error: {e}")
            return ticker, None

    # ── Parallel market fetch ─────────────────────────────────────────────────
    def _get_all_markets(self) -> dict:
        """
        Return a dict of market data for all TICKERS.
        Checks in-memory cache, then disk cache, then fetches fresh data.
        """
        # Fix 2 – Use UTC-aware datetime for in-memory cache age check.
        now = datetime.now(timezone.utc)

        if self._cache and self._cache_time:
            age = (now - self._cache_time).total_seconds()
            if age < _CACHE_TTL_SECS:
                return self._cache

        disk_data = self._load_disk_cache()
        if disk_data:
            self._cache      = disk_data
            self._cache_time = now
            return disk_data

        logger.info(
            "📊 Fetching fresh intermarket data "
            "(DXY, VIX, SPX, GOLD — parallel)…"
        )

        markets: dict = {}

        # Fix 3 – Removed dead _get_ticker_data() method; fetch logic lives
        # only here inside _fetch_market to avoid divergent code paths.
        def _fetch_market(market_name: str) -> tuple[str, dict]:
            tickers = self.TICKERS.get(market_name, [])
            for ticker in tickers:
                _, close = self._fetch_single_ticker(ticker)
                if close is not None and len(close) >= 2:
                    change_pct = float(
                        (close.iloc[-1] - close.iloc[-2])
                        / close.iloc[-2] * 100
                    )
                    logger.debug(
                        f"✅ {market_name} ({ticker}): {change_pct:+.2f}%"
                    )
                    return market_name, {
                        "ticker":     ticker,
                        "change_pct": round(change_pct, 3),
                        "price":      round(float(close.iloc[-1]), 4),
                        "ok":         True,
                    }
            logger.warning(
                f"⚠️ {market_name} unavailable — intermarket signal partial"
            )
            return market_name, {
                "ticker":     tickers[0] if tickers else market_name,
                "change_pct": 0.0,
                "price":      0.0,
                "ok":         False,
            }

        # Fix 4 – Call future.result() WITHOUT an inner timeout; as_completed
        # already enforces the outer deadline. The previous double-timeout
        # caused confusing race behaviour where futures could be abandoned
        # while still running.
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_market, name): name
                for name in self.TICKERS
            }
            try:
                for future in as_completed(
                    futures,
                    timeout=self.FETCH_TIMEOUT + 4,
                ):
                    try:
                        market_name, result = future.result()
                        markets[market_name] = result
                    except Exception as e:
                        market_name = futures[future]
                        logger.warning(
                            f"⚠️ {market_name} worker error: {e} "
                            f"— marked unavailable"
                        )
                        tickers = self.TICKERS.get(market_name, [market_name])
                        markets[market_name] = {
                            "ticker":     tickers[0],
                            "change_pct": 0.0,
                            "price":      0.0,
                            "ok":         False,
                        }
            except TimeoutError:
                # One or more markets did not complete within the outer
                # deadline — mark any that are still missing.
                logger.warning(
                    f"⚠️ Intermarket fetch timed out after "
                    f"{self.FETCH_TIMEOUT + 4}s — partial data used"
                )

        # Guarantee every expected market key exists.
        for name in self.TICKERS:
            if name not in markets:
                tickers = self.TICKERS.get(name, [name])
                markets[name] = {
                    "ticker":     tickers[0],
                    "change_pct": 0.0,
                    "price":      0.0,
                    "ok":         False,
                }

        ok_count = sum(1 for m in markets.values() if m.get("ok"))
        logger.info(
            f"📊 Intermarket: {ok_count}/{len(self.TICKERS)} markets loaded"
        )

        if ok_count > 0:
            self._save_disk_cache(markets)

        self._cache      = markets
        self._cache_time = now
        return markets

    # ── Signal ────────────────────────────────────────────────────────────────
    def get_intermarket_signal(self, symbol: str) -> dict:
        """
        Return a bias dict for `symbol` based on weighted correlated moves.

        Returns:
            {
                "bias":       "Bullish" | "Bearish" | "Neutral",
                "score":      float,   # normalised weighted score
                "confidence": float,   # 0.0 – 1.0
                "details":    dict,    # per-market breakdown
            }
        """
        default = {
            "bias":       "Neutral",
            "score":      0.0,
            "confidence": 0.0,
            "details":    {},
        }

        # Fix 8 – Normalise symbol before lookup so broker suffixes and
        # lowercase inputs (e.g. "EURUSDm", "eurusd") resolve correctly.
        symbol_key = symbol.upper()[:6]
        correlations = self.CORRELATIONS.get(symbol_key)
        if not correlations:
            logger.debug(
                f"No intermarket correlations defined for {symbol_key}"
            )
            return default

        markets = self._get_all_markets()
        score   = 0.0
        weight  = 0.0
        details = {}

        for market, corr in correlations.items():
            mdata = markets.get(market, {})
            if not mdata.get("ok"):
                continue
            chg          = mdata["change_pct"]
            contribution = chg * corr
            score       += contribution
            weight      += abs(corr)
            details[market] = {
                "change_pct":   chg,
                "correlation":  corr,
                "contribution": round(contribution, 4),
            }

        if weight == 0:
            return default

        norm_score = score / weight

        # Fix 5 – Scale so confidence reaches 1.0 at norm_score = 0.20
        # (a genuinely strong correlated move). Previous ×2 scale meant
        # any norm_score ≥ 0.5 gave confidence = 1.0, which is far too
        # easy to achieve on a routine intraday move.
        confidence = min(abs(norm_score) * 5, 1.0)

        if norm_score > 0.05:
            bias = "Bullish"
        elif norm_score < -0.05:
            bias = "Bearish"
        else:
            bias = "Neutral"

        return {
            "bias":       bias,
            "score":      round(norm_score, 4),
            "confidence": round(confidence, 4),
            "details":    details,
        }

    # ── Risk environment ──────────────────────────────────────────────────────
    def get_risk_environment(self) -> str:
        """
        Classify the current macro risk environment as Risk-On, Risk-Off,
        or Neutral using VIX level + change and SPX change.

        Fix 6 – Added absolute VIX level check alongside percentage change.
        VIX at 35 rising 6% is genuinely risk-off; VIX at 12 rising 6%
        is a minor uptick. Percentage change alone was misleading.
        """
        markets  = self._get_all_markets()
        vix_data = markets.get("VIX", {})
        spx_data = markets.get("SPX", {})

        if not vix_data.get("ok"):
            return "Unknown"

        vix_price = vix_data.get("price",      0.0)
        vix_chg   = vix_data.get("change_pct", 0.0)
        spx_chg   = (
            spx_data.get("change_pct", 0.0) if spx_data.get("ok") else 0.0
        )

        # Absolute VIX level ≥ 25 is inherently elevated regardless of direction.
        vix_elevated = vix_price >= 25

        if vix_elevated or vix_chg > 5 or spx_chg < -1:
            return "Risk-Off ⚠️"
        elif vix_chg < -3 and spx_chg > 0.5:
            return "Risk-On ✅"
        else:
            return "Neutral"

    # ── Summary ───────────────────────────────────────────────────────────────
    def print_market_summary(self) -> None:
        """
        Log a formatted intermarket overview to the shared logger.
        Fix 9 – Replaced print() calls with logger.info() so output
        passes through the MadridFormatter and can be filtered by level.
        """
        markets = self._get_all_markets()
        env     = self.get_risk_environment()
        lines   = [
            "",
            "📊 INTERMARKET OVERVIEW (EURUSD focused)",
            "=" * 50,
        ]

        for name, data in markets.items():
            if data.get("ok"):
                chg  = data["change_pct"]
                icon = "📈" if chg > 0 else "📉" if chg < 0 else "➡️"

                meaning = ""
                if name == "DXY":
                    meaning = "→ EUR bearish" if chg > 0 else "→ EUR bullish"
                elif name == "VIX":
                    meaning = (
                        "→ Risk-Off" if chg > 2
                        else "→ Risk-On" if chg < -2 else ""
                    )
                elif name == "SPX":
                    meaning = (
                        "→ EUR bearish" if chg < -0.5
                        else "→ EUR bullish" if chg > 0.5 else ""
                    )
                elif name == "GOLD":
                    meaning = (
                        "→ USD weak"    if chg > 0.3
                        else "→ USD strong" if chg < -0.3 else ""
                    )

                lines.append(
                    f"  {icon} {name:<6} {chg:+.2f}%"
                    f"  ({data['ticker']})  {meaning}"
                )
            else:
                lines.append(
                    f"  ⚠️  {name:<6} Unavailable — ({data['ticker']})"
                )

        lines.append(f"\n  Environment : {env}")

        result = self.get_intermarket_signal("EURUSD")
        bias   = result["bias"]
        score  = result["score"]
        icon   = (
            "🟢" if bias == "Bullish"
            else "🔴" if bias == "Bearish"
            else "⚪"
        )
        lines.append(f"  EURUSD Bias : {icon} {bias} (score {score:+.3f})")
        lines.append("=" * 50)

        for line in lines:
            logger.info(line)
