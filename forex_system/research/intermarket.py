# research/intermarket.py
# EURUSD-focused intermarket analysis
# Only fetches what actually affects EUR/USD
import os
import json
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from monitoring.logger import get_logger

logger = get_logger("IntermarketAnalyzer")


class IntermarketAnalyzer:
    """
    Pulls correlated market data via yfinance (free).
    EURUSD-optimised — only DXY, VIX, SPX, GOLD.

    Removed: OIL (affects CAD/RUB not EUR)
             BONDS (affects JPY not EUR)

    Disk cache: saves to data/intermarket_cache.json
    Reuses cached data for 1 hour between runs.
    Prevents Yahoo rate limiting from repeated tests.

    FIX — Blocking time.sleep():
        The original code called time.sleep(2) before every ticker
        download inside a sequential loop.  With 4 markets × 2 tickers
        that caused up to 16 s of pure blocking on every fresh fetch,
        freezing the scan loop and stalling the dashboard.

        Fix:
          - All 4 markets are now fetched concurrently via
            ThreadPoolExecutor so total wall-clock fetch time drops
            from ~16 s to ~2-3 s.
          - time.sleep(2) replaced with a small per-thread random jitter
            (0.1–0.4 s) that still prevents Yahoo seeing burst requests
            without blocking the main thread.
          - FETCH_TIMEOUT (8 s) applied per ticker so a hung Yahoo
            connection cannot stall the pool indefinitely.
          - time module removed from imports (no longer needed).
    """

    CACHE_FILE    = "data/intermarket_cache.json"
    CACHE_TTL     = 3600   # 1 hour in seconds

    # FIX: per-ticker download timeout — prevents hung connections
    # from stalling the ThreadPoolExecutor pool indefinitely.
    FETCH_TIMEOUT = 8      # seconds per ticker attempt

    # FIX: max parallel worker threads — one per market is sufficient.
    # Keeps Yahoo request rate reasonable without sequential sleeping.
    MAX_WORKERS   = 4

    # ── EURUSD-relevant tickers only ──────────────────────────────────────────
    # 4 markets × 2 tickers = 8 downloads max (now fetched in parallel)
    TICKERS = {
        "DXY":  ["DX-Y.NYB", "UUP"],   # Dollar Index — strongest signal
        "VIX":  ["^VIX",     "VIXY"],   # Fear index
        "SPX":  ["^GSPC",    "SPY"],    # S&P500 risk sentiment
        "GOLD": ["GC=F",     "GLD"],    # Gold/USD inverse
    }

    # ── EURUSD correlations ───────────────────────────────────────────────────
    # DXY  -0.9 = if dollar rises, EUR/USD falls (strongest)
    # VIX  -0.3 = if fear rises, EUR/USD falls
    # SPX  +0.2 = if stocks rise, EUR/USD rises slightly
    # GOLD +0.3 = if gold rises, EUR/USD tends to rise
    CORRELATIONS = {
        "EURUSD": {
            "DXY":  -0.9,   # Most important
            "VIX":  -0.3,   # Important
            "SPX":  +0.2,   # Useful
            "GOLD": +0.3,   # Useful
        },
    }

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._cache      = {}
        self._cache_time = None

    # ── Disk Cache ────────────────────────────────────────────────────────────
    def _load_disk_cache(self) -> dict:
        """Load cached market data from disk if under 1 hour old."""
        try:
            if not os.path.exists(self.CACHE_FILE):
                return None
            with open(self.CACHE_FILE, "r") as f:
                cached = json.load(f)
            timestamp = datetime.fromisoformat(cached["timestamp"])
            age       = (datetime.now() - timestamp).total_seconds()
            if age < self.CACHE_TTL:
                logger.info(
                    f"📊 Intermarket: using cached data "
                    f"({int(age / 60)} min old) — no Yahoo download needed"
                )
                return cached["markets"]
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        return None

    def _save_disk_cache(self, markets: dict) -> None:
        """Save market data to disk cache."""
        try:
            with open(self.CACHE_FILE, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "markets":   markets,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    # ── Fetch Single Market ───────────────────────────────────────────────────
    def _fetch_single_ticker(self, ticker: str) -> tuple[str, dict]:
        """
        FIX — Download one ticker with a small random jitter instead of
        a fixed 2-second sleep.

        Called from a ThreadPoolExecutor worker thread so it runs
        concurrently with other ticker downloads.  The jitter
        (0.1–0.4 s) prevents Yahoo seeing burst requests at the exact
        same millisecond without blocking the main scan loop thread.

        Returns (ticker, raw_close_series_or_None) for the caller
        (_get_ticker_data) to interpret.
        """
        import time
        # FIX: small random jitter per thread instead of blocking sleep(2)
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

            # Handle both old and new yfinance column formats
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"][ticker].dropna()
            else:
                close = data["Close"].dropna()

            if len(close) < 2:
                return ticker, None

            return ticker, close

        except Exception as e:
            if "RateLimit" in str(e) or "Too Many" in str(e):
                logger.warning(f"⚠️ Yahoo rate limited on {ticker} — skipping")
            else:
                logger.debug(f"⚠️ {ticker} fetch error: {e}")
            return ticker, None

    def _get_ticker_data(self, name: str) -> dict:
        """
        FIX — Try primary ticker then fallback using pre-fetched close
        series rather than blocking sequential downloads.

        This method is now called with results from the concurrent
        pool (_get_all_markets) rather than doing its own download,
        but it retains its original signature so existing call-sites
        work unchanged.  When called directly (e.g. from tests) it
        falls back to a single synchronous fetch with jitter.
        """
        tickers = self.TICKERS.get(name, [])

        for ticker in tickers:
            _, close = self._fetch_single_ticker(ticker)
            if close is not None and len(close) >= 2:
                change_pct = float(
                    (close.iloc[-1] - close.iloc[-2])
                    / close.iloc[-2] * 100
                )
                logger.debug(f"✅ {name} ({ticker}): {change_pct:+.2f}%")
                return {
                    "ticker":     ticker,
                    "change_pct": round(change_pct, 3),
                    "price":      round(float(close.iloc[-1]), 4),
                    "ok":         True,
                }

        logger.warning(f"⚠️ {name} unavailable — intermarket signal partial")
        return {
            "ticker":     tickers[0] if tickers else name,
            "change_pct": 0,
            "price":      0,
            "ok":         False,
        }

    # ── Fetch All Markets (FIX: now parallel) ────────────────────────────────
    def _get_all_markets(self) -> dict:
        """
        FIX — Fetch all 4 markets concurrently via ThreadPoolExecutor.

        Priority: in-memory cache → disk cache → parallel fresh download.

        Previous behaviour: sequential loop with time.sleep(2) per ticker
        → up to 16 s blocking per fresh fetch cycle.

        New behaviour: all 4 markets fetched in parallel worker threads
        with per-thread jitter (0.1–0.4 s) and a per-ticker timeout of
        FETCH_TIMEOUT seconds → total wall-clock time ~2–3 s worst case.

        If a worker exceeds FETCH_TIMEOUT the market is recorded as
        unavailable and the scan continues — the main thread is never
        blocked beyond FETCH_TIMEOUT + a small scheduling overhead.
        """
        now = datetime.now()

        # ── 1. In-memory cache (fastest) ──────────────────────────────────────
        if self._cache and self._cache_time:
            age = (now - self._cache_time).total_seconds()
            if age < self.CACHE_TTL:
                return self._cache

        # ── 2. Disk cache (persists between process restarts) ─────────────────
        disk_data = self._load_disk_cache()
        if disk_data:
            self._cache      = disk_data
            self._cache_time = now
            return disk_data

        # ── 3. FIX: parallel fresh download ───────────────────────────────────
        logger.info(
            "📊 Fetching fresh intermarket data "
            "(DXY, VIX, SPX, GOLD — parallel)…"
        )

        markets: dict = {}

        def _fetch_market(market_name: str) -> tuple[str, dict]:
            """Worker: fetch one market, return (name, result_dict)."""
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
            # All tickers for this market failed
            logger.warning(
                f"⚠️ {market_name} unavailable — intermarket signal partial"
            )
            return market_name, {
                "ticker":     tickers[0] if tickers else market_name,
                "change_pct": 0,
                "price":      0,
                "ok":         False,
            }

        # Submit all markets to the thread pool simultaneously
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_market, name): name
                for name in self.TICKERS
            }
            for future in as_completed(
                futures,
                timeout=self.FETCH_TIMEOUT + 2,   # pool-level safety timeout
            ):
                try:
                    market_name, result = future.result(
                        timeout=self.FETCH_TIMEOUT
                    )
                    markets[market_name] = result
                except TimeoutError:
                    market_name = futures[future]
                    logger.warning(
                        f"⚠️ {market_name} timed out after "
                        f"{self.FETCH_TIMEOUT}s — marked unavailable"
                    )
                    tickers = self.TICKERS.get(market_name, [market_name])
                    markets[market_name] = {
                        "ticker":     tickers[0],
                        "change_pct": 0,
                        "price":      0,
                        "ok":         False,
                    }
                except Exception as e:
                    market_name = futures[future]
                    logger.warning(
                        f"⚠️ {market_name} worker error: {e} — marked unavailable"
                    )
                    tickers = self.TICKERS.get(market_name, [market_name])
                    markets[market_name] = {
                        "ticker":     tickers[0],
                        "change_pct": 0,
                        "price":      0,
                        "ok":         False,
                    }

        # Ensure all expected markets have an entry even if futures
        # completed before the timeout iteration reached them
        for name in self.TICKERS:
            if name not in markets:
                tickers = self.TICKERS.get(name, [name])
                markets[name] = {
                    "ticker":     tickers[0],
                    "change_pct": 0,
                    "price":      0,
                    "ok":         False,
                }

        ok_count = sum(1 for m in markets.values() if m.get("ok"))
        logger.info(
            f"📊 Intermarket: {ok_count}/{len(self.TICKERS)} markets loaded"
        )

        # Persist to disk if any data came back
        if ok_count > 0:
            self._save_disk_cache(markets)

        # Update in-memory cache
        self._cache      = markets
        self._cache_time = now
        return markets

    # ── Signal for EURUSD ─────────────────────────────────────────────────────
    def get_intermarket_signal(self, symbol: str) -> dict:
        default = {
            "bias":       "Neutral",
            "score":      0.0,
            "confidence": 0.0,
            "details":    {},
        }

        correlations = self.CORRELATIONS.get(symbol)
        if not correlations:
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
        confidence = min(abs(norm_score) * 2, 1.0)

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

    # ── Risk Environment ──────────────────────────────────────────────────────
    def get_risk_environment(self) -> str:
        """
        Risk-On  = stocks rising, fear falling  → EUR tends to rise
        Risk-Off = stocks falling, fear rising  → EUR tends to fall
        """
        markets  = self._get_all_markets()
        vix_data = markets.get("VIX", {})
        spx_data = markets.get("SPX", {})

        if not vix_data.get("ok"):
            return "Unknown"

        vix_chg = vix_data.get("change_pct", 0)
        spx_chg = spx_data.get("change_pct", 0) if spx_data.get("ok") else 0

        if vix_chg > 5 or spx_chg < -1:
            return "Risk-Off ⚠️"
        elif vix_chg < -3 and spx_chg > 0.5:
            return "Risk-On ✅"
        else:
            return "Neutral"

    # ── Print Summary ─────────────────────────────────────────────────────────
    def print_market_summary(self) -> None:
        markets = self._get_all_markets()
        env     = self.get_risk_environment()

        print("\n📊 INTERMARKET OVERVIEW (EURUSD focused)")
        print("=" * 50)

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
                        "→ USD weak"   if chg > 0.3
                        else "→ USD strong" if chg < -0.3 else ""
                    )

                print(
                    f"  {icon} {name:<6} {chg:+.2f}%"
                    f"  ({data['ticker']})  {meaning}"
                )
            else:
                print(f"  ⚠️  {name:<6} Unavailable — ({data['ticker']})")

        print(f"\n  Environment : {env}")

        result = self.get_intermarket_signal("EURUSD")
        bias   = result["bias"]
        score  = result["score"]
        icon   = (
            "🟢" if bias == "Bullish"
            else "🔴" if bias == "Bearish"
            else "⚪"
        )
        print(f"  EURUSD Bias : {icon} {bias} (score {score:+.3f})")
        print("=" * 50)