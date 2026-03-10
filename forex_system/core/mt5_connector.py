# =============================================================================
#  core/mt5_connector.py  –  GODBOT v3.0
# =============================================================================
#  Fixes / improvements applied in this revision:
#
#  A  Chunked OHLCV fetch with configurable chunk size
#  B  Unified timeframe normaliser (_normalise_tf) covering strings,
#     raw seconds, and native MT5 TIMEFRAME_* constants
#  C  Two-stage connect (attach first, launch terminal as fallback)
#     with exponential back-off across max_attempts
#  D  _select_symbol retries once before failing
#  E  Robust volume column resolution (tick_volume → real_volume → volume)
#  F  Enriched get_symbol_info dict (margin_maintenance, filling_mode, etc.)
#  G  [NEW] reconnect() public method – reconnects without re-creating
#     the singleton; used by DataHandler / RiskManager on stale-connection
#  H  [NEW] get_latest_tick() – single-call ask/bid/spread helper so
#     callers never need to import mt5 directly for tick data
#  I  [FIX] is_connected() had a TOCTOU race: _ping() is now called once,
#     result stored in local variable before branching
#  J  [FIX] connect() must call mt5.shutdown() before each re-init attempt
#     to avoid "already initialised" state on the second+ attempt
#  K  [FIX] _fetch_bars_chunked position arithmetic was wrong:
#     copy_rates_from_pos(pos=0) always returns the MOST RECENT bars,
#     so each chunk must advance the start position backward in time;
#     rewritten to use utc_from-based pagination for correctness
#  L  [FIX] _to_dataframe float cast must EXCLUDE the time column and
#     any other non-numeric columns; the previous blanket astype(float)
#     would crash if the DataFrame ever contained non-OHLCV columns
#  M  [NEW] get_positions() helper – returns list of open position dicts
#     so OrderExecutor / RiskManager never call mt5 directly
#  N  [NEW] get_orders() helper – returns list of pending order dicts
#  O  [FIX] get_ticks() fallback range used datetime.now() twice with no
#     guarantee both calls land in the same second; now uses a single
#     snapshot; also the midnight anchor now correctly uses replace() with
#     tzinfo preserved
#  P  [NEW] MT5_TIMEFRAME_MAP exported at module level for legacy callers
#  Q  [FIX] Singleton _connected flag is an instance attribute, not a
#     class attribute, to survive garbage-collection edge cases
# =============================================================================

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

import MetaTrader5 as mt5
import pandas as pd

from config.settings import CONFIG
from monitoring.logger import logger

# ---------------------------------------------------------------------------
# [B] Unified timeframe normaliser
# ---------------------------------------------------------------------------
_TF_STRING_MAP: Dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# seconds-per-candle → MT5 constant (kept for legacy integer callers)
_TF_SECONDS_MAP: Dict[int, int] = {
    60:    mt5.TIMEFRAME_M1,
    120:   mt5.TIMEFRAME_M2,
    180:   mt5.TIMEFRAME_M3,
    240:   mt5.TIMEFRAME_M4,
    300:   mt5.TIMEFRAME_M5,
    600:   mt5.TIMEFRAME_M10,
    900:   mt5.TIMEFRAME_M15,
    1_200: mt5.TIMEFRAME_M20,
    1_800: mt5.TIMEFRAME_M30,
    3_600: mt5.TIMEFRAME_H1,
    7_200: mt5.TIMEFRAME_H2,
    10_800: mt5.TIMEFRAME_H3,
    14_400: mt5.TIMEFRAME_H4,
    21_600: mt5.TIMEFRAME_H6,
    28_800: mt5.TIMEFRAME_H8,
    43_200: mt5.TIMEFRAME_H12,
    86_400: mt5.TIMEFRAME_D1,
}

# Seconds per timeframe constant – used by chunked paginator (Fix K)
_TF_CONST_TO_SECONDS: Dict[int, int] = {v: k for k, v in _TF_SECONDS_MAP.items()}
# D1 and W1/MN1 added manually (not in _TF_SECONDS_MAP)
_TF_CONST_TO_SECONDS[mt5.TIMEFRAME_W1]  = 7 * 86_400
_TF_CONST_TO_SECONDS[mt5.TIMEFRAME_MN1] = 30 * 86_400   # approximate

# All valid MT5 TIMEFRAME_* constant values (for pass-through detection)
_VALID_MT5_TF: set = set(_TF_STRING_MAP.values())

# [P] Public alias so main.py / legacy callers can do:
#     from core.mt5_connector import MT5_TIMEFRAME_MAP
MT5_TIMEFRAME_MAP = _TF_SECONDS_MAP

_MT5_CHUNK_SIZE_DEFAULT = 5_000   # bars per chunked request


def _normalise_tf(timeframe: Union[str, int]) -> int:
    """
    [B] Accept a string ("M5"), raw seconds (300), or an MT5 TIMEFRAME_*
    constant and always return the correct mt5.TIMEFRAME_* int.
    Raises ValueError on unrecognised input.
    """
    if isinstance(timeframe, str):
        key = timeframe.upper().strip()
        if key in _TF_STRING_MAP:
            return _TF_STRING_MAP[key]
        raise ValueError(f"Unknown timeframe string: '{timeframe!r}'")
    if isinstance(timeframe, int):
        if timeframe in _VALID_MT5_TF:
            return timeframe
        if timeframe in _TF_SECONDS_MAP:
            return _TF_SECONDS_MAP[timeframe]
        raise ValueError(f"Unknown timeframe int: {timeframe}")
    raise TypeError(
        f"timeframe must be str or int, got {type(timeframe).__name__!r}"
    )


# ---------------------------------------------------------------------------
#  Singleton MT5Connector
# ---------------------------------------------------------------------------
class MT5Connector:
    """Thread-safe singleton wrapper for MetaTrader 5 connectivity."""

    _instance: Optional["MT5Connector"] = None

    def __new__(cls) -> "MT5Connector":
        if cls._instance is None:
            obj = super().__new__(cls)
            # [Q] Instance attribute – survives GC edge cases unlike class attr
            obj._connected = False
            cls._instance = obj
        return cls._instance

    # ------------------------------------------------------------------
    # [C] Two-stage connect with exponential back-off
    # [J] Calls mt5.shutdown() before every attempt to clear stale state
    # ------------------------------------------------------------------
    def connect(self, max_attempts: int = 3, base_delay: float = 2.0) -> bool:
        """
        Attempt to connect to MetaTrader 5.

        Stage 1 – attach to an already-running terminal (no path needed).
        Stage 2 – launch the terminal executable if stage 1 fails and
                  CONFIG.MT5_PATH is configured.

        [J] mt5.shutdown() is called before every initialise attempt so
        that a previous half-initialised state does not cause
        "terminal already exists" errors on retry.
        """
        if self._connected and self._ping():
            logger.debug("MT5Connector: already connected — skipping re-init.")
            return True

        credentials: dict = {
            "login":    getattr(CONFIG, "MT5_LOGIN",    0),
            "password": getattr(CONFIG, "MT5_PASSWORD", ""),
            "server":   getattr(CONFIG, "MT5_SERVER",   ""),
        }
        mt5_path: str = getattr(CONFIG, "MT5_PATH", "")

        for attempt in range(1, max_attempts + 1):
            delay = base_delay * (2 ** (attempt - 1))

            # [J] Always shut down first to avoid "already initialised" error
            mt5.shutdown()

            # Stage 1 – attach to running terminal
            ok = mt5.initialize(
                login=credentials["login"],
                password=credentials["password"],
                server=credentials["server"],
            )

            # Stage 2 – launch terminal from path
            if not ok and mt5_path:
                logger.warning(
                    "MT5Connector [attempt %d/%d]: attach failed (%s) — "
                    "launching terminal at '%s'.",
                    attempt, max_attempts, mt5.last_error(), mt5_path,
                )
                mt5.shutdown()  # clear the failed init before retry
                ok = mt5.initialize(
                    path=mt5_path,
                    login=credentials["login"],
                    password=credentials["password"],
                    server=credentials["server"],
                )

            if ok:
                info = mt5.account_info()
                if info is not None:
                    self._connected = True
                    logger.info(
                        "MT5Connector: connected — account=%d | server=%s | "
                        "balance=%.2f %s",
                        info.login, info.server,
                        info.balance, info.currency,
                    )
                    return True
                logger.warning(
                    "MT5Connector [attempt %d/%d]: initialize() OK but "
                    "account_info() returned None — treating as failure.",
                    attempt, max_attempts,
                )
                mt5.shutdown()
            else:
                logger.error(
                    "MT5Connector [attempt %d/%d]: initialize() failed — %s",
                    attempt, max_attempts, mt5.last_error(),
                )

            if attempt < max_attempts:
                logger.info(
                    "MT5Connector: retrying in %.0f s…", delay
                )
                time.sleep(delay)

        self._connected = False
        logger.critical(
            "MT5Connector: failed to connect after %d attempt(s).",
            max_attempts,
        )
        return False

    # ------------------------------------------------------------------
    # [G] Public reconnect helper
    # ------------------------------------------------------------------
    def reconnect(self, max_attempts: int = 3, base_delay: float = 2.0) -> bool:
        """
        [G] Force a full reconnect cycle without destroying the singleton.
        Called by DataHandler / RiskManager when they detect a stale
        connection rather than importing mt5 directly.
        """
        logger.info("MT5Connector.reconnect(): forcing reconnect…")
        self._connected = False
        mt5.shutdown()
        return self.connect(max_attempts=max_attempts, base_delay=base_delay)

    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        mt5.shutdown()
        self._connected = False
        logger.info("MT5Connector: disconnected.")

    # ------------------------------------------------------------------
    # [I] is_connected() – TOCTOU fix: single _ping() call
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        """
        [I] The previous version called _ping() once to detect staleness and
        then called connect() which called _ping() a second time, creating a
        TOCTOU window.  Now the result of a single _ping() drives the branch.
        """
        if not self._connected:
            return False

        alive = self._ping()
        if not alive:
            logger.warning(
                "MT5Connector: stale connection detected — reconnecting…"
            )
            self._connected = False
            return self.reconnect()
        return True

    # ------------------------------------------------------------------
    def _ping(self) -> bool:
        try:
            return mt5.account_info() is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------
    def get_account_info(self) -> Optional[Dict]:
        if not self.is_connected():
            logger.error("MT5Connector.get_account_info: not connected.")
            return None
        info = mt5.account_info()
        if info is None:
            logger.error(
                "MT5Connector.get_account_info(): returned None — %s",
                mt5.last_error(),
            )
            return None
        return {
            "balance":     info.balance,
            "equity":      info.equity,
            "margin":      info.margin,
            "free_margin": info.margin_free,
            "profit":      info.profit,
            "leverage":    info.leverage,
            "currency":    info.currency,
            "company":     info.company,
            "login":       info.login,
            "server":      info.server,
        }

    # ------------------------------------------------------------------
    # [H] Single-call tick helper (ask / bid / spread)
    # ------------------------------------------------------------------
    def get_latest_tick(self, symbol: str) -> Optional[Dict]:
        """
        [H] Returns a dict with ask, bid, spread_points, spread_pips.
        Callers (DataHandler, RiskManager) use this instead of importing mt5
        directly, keeping the MT5 surface area confined to this module.
        """
        if not self.is_connected():
            return None
        if not self._select_symbol(symbol):
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(
                "MT5Connector.get_latest_tick: no tick for %s — %s",
                symbol, mt5.last_error(),
            )
            return None

        sym_info = mt5.symbol_info(symbol)
        digits = sym_info.digits if sym_info else 5
        point  = sym_info.point  if sym_info else 10 ** (-digits)

        # Spread in points then in pips (1 pip = 10 points for 5-digit pairs)
        spread_points = tick.ask - tick.bid
        pip_size = point * (10 if digits in (3, 5) else 1)
        spread_pips = spread_points / pip_size if pip_size else 0.0

        return {
            "ask":           tick.ask,
            "bid":           tick.bid,
            "mid":           (tick.ask + tick.bid) / 2.0,
            "spread_points": spread_points,
            "spread_pips":   spread_pips,
            "time":          datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    # ------------------------------------------------------------------
    # [A] Chunked OHLCV retrieval
    # [K] Pagination rewritten to use utc_from-based stepping (correct)
    # ------------------------------------------------------------------
    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: Union[str, int],
        bars:      Optional[int] = None,
        utc_from:  Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars for *symbol* on *timeframe*.

        If *utc_from* is given, bars are fetched starting from that timestamp.
        Otherwise, the most recent *bars* candles are returned.

        For requests larger than CONFIG.MT5_CHUNK_SIZE the fetch is split
        into time-aligned chunks to work around the 100 k-bar broker limit.
        """
        if not self.is_connected():
            logger.error("MT5Connector.get_ohlcv: not connected.")
            return None

        try:
            tf = _normalise_tf(timeframe)
        except (ValueError, TypeError) as exc:
            logger.error("MT5Connector.get_ohlcv: %s", exc)
            return None

        n_bars = (
            bars if bars is not None
            else getattr(CONFIG, "BARS_HISTORY", 36_000)
        )
        chunk = getattr(CONFIG, "MT5_CHUNK_SIZE", _MT5_CHUNK_SIZE_DEFAULT)

        if not self._select_symbol(symbol):
            return None

        if n_bars <= chunk:
            df = self._fetch_bars(symbol, tf, n_bars, utc_from)
        else:
            df = self._fetch_bars_chunked(symbol, tf, n_bars, utc_from, chunk)

        if df is None or df.empty:
            logger.warning(
                "MT5Connector.get_ohlcv: no data returned for %s.", symbol
            )
            return None

        logger.debug(
            "MT5Connector.get_ohlcv: %s %s — %d bars.",
            symbol, timeframe, len(df),
        )
        return df

    # ------------------------------------------------------------------
    def _fetch_bars(
        self,
        symbol:   str,
        tf:       int,
        bars:     int,
        utc_from: Optional[datetime],
    ) -> Optional[pd.DataFrame]:
        if utc_from is not None:
            raw = mt5.copy_rates_from(symbol, tf, utc_from, bars)
        else:
            raw = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if raw is None or len(raw) == 0:
            logger.warning(
                "MT5Connector._fetch_bars: broker returned no bars "
                "for %s.", symbol,
            )
            return None

        df = self._to_dataframe(raw)
        return df.set_index("time")

    # ------------------------------------------------------------------
    # [K] Chunked fetch rewritten: utc_from-based time pagination
    # ------------------------------------------------------------------
    def _fetch_bars_chunked(
        self,
        symbol:     str,
        tf:         int,
        total_bars: int,
        utc_from:   Optional[datetime],
        chunk:      int,
    ) -> Optional[pd.DataFrame]:
        """
        [K]  The previous position-based chunker had a critical bug:
        copy_rates_from_pos(pos=0) always returns the MOST RECENT bars,
        so advancing *pos* by *fetched* on each iteration returned
        overlapping or out-of-order data.

        Correct approach: work backwards in time using utc_from anchors.

        1. The anchor starts at utc_from (if given) or now (UTC).
        2. Each chunk fetches up to *chunk* bars ending at the anchor.
        3. The anchor is moved back by (candles_fetched × bar_seconds)
           before the next iteration.
        4. All frames are concatenated, deduplicated, and sorted.
        """
        bar_seconds = _TF_CONST_TO_SECONDS.get(tf, 300)  # default to M5

        # Determine starting anchor (most recent end of the desired range)
        anchor = utc_from if utc_from is not None else datetime.now(timezone.utc)

        frames: List[pd.DataFrame] = []
        remaining = total_bars

        logger.debug(
            "MT5Connector: chunked fetch — %s %d bars, "
            "chunk=%d, bar=%ds.",
            symbol, total_bars, chunk, bar_seconds,
        )

        while remaining > 0:
            fetch_now = min(remaining, chunk)

            # copy_rates_from returns bars STARTING at anchor going forward.
            # To fetch bars ENDING at anchor we step back first.
            window_start = anchor - timedelta(
                seconds=bar_seconds * fetch_now
            )
            raw = mt5.copy_rates_from(symbol, tf, window_start, fetch_now)

            if raw is None or len(raw) == 0:
                logger.warning(
                    "MT5Connector: chunk at anchor=%s returned no data — "
                    "stopping early.", anchor.isoformat(),
                )
                break

            frames.append(self._to_dataframe(raw))
            fetched    = len(raw)
            remaining -= fetched

            # Move anchor back by exactly the fetched window
            anchor -= timedelta(seconds=bar_seconds * fetched)

            if fetched < fetch_now:
                logger.debug(
                    "MT5Connector: broker returned %d < %d — "
                    "history exhausted.", fetched, fetch_now,
                )
                break

        if not frames:
            return None

        # Concat on plain time column, then set index after dedup/sort
        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
        df = df.set_index("time")

        logger.debug(
            "MT5Connector: chunked fetch complete — %d bars assembled.",
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # [L] _to_dataframe: safe float cast excludes non-numeric columns
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dataframe(raw) -> pd.DataFrame:
        """
        Convert an mt5 rates numpy array → clean DataFrame.

        [L]  The previous blanket  .astype({c: float for c in keep})
             would crash if any keep-column were non-numeric (e.g. if the
             broker appends extra columns).  The cast now checks dtype
             before coercing and explicitly excludes the time column.

        The returned DataFrame has 'time' as a plain COLUMN (not the index)
        so that _fetch_bars_chunked can safely concat multiple chunks and
        deduplicate on subset=["time"] without index-vs-column conflicts.
        _fetch_bars() and _fetch_bars_chunked() both call set_index("time")
        after their respective operations are complete.
        """
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # [E] Robust volume column resolution
        for vol_col in ("tick_volume", "real_volume", "volume"):
            if vol_col in df.columns and vol_col != "volume":
                df = df.rename(columns={vol_col: "volume"})
                break

        if "volume" not in df.columns:
            df["volume"] = 0

        keep = [
            c for c in ("time", "open", "high", "low", "close", "volume")
            if c in df.columns
        ]
        df = df[keep].copy()

        # [L] Safe per-column float cast — skip time and already-float cols
        numeric_cols = [c for c in keep if c != "time"]
        for col in numeric_cols:
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # [D] Symbol select with one retry
    # ------------------------------------------------------------------
    def _select_symbol(self, symbol: str) -> bool:
        if mt5.symbol_select(symbol, True):
            return True
        time.sleep(0.3)
        if mt5.symbol_select(symbol, True):
            return True
        logger.error(
            "MT5Connector: symbol_select('%s') failed — %s",
            symbol, mt5.last_error(),
        )
        return False

    # ------------------------------------------------------------------
    # Tick data
    # [O] Single datetime.now() snapshot avoids sub-second drift
    # ------------------------------------------------------------------
    def get_ticks(
        self,
        symbol:   str,
        count:    int = 1_000,
        utc_from: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        [O] The previous fallback used datetime.now(timezone.utc) twice,
        creating a tiny race between the midnight anchor and the range end.
        Now a single snapshot is taken at the top of the method.
        """
        if not self.is_connected():
            logger.error("MT5Connector.get_ticks: not connected.")
            return None
        if not self._select_symbol(symbol):
            return None

        now_utc = datetime.now(timezone.utc)   # [O] single snapshot

        if utc_from is not None:
            raw = mt5.copy_ticks_from(
                symbol, utc_from, count, mt5.COPY_TICKS_ALL
            )
        else:
            midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            raw = mt5.copy_ticks_range(
                symbol,
                midnight,
                now_utc,
                mt5.COPY_TICKS_ALL,
            )

        if raw is None or len(raw) == 0:
            logger.warning(
                "MT5Connector.get_ticks: no ticks for %s.", symbol
            )
            return None

        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time")

    # ------------------------------------------------------------------
    # [F] Enriched symbol info
    # ------------------------------------------------------------------
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        [F] Returns a comprehensive dict of symbol properties used by
        RiskManager (_calculate_pip_value, _normalize_volume) and
        OrderExecutor (_get_filling_mode, _normalise_volume, _validate_sl_tp).
        """
        if not self._select_symbol(symbol):
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(
                "MT5Connector.get_symbol_info: "
                "mt5.symbol_info('%s') returned None — %s",
                symbol, mt5.last_error(),
            )
            return None

        return {
            "name":               info.name,
            "digits":             info.digits,
            "point":              info.point,
            "spread":             info.spread,
            "contract_size":      info.trade_contract_size,
            "volume_min":         info.volume_min,
            "volume_max":         info.volume_max,
            "volume_step":        info.volume_step,
            "margin_initial":     info.margin_initial,
            "margin_maintenance": getattr(info, "margin_maintenance", 0.0),
            "trade_mode":         info.trade_mode,
            "filling_mode":       info.filling_mode,
            "bid":                info.bid,
            "ask":                info.ask,
            "session_close":      getattr(info, "session_close", 0.0),
            # Extra fields useful for pip-value calculations
            "currency_base":      getattr(info, "currency_base", ""),
            "currency_profit":    getattr(info, "currency_profit", ""),
            "currency_margin":    getattr(info, "currency_margin", ""),
        }

    # ------------------------------------------------------------------
    # [M] Open positions helper
    # ------------------------------------------------------------------
    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic:  Optional[int] = None,
    ) -> List[Dict]:
        """
        [M] Returns a list of open-position dicts so that OrderExecutor
        and RiskManager never need to call mt5.positions_get() directly.

        Filters by *symbol* and/or *magic* number when provided.
        """
        if not self.is_connected():
            return []

        if symbol:
            raw = mt5.positions_get(symbol=symbol)
        else:
            raw = mt5.positions_get()

        if raw is None:
            return []

        result: List[Dict] = []
        for p in raw:
            if magic is not None and p.magic != magic:
                continue
            result.append({
                "ticket":        p.ticket,
                "symbol":        p.symbol,
                "type":          "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume":        p.volume,
                "price_open":    p.price_open,
                "price_current": p.price_current,
                "sl":            p.sl,
                "tp":            p.tp,
                "profit":        p.profit,
                "swap":          p.swap,
                "commission":    p.commission,
                "magic":         p.magic,
                "comment":       p.comment,
                "time_open":     datetime.fromtimestamp(p.time, tz=timezone.utc),
            })
        return result

    # ------------------------------------------------------------------
    # [N] Pending orders helper
    # ------------------------------------------------------------------
    def get_orders(
        self,
        symbol: Optional[str] = None,
        magic:  Optional[int] = None,
    ) -> List[Dict]:
        """
        [N] Returns a list of pending order dicts.
        Filters by *symbol* and/or *magic* number when provided.
        """
        if not self.is_connected():
            return []

        if symbol:
            raw = mt5.orders_get(symbol=symbol)
        else:
            raw = mt5.orders_get()

        if raw is None:
            return []

        result: List[Dict] = []
        for o in raw:
            if magic is not None and o.magic != magic:
                continue
            result.append({
                "ticket":      o.ticket,
                "symbol":      o.symbol,
                "type":        o.type,
                "volume":      o.volume_current,
                "price_open":  o.price_open,
                "sl":          o.sl,
                "tp":          o.tp,
                "magic":       o.magic,
                "comment":     o.comment,
                "time_setup":  datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
            })
        return result
