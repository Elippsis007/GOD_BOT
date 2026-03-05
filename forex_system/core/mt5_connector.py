# core/mt5_connector.py
import MetaTrader5 as mt5
import pandas as pd
import pytz
import time
from datetime import datetime
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("MT5Connector")

# ── Timeframe normalisation map ───────────────────────────────────────────────
# config/settings.py stores timeframes as raw integers (seconds or MT5 internal
# constants). MT5 API only accepts its own TIMEFRAME_* constants. This map
# converts every possible input value to the correct MT5 constant.
MT5_TIMEFRAME_MAP = {
    # seconds-based values (used in some config styles)
    1:      mt5.TIMEFRAME_M1,
    5:      mt5.TIMEFRAME_M5,
    15:     mt5.TIMEFRAME_M15,
    30:     mt5.TIMEFRAME_M30,
    60:     mt5.TIMEFRAME_H1,
    240:    mt5.TIMEFRAME_H4,
    900:    mt5.TIMEFRAME_M15,
    3600:   mt5.TIMEFRAME_H1,
    14400:  mt5.TIMEFRAME_H4,
    86400:  mt5.TIMEFRAME_D1,
    # MT5 internal constants (pass-through — already correct)
    16385:  mt5.TIMEFRAME_H1,
    16386:  mt5.TIMEFRAME_H2,
    16387:  mt5.TIMEFRAME_H3,
    16388:  mt5.TIMEFRAME_H4,
    16390:  mt5.TIMEFRAME_H6,
    16392:  mt5.TIMEFRAME_H8,
    16396:  mt5.TIMEFRAME_H12,
    16408:  mt5.TIMEFRAME_D1,
    32769:  mt5.TIMEFRAME_W1,
    49153:  mt5.TIMEFRAME_MN1,
}


class MT5Connector:
    """
    Singleton connector — manages MT5 session lifecycle,
    account info, and raw market data retrieval.

    All DataFrames returned use lowercase column names
    (open, high, low, close, volume) to be consistent with
    DataHandler, IndicatorEngine, and MLSignalModel.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance

    # ── Connection Management ─────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Attempt to connect to MT5 in two stages:

        Stage 1 — attach to an already-running terminal, passing full
                   credentials so the session is properly authenticated.
                   A bare mt5.initialize() with no credentials attaches
                   anonymously and account_info() returns None.

        Stage 2 — launch the terminal from MT5_PATH if Stage 1 fails.
                   Uses a longer settle delay because the process needs
                   time to fully start before accepting API calls.
        """
        # Stage 1 — attach to running terminal with credentials
        if mt5.initialize(
            login=CONFIG.MT5_LOGIN,
            password=CONFIG.MT5_PASSWORD,
            server=CONFIG.MT5_SERVER,
        ):
            time.sleep(1)
            if mt5.account_info() is not None:
                self._connected = True
                logger.info("✅ Attached to running MT5 terminal")
                return True
            else:
                logger.warning(
                    "⚠️ MT5 initialized but account_info() returned None — "
                    "trying full launch with path"
                )
                mt5.shutdown()

        # Stage 2 — launch terminal from full path
        if mt5.initialize(
            path=CONFIG.MT5_PATH,
            login=CONFIG.MT5_LOGIN,
            password=CONFIG.MT5_PASSWORD,
            server=CONFIG.MT5_SERVER,
        ):
            time.sleep(2)
            if mt5.account_info() is not None:
                self._connected = True
                logger.info("✅ MT5 terminal launched successfully")
                return True
            else:
                logger.error(
                    "❌ MT5 launched but account_info() still None — "
                    "check login credentials and server name"
                )
                mt5.shutdown()

        logger.error(f"❌ MT5 connection failed: {mt5.last_error()}")
        return False

    def disconnect(self) -> None:
        mt5.shutdown()
        self._connected = False
        logger.info("🔌 MT5 disconnected")

    def is_connected(self) -> bool:
        return self._connected and mt5.account_info() is not None

    # ── Account Info ──────────────────────────────────────────────────────────
    def get_account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            return {}
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

    # ── OHLCV Data ────────────────────────────────────────────────────────────
    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: int,
        bars:      int = None,
        utc_from:  Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        FIX: normalise timeframe via MT5_TIMEFRAME_MAP before calling MT5.
        Previously passed raw integers (e.g. 16385, 900) directly which MT5
        silently rejected, returning None and logging 'No OHLCV data'.
        """
        try:
            bars = bars or CONFIG.BARS_HISTORY

            # ── Normalise timeframe to MT5 constant ───────────────────────────
            tf = MT5_TIMEFRAME_MAP.get(timeframe)
            if tf is None:
                logger.warning(
                    f"Unknown timeframe value {timeframe} — "
                    f"defaulting to TIMEFRAME_H1"
                )
                tf = mt5.TIMEFRAME_H1

            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available in Market Watch")
                return None

            if utc_from is not None:
                rates = mt5.copy_rates_from(symbol, tf, utc_from, bars)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None or len(rates) == 0:
                logger.warning(
                    f"No OHLCV data returned for {symbol} "
                    f"(tf={timeframe}→{tf}, bars={bars})"
                )
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)

            # Rename volume column (broker-dependent)
            if "tick_volume" in df.columns:
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
            elif "real_volume" in df.columns:
                df.rename(columns={"real_volume": "volume"}, inplace=True)

            available = [c for c in ["open", "high", "low", "close", "volume"]
                         if c in df.columns]
            df = df[available]

            logger.debug(
                f"📊 {symbol} | {len(df)} bars loaded "
                f"(tf={timeframe}→{tf})"
            )
            return df

        except Exception as e:
            logger.error(f"get_ohlcv({symbol}) error: {e}")
            return None

    # ── Tick Data ─────────────────────────────────────────────────────────────
    def get_ticks(
        self,
        symbol:   str,
        count:    int = 1000,
        utc_from: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available in Market Watch")
                return None

            utc_from = utc_from or datetime.now(pytz.utc)
            ticks = mt5.copy_ticks_from(
                symbol, utc_from, count, mt5.COPY_TICKS_ALL
            )

            if ticks is None or len(ticks) == 0:
                logger.warning(f"No tick data returned for {symbol}")
                return None

            df = pd.DataFrame(ticks)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            return df

        except Exception as e:
            logger.error(f"get_ticks({symbol}) error: {e}")
            return None

    # ── Symbol Info ───────────────────────────────────────────────────────────
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"symbol_info({symbol}) returned None")
            return None
        return {
            "symbol":              info.name,
            "digits":              info.digits,
            "point":               info.point,
            "spread":              info.spread,
            "trade_contract_size": info.trade_contract_size,
            "volume_min":          info.volume_min,
            "volume_max":          info.volume_max,
            "volume_step":         info.volume_step,
        }
