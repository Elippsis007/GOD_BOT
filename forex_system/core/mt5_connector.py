# core/mt5_connector.py
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("MT5Connector")

class MT5Connector:
    """
    Singleton connector — manages MT5 session lifecycle,
    account info, and raw market data retrieval.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance

    # ── Connection Management ─────────────────────────────
    def connect(self) -> bool:
        # Always try attaching to the already-running terminal first
        if mt5.initialize():
            self._connected = True
            logger.info("✅ Attached to running MT5 terminal")
            return True
        # Fallback: launch terminal with credentials
        if mt5.initialize(
            path=CONFIG.MT5_PATH,
            login=CONFIG.MT5_LOGIN,
            password=CONFIG.MT5_PASSWORD,
            server=CONFIG.MT5_SERVER
        ):
            self._connected = True
            logger.info("✅ MT5 terminal launched successfully")
            return True
        logger.error(f"❌ MT5 connection failed: {mt5.last_error()}")
        return False

    def disconnect(self):
        mt5.shutdown()
        self._connected = False
        logger.info("🔌 MT5 disconnected")

    def is_connected(self) -> bool:
        return self._connected and mt5.terminal_info() is not None

    # ── Account Info ──────────────────────────────────────
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
            "currency":    info.currency
        }

    # ── OHLCV Data ────────────────────────────────────────
    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: int,
        bars:      int = None,
        utc_from:  Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        try:
            bars = bars or CONFIG.BARS_HISTORY

            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Symbol {symbol} not available")

            utc_from = utc_from or datetime.now(pytz.utc)
            rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)

            if rates is None or len(rates) == 0:
                raise ValueError(f"No data returned for {symbol}")

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={
                "open":        "Open",
                "high":        "High",
                "low":         "Low",
                "close":       "Close",
                "tick_volume": "Volume"
            }, inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            logger.debug(f"📊 {symbol} | {len(df)} bars loaded")
            return df

        except Exception as e:
            logger.error(f"get_ohlcv({symbol}) error: {e}")
            return None

    # ── Tick Data ─────────────────────────────────────────
    def get_ticks(
        self,
        symbol:   str,
        count:    int = 1000,
        utc_from: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        try:
            utc_from = utc_from or datetime.now(pytz.utc)
            ticks = mt5.copy_ticks_from(
                symbol, utc_from, count, mt5.COPY_TICKS_ALL
            )

            if ticks is None:
                raise ValueError(f"No ticks returned for {symbol}")

            df = pd.DataFrame(ticks)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            return df

        except Exception as e:
            logger.error(f"get_ticks({symbol}) error: {e}")
            return None

    # ── Symbol Info ───────────────────────────────────────
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        info = mt5.symbol_info(symbol)
        if info is None:
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