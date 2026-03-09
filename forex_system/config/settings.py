# config/settings.py
import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class TradingConfig:

    # â”€â”€ MT5 Account â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    MT5_LOGIN:    int = field(default_factory=lambda: int(os.getenv("MT5_LOGIN", "62111571")))
    MT5_PASSWORD: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", "ue.2drzZyq"))
    MT5_SERVER:   str = field(default_factory=lambda: os.getenv("MT5_SERVER",   "PepperstoneUK-Demo"))
    MT5_PATH:     str = field(default_factory=lambda: os.getenv(
        "MT5_PATH", r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe"
    ))

    # â”€â”€ Trader Identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TRADER_NAME:     str = field(default_factory=lambda: os.getenv("TRADER_NAME",     "Michael"))
    TRADER_TIMEZONE: str = field(default_factory=lambda: os.getenv("TRADER_TIMEZONE", "Europe/Madrid"))

    # â”€â”€ Telegram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TELEGRAM_TOKEN:   str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN",   "8693437372:AAHQm4RwedYLkgKkYTlD5XZACHqMnvHVUBE"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", "7688107635"))

    # â”€â”€ Gemini â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    GEMINI_API_KEY: str  = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "AIzaSyBKtwbd1e1M7tLYsMIrmibKTrTGAn01XZM"))
    GEMINI_ENABLED: bool = True

    # â”€â”€ Watchlist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    WATCHLIST: List[str] = field(default_factory=lambda: ["EURUSD"])

    # â”€â”€ Symbols â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD",
    ])

    # â”€â”€ Timeframes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    PRIMARY_TF:   int = 16385   # H1
    CONFIRM_TF:   int = 16388   # H4
    BARS_HISTORY: int = 1000

    # â”€â”€ Risk â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    RISK_PER_TRADE:     float = 0.01
    MAX_OPEN_TRADES:    int   = 3
    MAX_DAILY_LOSS:     float = 0.03
    RR_RATIO:           float = 2.0
    MAX_LOSS_PER_TRADE: float = 50.0
    TRAILING_STOP_PIPS: int   = 15

    # â”€â”€ Indicators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    EMA_FAST:       int   = 9
    EMA_SLOW:       int   = 21
    EMA_TREND:      int   = 50
    RSI_PERIOD:     int   = 14
    RSI_OVERBOUGHT: int   = 70
    RSI_OVERSOLD:   int   = 30
    BB_PERIOD:      int   = 20
    BB_STD:         float = 2.0
    ATR_PERIOD:     int   = 14
    MACD_FAST:      int   = 12
    MACD_SLOW:      int   = 26
    MACD_SIGNAL:    int   = 9

    # â”€â”€ ML Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ML_LOOKBACK:       int   = 60
    ML_MIN_CONFIDENCE: float = 0.35
    ML_FEATURES: List[str] = field(default_factory=lambda: [
        "ema_fast", "ema_slow", "rsi", "bb_upper", "bb_lower",
        "atr", "macd", "macd_signal", "volume",
    ])

    # â”€â”€ Execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SLIPPAGE:     int = 20
    MAGIC_NUMBER: int = 202401
    COMMENT:      str = "GODBOT_v1"

    # â”€â”€ Sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TRADE_SESSIONS: List[str] = field(default_factory=lambda: [
        "london", "newyork", "overlap",
    ])

    # â”€â”€ Signal Score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    MIN_SIGNAL_SCORE: int = 3

    # â”€â”€ Telegram Alert Flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TELEGRAM_SEND_SIGNALS: bool = True
    TELEGRAM_SEND_DANGER:  bool = True
    TELEGRAM_SEND_NEWS:    bool = True
    TELEGRAM_SEND_OPENED:  bool = True
    TELEGRAM_SEND_CLOSED:  bool = True
    TELEGRAM_SEND_SUMMARY: bool = True
    TELEGRAM_SEND_HOLD:    bool = False
    TELEGRAM_SEND_EXIT:    bool = False

    # â”€â”€ Quiet Hours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TELEGRAM_QUIET_ON:    bool = True
    TELEGRAM_QUIET_START: int  = 23
    TELEGRAM_QUIET_END:   int  = 8

    # â”€â”€ Alert Cooldowns (seconds) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ALERT_COOLDOWN_SIGNAL: int = 300
    ALERT_COOLDOWN_HOLD:   int = 900
    ALERT_COOLDOWN_EXIT:   int = 600
    ALERT_COOLDOWN_DANGER: int = 180
    ALERT_COOLDOWN_NEWS:   int = 3600

    # â”€â”€ Alert Confidence Threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ALERT_MIN_CONFIDENCE: float = 0.35

    # â”€â”€ Scalper Style (M5) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SCALPER_TF_PRIMARY:   int   = 5       # M5 â€” primary signal timeframe
    SCALPER_TF_CONFIRM:   int   = 15      # M15 â€” confirmation timeframe
    SCALPER_TP_PIPS:      float = 12.0
    SCALPER_SL_PIPS:      float = 6.0
    SCALPER_MAX_SPREAD:   float = 2.0
    SCALPER_SCAN_SECS:    int   = 30
    SCALPER_SIGNAL_SCORE: int   = 3

    # â”€â”€ Day Trader Style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    DAYTRADER_TF_PRIMARY:   int   = 900
    DAYTRADER_TF_CONFIRM:   int   = 16385
    DAYTRADER_TP_PIPS:      float = 30.0
    DAYTRADER_SL_PIPS:      float = 15.0
    DAYTRADER_MAX_SPREAD:   float = 2.0
    DAYTRADER_SCAN_SECS:    int   = 60
    DAYTRADER_SIGNAL_SCORE: int   = 4

    # â”€â”€ Session Times (CET hour) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LONDON_OPEN_CET: int = 8
    NY_CLOSE_CET:    int = 23
    OVERLAP_START:   int = 14
    OVERLAP_END:     int = 17

    # â”€â”€ Profile File â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    PROFILE_FILE: str = "data/profile.json"


# â”€â”€ Single global instance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONFIG = TradingConfig()

# â”€â”€ Module-level aliases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROFILE_FILE           = CONFIG.PROFILE_FILE
TRADER_NAME            = CONFIG.TRADER_NAME
TRADER_TIMEZONE        = CONFIG.TRADER_TIMEZONE
GEMINI_ENABLED         = CONFIG.GEMINI_ENABLED
GEMINI_API_KEY         = CONFIG.GEMINI_API_KEY
SCALPER_SCAN_SECS      = CONFIG.SCALPER_SCAN_SECS
DAYTRADER_SCAN_SECS    = CONFIG.DAYTRADER_SCAN_SECS
SCALPER_TF_PRIMARY     = CONFIG.SCALPER_TF_PRIMARY
SCALPER_TF_CONFIRM     = CONFIG.SCALPER_TF_CONFIRM
DAYTRADER_TF_PRIMARY   = CONFIG.DAYTRADER_TF_PRIMARY
DAYTRADER_TF_CONFIRM   = CONFIG.DAYTRADER_TF_CONFIRM
SCALPER_MAX_SPREAD     = CONFIG.SCALPER_MAX_SPREAD
DAYTRADER_MAX_SPREAD   = CONFIG.DAYTRADER_MAX_SPREAD
SCALPER_SIGNAL_SCORE   = CONFIG.SCALPER_SIGNAL_SCORE
DAYTRADER_SIGNAL_SCORE = CONFIG.DAYTRADER_SIGNAL_SCORE
SCALPER_SL_PIPS        = CONFIG.SCALPER_SL_PIPS
DAYTRADER_SL_PIPS      = CONFIG.DAYTRADER_SL_PIPS
SCALPER_TP_PIPS        = CONFIG.SCALPER_TP_PIPS
DAYTRADER_TP_PIPS      = CONFIG.DAYTRADER_TP_PIPS
LONDON_OPEN_CET        = CONFIG.LONDON_OPEN_CET
NY_CLOSE_CET           = CONFIG.NY_CLOSE_CET
OVERLAP_START          = CONFIG.OVERLAP_START
OVERLAP_END            = CONFIG.OVERLAP_END

