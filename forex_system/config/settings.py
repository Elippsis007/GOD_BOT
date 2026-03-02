# config/settings.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class TradingConfig:

    # ──────────────────────────────────────────────────────
    # ACCOUNT SETTINGS
    # ──────────────────────────────────────────────────────
    MT5_LOGIN:    int = 5046802311
    MT5_PASSWORD: str = "-8LblzDe"
    MT5_SERVER:   str = "MetaQuotes-Demo"
    MT5_PATH:     str = r"C:\Program Files\StoneX Europe MT5 Terminal\terminal64.exe"

    # ──────────────────────────────────────────────────────
    # SYMBOLS
    # ──────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD",   # Main focus — tightest spread
        "GBPUSD",   # High volatility
        "USDJPY",   # Asian session strength
        "AUDUSD",   # Commodity currency
        "USDCAD",   # Oil-correlated
        "XAUUSD"    # Gold — strong trends
    ])

    # ──────────────────────────────────────────────────────
    # TIMEFRAMES
    # ──────────────────────────────────────────────────────
    PRIMARY_TF:   int = 16385   # H1  — signal generation
    CONFIRM_TF:   int = 16388   # H4  — trend confirmation
    BARS_HISTORY: int = 1000    # Bars to load for ML training

    # ──────────────────────────────────────────────────────
    # RISK SETTINGS
    # ──────────────────────────────────────────────────────
    RISK_PER_TRADE:  float = 0.01  # 1% per trade (~€1000 on 100k)
    MAX_OPEN_TRADES: int   = 3     # Max 3 simultaneous trades
    MAX_DAILY_LOSS:  float = 0.03  # Stop at 3% daily loss (~€3000)
    RR_RATIO:        float = 2.0   # Min 1:2 risk to reward

    # ──────────────────────────────────────────────────────
    # INDICATOR SETTINGS
    # ──────────────────────────────────────────────────────
    EMA_FAST:  int = 9
    EMA_SLOW:  int = 21
    EMA_TREND: int = 50

    RSI_PERIOD:     int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD:   int = 30

    BB_PERIOD: int   = 20
    BB_STD:    float = 2.0

    ATR_PERIOD: int = 14

    MACD_FAST:   int = 12
    MACD_SLOW:   int = 26
    MACD_SIGNAL: int = 9

    # ──────────────────────────────────────────────────────
    # ML MODEL SETTINGS
    # ── PHASE 1 UPGRADE: confidence raised 0.60 → 0.70 ───
    # ──────────────────────────────────────────────────────
    ML_LOOKBACK:       int   = 60
    ML_MIN_CONFIDENCE: float = 0.70   # ← UPGRADED from 0.60
    ML_FEATURES: List[str] = field(default_factory=lambda: [
        "ema_fast",
        "ema_slow",
        "rsi",
        "bb_upper",
        "bb_lower",
        "atr",
        "macd",
        "macd_signal",
        "volume"
    ])

    # ──────────────────────────────────────────────────────
    # EXECUTION SETTINGS
    # ──────────────────────────────────────────────────────
    SLIPPAGE:     int = 20
    MAGIC_NUMBER: int = 202401
    COMMENT:      str = "GODBOT_v1"

    # ──────────────────────────────────────────────────────
    # SESSION FILTER
    # ── PHASE 1 UPGRADE: overlap only (highest volume) ───
    # London 08-17 CET, NY 14-23 CET, Overlap 14-17 CET
    # ──────────────────────────────────────────────────────
    TRADE_SESSIONS: List[str] = field(default_factory=lambda: [
        "overlap"    # ← UPGRADED: 14:00-17:00 CET only
                     # Highest volume = cleanest moves
                     # Change back to ["london","newyork","overlap"]
                     # after 30 days if you want more signals
    ])

    # ──────────────────────────────────────────────────────
    # SIGNAL STRENGTH FILTER
    # ── Already at 6 — keep here ─────────────────────────
    # ──────────────────────────────────────────────────────
    MIN_SIGNAL_SCORE: int = 6   # Out of 8 filters must pass
                                 # 6 = very selective (best quality)

    # ──────────────────────────────────────────────────────
    # TELEGRAM ALERT CONTROLS
    # ──────────────────────────────────────────────────────
    TELEGRAM_SEND_SIGNALS: bool = True    # BUY/SELL alerts
    TELEGRAM_SEND_DANGER:  bool = True    # Danger exit — always ON
    TELEGRAM_SEND_NEWS:    bool = True    # News warnings
    TELEGRAM_SEND_OPENED:  bool = True    # Trade opened
    TELEGRAM_SEND_CLOSED:  bool = True    # Trade closed
    TELEGRAM_SEND_SUMMARY: bool = True    # Daily report
    TELEGRAM_SEND_HOLD:    bool = False   # Hold alerts — OFF (noisy)
    TELEGRAM_SEND_EXIT:    bool = False   # Potential exit — OFF (noisy)

    # ──────────────────────────────────────────────────────
    # QUIET HOURS — Spain/CET timezone
    # ──────────────────────────────────────────────────────
    TELEGRAM_QUIET_ON:    bool = True
    TELEGRAM_QUIET_START: int  = 23   # Silence at 11pm CET
    TELEGRAM_QUIET_END:   int  = 8    # Resume at 8am CET

    # ──────────────────────────────────────────────────────
    # ALERT COOLDOWN TIMERS (seconds)
    # ──────────────────────────────────────────────────────
    ALERT_COOLDOWN_SIGNAL: int = 300    # 5 min between same signal
    ALERT_COOLDOWN_HOLD:   int = 900    # 15 min between hold alerts
    ALERT_COOLDOWN_EXIT:   int = 600    # 10 min between exit alerts
    ALERT_COOLDOWN_DANGER: int = 180    # 3 min between danger alerts
    ALERT_COOLDOWN_NEWS:   int = 3600   # 60 min between news alerts

    # ──────────────────────────────────────────────────────
    # MINIMUM CONFIDENCE TO SEND ALERT
    # ──────────────────────────────────────────────────────
    ALERT_MIN_CONFIDENCE: float = 0.70  # Only alert on 70%+ confidence


# ── Single config instance used across all modules ────────
CONFIG = TradingConfig()


# ══════════════════════════════════════════════════════════
# CREDENTIALS & KEYS
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# TRADER PROFILE
# ──────────────────────────────────────────────────────────
TRADER_NAME:     str = "Michael"
TRADER_TIMEZONE: str = "Europe/Madrid"

# ──────────────────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────────────────
TELEGRAM_TOKEN:   str = "8693437372:AAHQm4RwedYLkgKkYTlD5XZACHqMnvHVUBE"
TELEGRAM_CHAT_ID: str = "7688107635"

# ──────────────────────────────────────────────────────────
# GOOGLE GEMINI API
# ── Phase 2 upgrade — better sentiment analysis ──────────
# Get free key at: aistudio.google.com
# Free tier: 1,500 requests/day — more than enough
# ──────────────────────────────────────────────────────────
GEMINI_API_KEY: str = "AIzaSyBKtwbd1e1M7tLYsMIrmibKTrTGAn01XZM"
GEMINI_ENABLED: bool = True   # ← Change to True once key is added

# ══════════════════════════════════════════════════════════
# TRADING STYLE SETTINGS
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# SCALPER — M1/M5, 5-10 pip targets
# ──────────────────────────────────────────────────────────
SCALPER_TF_PRIMARY:   int   = 1      # M1
SCALPER_TF_CONFIRM:   int   = 5      # M5
SCALPER_TP_PIPS:      float = 7.0    # Target 7 pips
SCALPER_SL_PIPS:      float = 4.0    # Risk 4 pips
SCALPER_MAX_SPREAD:   float = 1.5    # Skip if spread > 1.5 pips
SCALPER_SCAN_SECS:    int   = 10     # Scan every 10 seconds
SCALPER_SIGNAL_SCORE: int   = 3      # Lower threshold for scalping

# ──────────────────────────────────────────────────────────
# DAY TRADER — M15/H1, 20-40 pip targets
# ──────────────────────────────────────────────────────────
DAYTRADER_TF_PRIMARY:   int   = 900    # M15
DAYTRADER_TF_CONFIRM:   int   = 16385  # H1
DAYTRADER_TP_PIPS:      float = 30.0   # Target 30 pips
DAYTRADER_SL_PIPS:      float = 15.0   # Risk 15 pips
DAYTRADER_MAX_SPREAD:   float = 2.0    # Skip if spread > 2 pips
DAYTRADER_SCAN_SECS:    int   = 60     # Scan every 60 seconds
DAYTRADER_SIGNAL_SCORE: int   = 4      # Standard threshold

# ──────────────────────────────────────────────────────────
# SESSION TIMES (CET — Spain/Madrid)
# ──────────────────────────────────────────────────────────
LONDON_OPEN_CET:  int = 8     # 08:00 CET
LONDON_CLOSE_CET: int = 17    # 17:00 CET
NY_OPEN_CET:      int = 14    # 14:00 CET
NY_CLOSE_CET:     int = 23    # 23:00 CET
OVERLAP_START:    int = 14    # 14:00 CET ← PRIME TIME
OVERLAP_END:      int = 17    # 17:00 CET ← PRIME TIME

# ──────────────────────────────────────────────────────────
# PROFILE SAVE FILE
# ──────────────────────────────────────────────────────────
PROFILE_FILE: str = "data/profile.json"