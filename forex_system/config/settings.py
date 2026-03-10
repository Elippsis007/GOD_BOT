# config/settings.py
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class TradingConfig:

    # ── MT5 Account ───────────────────────────────────────────────────────────
    MT5_LOGIN:    int = field(default_factory=lambda: int(os.getenv("MT5_LOGIN", "62111571")))
    MT5_PASSWORD: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", "ue.2drzZyq"))
    MT5_SERVER:   str = field(default_factory=lambda: os.getenv("MT5_SERVER",   "PepperstoneUK-Demo"))
    MT5_PATH:     str = field(default_factory=lambda: os.getenv(
        "MT5_PATH", r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe"
    ))

    # ── Trader Identity ───────────────────────────────────────────────────────
    TRADER_NAME:     str = field(default_factory=lambda: os.getenv("TRADER_NAME",     "Michael"))
    TRADER_TIMEZONE: str = field(default_factory=lambda: os.getenv("TRADER_TIMEZONE", "Europe/Madrid"))

    # ── Telegram ──────────────────────────────────────────────────────────────
    TELEGRAM_TOKEN:   str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN",   "8693437372:AAHQm4RwedYLkgKkYTlD5XZACHqMnvHVUBE"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", "7688107635"))

    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str  = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "AIzaSyBKtwbd1e1M7tLYsMIrmibKTrTGAn01XZM"))
    GEMINI_ENABLED: bool = True

    # ── Watchlist ─────────────────────────────────────────────────────────────
    WATCHLIST: List[str] = field(default_factory=lambda: ["EURUSD"])

    # ── Symbols ───────────────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: ["EURUSD"])

    # ── Timeframes (day-trader / generic defaults) ────────────────────────────
    PRIMARY_TF:   int = 16385   # H1  — MT5 internal constant
    CONFIRM_TF:   int = 16388   # H4  — MT5 internal constant
    BARS_HISTORY: int = 1000

    # ── Risk ──────────────────────────────────────────────────────────────────
    # RISK_PER_TRADE × MAX_OPEN_TRADES = 0.005 × 3 = 1.5% max simultaneous exposure
    RISK_PER_TRADE:     float = 0.005
    MAX_OPEN_TRADES:    int   = 3
    MAX_DAILY_LOSS:     float = 0.03    # 3% equity drawdown halts trading for the day
    RR_RATIO:           float = 2.0     # TP must be SL × 2.0 — enforced by RiskManager
    MAX_LOSS_PER_TRADE: float = 5.0     # Hard absolute cap in account currency
    TRAILING_STOP_PIPS: int   = 5       # Fallback — profile value takes priority

    # ── Indicators (shared / day-trader defaults) ─────────────────────────────
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
    ADX_PERIOD:     int   = 14

    # ── ML Model ──────────────────────────────────────────────────────────────
    ML_LOOKBACK:       int   = 60
    ML_MIN_CONFIDENCE: float = 0.30     # Above 3-class random baseline of 0.333

    ML_REQUIRED_FEATURES: List[str] = field(default_factory=lambda: [
        "ema_fast", "ema_slow", "ema_trend",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_mid", "atr",
        "volume", "adx",
    ])

    # ── Execution ─────────────────────────────────────────────────────────────
    # 3 points = 0.3 pips on a 5-decimal ECN broker (Pepperstone)
    SLIPPAGE:     int = 3
    MAGIC_NUMBER: int = 202401
    COMMENT:      str = "GODBOT_v3"

    # ── Sessions ──────────────────────────────────────────────────────────────
    TRADE_SESSIONS: List[str] = field(default_factory=lambda: [
        "london", "newyork", "overlap",
    ])

    # ── Signal Score ──────────────────────────────────────────────────────────
    MIN_SIGNAL_SCORE: int = 3

    # ── Telegram Alert Flags ──────────────────────────────────────────────────
    TELEGRAM_SEND_SIGNALS: bool = True
    TELEGRAM_SEND_DANGER:  bool = True
    TELEGRAM_SEND_NEWS:    bool = True
    TELEGRAM_SEND_OPENED:  bool = True
    TELEGRAM_SEND_CLOSED:  bool = True
    TELEGRAM_SEND_SUMMARY: bool = True
    TELEGRAM_SEND_HOLD:    bool = True
    TELEGRAM_SEND_EXIT:    bool = True

    # ── Quiet Hours ───────────────────────────────────────────────────────────
    TELEGRAM_QUIET_ON:    bool  = True
    TELEGRAM_QUIET_START: int   = 23
    TELEGRAM_QUIET_END:   int   = 7
    ENABLE_QUIET_HOURS:   bool  = False     # FIX: was True — blocked all night scans
    QUIET_HOURS_UTC:      tuple = (22, 7)   # Only active when ENABLE_QUIET_HOURS=True

    # ── Alert Cooldowns (seconds) ─────────────────────────────────────────────
    ALERT_COOLDOWN_SIGNAL: int = 300
    ALERT_COOLDOWN_HOLD:   int = 120
    ALERT_COOLDOWN_EXIT:   int = 120
    ALERT_COOLDOWN_DANGER: int = 30
    ALERT_COOLDOWN_NEWS:   int = 3600

    # ── Alert Confidence Threshold ────────────────────────────────────────────
    ALERT_MIN_CONFIDENCE: float = 0.55

    # ── EOD Controls ──────────────────────────────────────────────────────────
    CLOSE_TRADES_EOD:   bool = True
    EOD_CLOSE_HOUR_UTC: int  = 21      # 21:00 UTC Friday = end of FX week

    # ── Scalper flat aliases ───────────────────────────────────────────────────
    # Written by get_scalper_profile() at runtime. These M5 defaults ensure
    # CONFIG.SCALPER_MAX_SPREAD etc. are always valid attributes even if
    # get_scalper_profile() has not yet been called.
    SCALPER_MAX_SPREAD:    float = 1.0
    SCALPER_TP_PIPS:       float = 12.0   # FIX: was 10.0 — must be SL(6) × RR(2.0) = 12.0
    SCALPER_SL_PIPS:       float = 6.0
    SCALPER_TRAILING_STOP: int   = 5
    SCALPER_SCAN_SECS:     int   = 20
    SCALPER_SIGNAL_SCORE:  int   = 3
    SCALPER_ADX_THRESHOLD: int   = 18   # FIX: was 25 — flat alias kept in sync with M5 profile

    # ══════════════════════════════════════════════════════════════════════════
    #  SCALPER PROFILES  —  M1 and M5
    # ══════════════════════════════════════════════════════════════════════════

    SCALPER_TF_SELECTED: int = 5   # Default M5 — overwritten by startup menu

    # ── M1 Profile ────────────────────────────────────────────────────────────
    SCALPER_M1_TF_PRIMARY:      int   = 1
    SCALPER_M1_TF_CONFIRM:      int   = 5
    SCALPER_M1_RSI_PERIOD:      int   = 7
    SCALPER_M1_RSI_OVERBOUGHT:  int   = 80
    SCALPER_M1_RSI_OVERSOLD:    int   = 20
    SCALPER_M1_RSI_BULL_LOW:    int   = 30
    SCALPER_M1_RSI_BULL_HIGH:   int   = 55
    SCALPER_M1_RSI_BEAR_LOW:    int   = 55
    SCALPER_M1_RSI_BEAR_HIGH:   int   = 80
    SCALPER_M1_EMA_FAST:        int   = 5
    SCALPER_M1_EMA_SLOW:        int   = 13
    SCALPER_M1_EMA_TREND:       int   = 34
    SCALPER_M1_MACD_FAST:       int   = 5
    SCALPER_M1_MACD_SLOW:       int   = 13
    SCALPER_M1_MACD_SIGNAL:     int   = 4
    SCALPER_M1_BB_PERIOD:       int   = 10
    SCALPER_M1_BB_STD:          float = 2.0
    SCALPER_M1_ATR_PERIOD:      int   = 7
    SCALPER_M1_ADX_PERIOD:      int   = 7
    SCALPER_M1_ADX_THRESHOLD:   int   = 18
    SCALPER_M1_STOCH_K:         int   = 3
    SCALPER_M1_STOCH_D:         int   = 3
    SCALPER_M1_STOCH_BULL_ZONE: int   = 25
    SCALPER_M1_STOCH_BEAR_ZONE: int   = 75
    SCALPER_M1_CMF_THRESHOLD:   float = 0.03
    SCALPER_M1_TP_PIPS:         float = 6.0
    SCALPER_M1_SL_PIPS:         float = 3.0
    SCALPER_M1_MAX_SPREAD:      float = 0.8
    SCALPER_M1_SCAN_SECS:       int   = 10
    SCALPER_M1_SIGNAL_SCORE:    int   = 4
    SCALPER_M1_TRAILING_STOP:   int   = 3

    # ── M5 Profile ────────────────────────────────────────────────────────────
    #
    #  SCALPER_M5_TP_PIPS: FIX — raised from 10.0 to 12.0.
    #    RR_RATIO = 2.0 requires TP = SL × RR = 6 × 2.0 = 12 pips.
    #    With TP = 10 and SL = 6, RR = 1.667 which is BELOW the 2.0
    #    minimum enforced by RiskManager._pre_trade_checks(). Every M5
    #    scalp signal was being silently rejected before order submission.
    #    12 pips is achievable on EURUSD M5 with ADX >= 18 confirming
    #    a genuine directional move.
    #
    #  SCALPER_M5_CMF_THRESHOLD: FIX — lowered from 0.05 to 0.03.
    #    MT5 provides tick volume (tick count), not real traded volume.
    #    Tick-volume CMF is noisier than real-volume CMF. The standard
    #    0.05 threshold was calibrated for real volume data and was
    #    rejecting valid flow signals on tick-volume feeds. 0.03 captures
    #    genuine institutional flow pressure without requiring a signal
    #    strength that tick volume cannot reliably produce.
    #
    #  SCALPER_M5_ADX_THRESHOLD: FIX — lowered from 25 to 18.
    #    ADX 25 is only reached during London/NY sessions. During Asian
    #    hours EURUSD ADX is typically 12-18. Lowering to 18 allows the
    #    signal engine to fire during overnight hours as the user requested
    #    (trade EURUSD 24/7 while market is open, not just peak sessions).
    #
    SCALPER_M5_TF_PRIMARY:      int   = 5
    SCALPER_M5_TF_CONFIRM:      int   = 15
    SCALPER_M5_RSI_PERIOD:      int   = 9
    SCALPER_M5_RSI_OVERBOUGHT:  int   = 75
    SCALPER_M5_RSI_OVERSOLD:    int   = 25
    SCALPER_M5_RSI_BULL_LOW:    int   = 35
    SCALPER_M5_RSI_BULL_HIGH:   int   = 60
    SCALPER_M5_RSI_BEAR_LOW:    int   = 60
    SCALPER_M5_RSI_BEAR_HIGH:   int   = 75
    SCALPER_M5_EMA_FAST:        int   = 8
    SCALPER_M5_EMA_SLOW:        int   = 21
    SCALPER_M5_EMA_TREND:       int   = 50
    SCALPER_M5_MACD_FAST:       int   = 8
    SCALPER_M5_MACD_SLOW:       int   = 21
    SCALPER_M5_MACD_SIGNAL:     int   = 5
    SCALPER_M5_BB_PERIOD:       int   = 15
    SCALPER_M5_BB_STD:          float = 2.0
    SCALPER_M5_ATR_PERIOD:      int   = 10
    SCALPER_M5_ADX_PERIOD:      int   = 10
    SCALPER_M5_ADX_THRESHOLD:   int   = 18   # FIX: was 25 — enables signals during Asian/overnight hours
    SCALPER_M5_STOCH_K:         int   = 5
    SCALPER_M5_STOCH_D:         int   = 3
    SCALPER_M5_STOCH_BULL_ZONE: int   = 25
    SCALPER_M5_STOCH_BEAR_ZONE: int   = 75
    SCALPER_M5_CMF_THRESHOLD:   float = 0.03   # FIX: was 0.05 — calibrated for tick volume
    SCALPER_M5_TP_PIPS:         float = 12.0   # FIX: was 10.0 — satisfies RR_RATIO = 2.0
    SCALPER_M5_SL_PIPS:         float = 6.0
    SCALPER_M5_MAX_SPREAD:      float = 1.0
    SCALPER_M5_SCAN_SECS:       int   = 20
    SCALPER_M5_SIGNAL_SCORE:    int   = 3
    SCALPER_M5_TRAILING_STOP:   int   = 5

    # ── Day Trader Profile ────────────────────────────────────────────────────
    DAYTRADER_TF_PRIMARY:      int   = 15
    DAYTRADER_TF_CONFIRM:      int   = 16385
    DAYTRADER_TP_PIPS:         float = 30.0
    DAYTRADER_SL_PIPS:         float = 15.0
    DAYTRADER_MAX_SPREAD:      float = 1.5
    DAYTRADER_SCAN_SECS:       int   = 60
    DAYTRADER_SIGNAL_SCORE:    int   = 4
    DAYTRADER_ADX_THRESHOLD:   int   = 25
    DAYTRADER_RSI_OVERBOUGHT:  int   = 70
    DAYTRADER_RSI_OVERSOLD:    int   = 30
    DAYTRADER_RSI_BULL_LOW:    int   = 40
    DAYTRADER_RSI_BULL_HIGH:   int   = 60
    DAYTRADER_RSI_BEAR_LOW:    int   = 60
    DAYTRADER_RSI_BEAR_HIGH:   int   = 70
    DAYTRADER_CMF_THRESHOLD:   float = 0.05
    DAYTRADER_STOCH_BULL_ZONE: int   = 20
    DAYTRADER_STOCH_BEAR_ZONE: int   = 80

    # ── Session Times (CET hour) ──────────────────────────────────────────────
    LONDON_OPEN_CET:  int = 8
    LONDON_CLOSE_CET: int = 17     # London closes 17:00 CET
    NY_OPEN_CET:      int = 14
    NY_CLOSE_CET:     int = 22     # NY closes ~22:00 CET
    OVERLAP_START:    int = 14
    OVERLAP_END:      int = 17     # Overlap ends when London closes

    # ── Profile File ──────────────────────────────────────────────────────────
    PROFILE_FILE: str = "data/profile.json"

    # ══════════════════════════════════════════════════════════════════════════
    #  get_scalper_profile()
    # ══════════════════════════════════════════════════════════════════════════

    def get_scalper_profile(self) -> Dict[str, Any]:
        """
        Return the complete scalper settings dict for the active timeframe.
        SCALPER_TF_SELECTED = 1  ->  M1 profile
        SCALPER_TF_SELECTED = 5  ->  M5 profile  (default)

        FIX: alias write-back now runs for BOTH M1 and M5 before returning,
        so CONFIG.SCALPER_MAX_SPREAD / TP / SL etc. are always in sync with
        whichever timeframe is active. Previously the M1 branch returned
        early and the aliases were never updated when M1 was selected.
        """
        if self.SCALPER_TF_SELECTED == 1:
            profile = {
                # ── Timeframes ────────────────────────────────────────────
                "tf_primary":    self.SCALPER_M1_TF_PRIMARY,
                "tf_confirm":    self.SCALPER_M1_TF_CONFIRM,
                "tf_label":      "M1",
                # ── RSI ───────────────────────────────────────────────────
                "rsi_period":     self.SCALPER_M1_RSI_PERIOD,
                "rsi_overbought": self.SCALPER_M1_RSI_OVERBOUGHT,
                "rsi_oversold":   self.SCALPER_M1_RSI_OVERSOLD,
                "rsi_bull_low":   self.SCALPER_M1_RSI_BULL_LOW,
                "rsi_bull_high":  self.SCALPER_M1_RSI_BULL_HIGH,
                "rsi_bear_low":   self.SCALPER_M1_RSI_BEAR_LOW,
                "rsi_bear_high":  self.SCALPER_M1_RSI_BEAR_HIGH,
                # ── EMA ───────────────────────────────────────────────────
                "ema_fast":  self.SCALPER_M1_EMA_FAST,
                "ema_slow":  self.SCALPER_M1_EMA_SLOW,
                "ema_trend": self.SCALPER_M1_EMA_TREND,
                # ── MACD ──────────────────────────────────────────────────
                "macd_fast":   self.SCALPER_M1_MACD_FAST,
                "macd_slow":   self.SCALPER_M1_MACD_SLOW,
                "macd_signal": self.SCALPER_M1_MACD_SIGNAL,
                # ── Bollinger Bands ───────────────────────────────────────
                "bb_period": self.SCALPER_M1_BB_PERIOD,
                "bb_std":    self.SCALPER_M1_BB_STD,
                # ── ATR ───────────────────────────────────────────────────
                "atr_period": self.SCALPER_M1_ATR_PERIOD,
                # ── ADX ───────────────────────────────────────────────────
                "adx_period":    self.SCALPER_M1_ADX_PERIOD,
                "adx_threshold": self.SCALPER_M1_ADX_THRESHOLD,
                # ── Stochastic ────────────────────────────────────────────
                "stoch_k":         self.SCALPER_M1_STOCH_K,
                "stoch_d":         self.SCALPER_M1_STOCH_D,
                "stoch_bull_zone": self.SCALPER_M1_STOCH_BULL_ZONE,
                "stoch_bear_zone": self.SCALPER_M1_STOCH_BEAR_ZONE,
                # ── CMF ───────────────────────────────────────────────────
                "cmf_threshold": self.SCALPER_M1_CMF_THRESHOLD,
                # ── Trade sizing ──────────────────────────────────────────
                "tp_pips":       self.SCALPER_M1_TP_PIPS,
                "sl_pips":       self.SCALPER_M1_SL_PIPS,
                "trailing_stop": self.SCALPER_M1_TRAILING_STOP,
                # ── Execution ─────────────────────────────────────────────
                "max_spread":   self.SCALPER_M1_MAX_SPREAD,
                "scan_secs":    self.SCALPER_M1_SCAN_SECS,
                "signal_score": self.SCALPER_M1_SIGNAL_SCORE,
            }
        else:
            profile = {
                # ── Timeframes ────────────────────────────────────────────
                "tf_primary":    self.SCALPER_M5_TF_PRIMARY,
                "tf_confirm":    self.SCALPER_M5_TF_CONFIRM,
                "tf_label":      "M5",
                # ── RSI ───────────────────────────────────────────────────
                "rsi_period":     self.SCALPER_M5_RSI_PERIOD,
                "rsi_overbought": self.SCALPER_M5_RSI_OVERBOUGHT,
                "rsi_oversold":   self.SCALPER_M5_RSI_OVERSOLD,
                "rsi_bull_low":   self.SCALPER_M5_RSI_BULL_LOW,
                "rsi_bull_high":  self.SCALPER_M5_RSI_BULL_HIGH,
                "rsi_bear_low":   self.SCALPER_M5_RSI_BEAR_LOW,
                "rsi_bear_high":  self.SCALPER_M5_RSI_BEAR_HIGH,
                # ── EMA ───────────────────────────────────────────────────
                "ema_fast":  self.SCALPER_M5_EMA_FAST,
                "ema_slow":  self.SCALPER_M5_EMA_SLOW,
                "ema_trend": self.SCALPER_M5_EMA_TREND,
                # ── MACD ──────────────────────────────────────────────────
                "macd_fast":   self.SCALPER_M5_MACD_FAST,
                "macd_slow":   self.SCALPER_M5_MACD_SLOW,
                "macd_signal": self.SCALPER_M5_MACD_SIGNAL,
                # ── Bollinger Bands ───────────────────────────────────────
                "bb_period": self.SCALPER_M5_BB_PERIOD,
                "bb_std":    self.SCALPER_M5_BB_STD,
                # ── ATR ───────────────────────────────────────────────────
                "atr_period": self.SCALPER_M5_ATR_PERIOD,
                # ── ADX ───────────────────────────────────────────────────
                "adx_period":    self.SCALPER_M5_ADX_PERIOD,
                "adx_threshold": self.SCALPER_M5_ADX_THRESHOLD,
                # ── Stochastic ────────────────────────────────────────────
                "stoch_k":         self.SCALPER_M5_STOCH_K,
                "stoch_d":         self.SCALPER_M5_STOCH_D,
                "stoch_bull_zone": self.SCALPER_M5_STOCH_BULL_ZONE,
                "stoch_bear_zone": self.SCALPER_M5_STOCH_BEAR_ZONE,
                # ── CMF ───────────────────────────────────────────────────
                "cmf_threshold": self.SCALPER_M5_CMF_THRESHOLD,
                # ── Trade sizing ──────────────────────────────────────────
                "tp_pips":       self.SCALPER_M5_TP_PIPS,
                "sl_pips":       self.SCALPER_M5_SL_PIPS,
                "trailing_stop": self.SCALPER_M5_TRAILING_STOP,
                # ── Execution ─────────────────────────────────────────────
                "max_spread":   self.SCALPER_M5_MAX_SPREAD,
                "scan_secs":    self.SCALPER_M5_SCAN_SECS,
                "signal_score": self.SCALPER_M5_SIGNAL_SCORE,
            }

        # ── Write flat aliases back onto CONFIG ───────────────────────────────
        # FIX: moved OUTSIDE the if/else so aliases are always updated
        # regardless of which timeframe is selected. Previously the M1
        # branch returned before reaching this block, leaving all
        # CONFIG.SCALPER_* aliases pointing at M5 values even when M1
        # was the active profile.
        self.SCALPER_MAX_SPREAD    = profile["max_spread"]
        self.SCALPER_TP_PIPS       = profile["tp_pips"]
        self.SCALPER_SL_PIPS       = profile["sl_pips"]
        self.SCALPER_TRAILING_STOP = profile["trailing_stop"]
        self.SCALPER_SCAN_SECS     = profile["scan_secs"]
        self.SCALPER_SIGNAL_SCORE  = profile["signal_score"]
        self.SCALPER_ADX_THRESHOLD = profile["adx_threshold"]
        return profile


# ── Single global instance ────────────────────────────────────────────────────
CONFIG = TradingConfig()

# ── Module-level aliases ──────────────────────────────────────────────────────
PROFILE_FILE           = CONFIG.PROFILE_FILE
TRADER_NAME            = CONFIG.TRADER_NAME
TRADER_TIMEZONE        = CONFIG.TRADER_TIMEZONE
GEMINI_ENABLED         = CONFIG.GEMINI_ENABLED
GEMINI_API_KEY         = CONFIG.GEMINI_API_KEY
SCALPER_SCAN_SECS      = CONFIG.SCALPER_M5_SCAN_SECS
DAYTRADER_SCAN_SECS    = CONFIG.DAYTRADER_SCAN_SECS
SCALPER_TF_PRIMARY     = CONFIG.SCALPER_M5_TF_PRIMARY
SCALPER_TF_CONFIRM     = CONFIG.SCALPER_M5_TF_CONFIRM
DAYTRADER_TF_PRIMARY   = CONFIG.DAYTRADER_TF_PRIMARY
DAYTRADER_TF_CONFIRM   = CONFIG.DAYTRADER_TF_CONFIRM
SCALPER_MAX_SPREAD     = CONFIG.SCALPER_M5_MAX_SPREAD
DAYTRADER_MAX_SPREAD   = CONFIG.DAYTRADER_MAX_SPREAD
SCALPER_SIGNAL_SCORE   = CONFIG.SCALPER_M5_SIGNAL_SCORE
DAYTRADER_SIGNAL_SCORE = CONFIG.DAYTRADER_SIGNAL_SCORE
SCALPER_SL_PIPS        = CONFIG.SCALPER_M5_SL_PIPS
DAYTRADER_SL_PIPS      = CONFIG.DAYTRADER_SL_PIPS
SCALPER_TP_PIPS        = CONFIG.SCALPER_M5_TP_PIPS        # Now correctly 12.0
DAYTRADER_TP_PIPS      = CONFIG.DAYTRADER_TP_PIPS
LONDON_OPEN_CET        = CONFIG.LONDON_OPEN_CET
LONDON_CLOSE_CET       = CONFIG.LONDON_CLOSE_CET
NY_OPEN_CET            = CONFIG.NY_OPEN_CET
NY_CLOSE_CET           = CONFIG.NY_CLOSE_CET
OVERLAP_START          = CONFIG.OVERLAP_START
OVERLAP_END            = CONFIG.OVERLAP_END
