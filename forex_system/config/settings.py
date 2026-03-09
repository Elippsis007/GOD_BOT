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
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD",
    ])

    # ── Timeframes ────────────────────────────────────────────────────────────
    PRIMARY_TF:   int = 16385   # H1
    CONFIRM_TF:   int = 16388   # H4
    BARS_HISTORY: int = 1000

    # ── Risk ──────────────────────────────────────────────────────────────────
    RISK_PER_TRADE:     float = 0.01
    MAX_OPEN_TRADES:    int   = 3
    MAX_DAILY_LOSS:     float = 0.03
    RR_RATIO:           float = 2.0
    MAX_LOSS_PER_TRADE: float = 50.0
    TRAILING_STOP_PIPS: int   = 15

    # ── Indicators (shared / day-trader defaults) ─────────────────────────────
    # H1/H4 day-trader periods — scalper overrides are in the profiles below.
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

    # ── ML Model ──────────────────────────────────────────────────────────────
    ML_LOOKBACK:       int   = 60
    ML_MIN_CONFIDENCE: float = 0.35
    ML_FEATURES: List[str] = field(default_factory=lambda: [
        "ema_fast", "ema_slow", "rsi", "bb_upper", "bb_lower",
        "atr", "macd", "macd_signal", "volume",
    ])

    # ── Execution ─────────────────────────────────────────────────────────────
    SLIPPAGE:     int = 20
    MAGIC_NUMBER: int = 202401
    COMMENT:      str = "GODBOT_v1"

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
    TELEGRAM_SEND_HOLD:    bool = False
    TELEGRAM_SEND_EXIT:    bool = False

    # ── Quiet Hours ───────────────────────────────────────────────────────────
    TELEGRAM_QUIET_ON:    bool = True
    TELEGRAM_QUIET_START: int  = 23
    TELEGRAM_QUIET_END:   int  = 8

    # ── Alert Cooldowns (seconds) ─────────────────────────────────────────────
    ALERT_COOLDOWN_SIGNAL: int = 300
    ALERT_COOLDOWN_HOLD:   int = 900
    ALERT_COOLDOWN_EXIT:   int = 600
    ALERT_COOLDOWN_DANGER: int = 180
    ALERT_COOLDOWN_NEWS:   int = 3600

    # ── Alert Confidence Threshold ────────────────────────────────────────────
    ALERT_MIN_CONFIDENCE: float = 0.35

    # ══════════════════════════════════════════════════════════════════════════
    #  SCALPER PROFILES
    #  Two complete, independently tuned profiles — M1 and M5.
    #  Every indicator period, threshold, TP/SL, spread and scan interval
    #  is set correctly for the specific timeframe.
    #
    #  Runtime selection:
    #    SCALPER_TF_SELECTED is set at startup by main.py when the user
    #    picks M1 or M5 from the scalper sub-menu.
    #    get_scalper_profile() returns the correct dict for whichever
    #    timeframe is active — signal_engine.py and main.py call this
    #    once and use the returned values throughout the scan loop.
    #
    #  M1 context (each bar = 1 minute):
    #    RSI(7)   = 7 min lookback  — reacts fast, filters with 80/20
    #    EMA 5/13 = 5 and 13 min    — tight trend detection
    #    MACD 5/13/4                — fastest usable MACD on M1
    #    BB(10)   = 10 min midline  — tight dynamic S/R
    #    ATR(7)   = 7 min volatility — current M1 range
    #    ADX(7) threshold 18        — M1 trends are weak by nature
    #    Stoch 3,3,3                — ultra-fast stochastic for M1
    #
    #  M5 context (each bar = 5 minutes):
    #    RSI(9)   = 45 min lookback — professional M5 sweet spot
    #    EMA 8/21 = 40 and 105 min  — standard M5 scalp pair
    #    MACD 8/21/5                — M5 professional standard
    #    BB(15)   = 75 min midline  — dynamic S/R on M5
    #    ATR(10)  = 50 min volatility — current M5 range
    #    ADX(10) threshold 20       — M5 trend strength standard
    #    Stoch 5,3,3                — industry M5 scalp standard
    # ══════════════════════════════════════════════════════════════════════════

    # ── Active scalper timeframe (set at runtime by main.py) ─────────────────
    # Default is M5.  main.py updates this to 1 or 5 after user selection.
    SCALPER_TF_SELECTED: int = 5

    # ── Scalper M1 Profile ────────────────────────────────────────────────────
    #
    # RSI: period 7, OB/OS 80/20
    #   M1 generates noise — tighter 80/20 bands prevent false OB/OS signals.
    #   Period 7 provides 7-minute lookback, fast enough for M1 momentum.
    #
    # EMA: 5/13/34
    #   Classic M1 scalp triple-EMA.  5 for entry timing, 13 for short-term
    #   trend, 34 for directional bias.
    #
    # MACD: 5/13/4
    #   Fastest practical MACD for M1.  Shorter periods generate too many
    #   whipsaws; this is the minimum viable MACD on M1.
    #
    # BB: period 10, std 2.0
    #   10-bar midline = 10-minute SMA.  Tight enough to show M1 mean-reversion
    #   zones without reacting to every tick.
    #
    # ATR: period 7
    #   7-minute ATR captures current M1 volatility for SL/TP sizing.
    #
    # ADX: period 7, threshold 18
    #   M1 trends rarely exceed ADX 20.  Threshold 18 allows the filter to
    #   identify genuine M1 directional moves without over-blocking.
    #
    # Stochastic: K=3, D=3  (3,3,3 ultra-fast)
    #   Standard ultra-fast stochastic for M1 scalping.
    #
    # TP/SL: 6/3 pips
    #   M1 targets are smaller.  6-pip TP and 3-pip SL gives 2:1 R:R.
    #   Tight SL is essential on M1 to prevent stop runs.
    #
    # Max spread: 0.8 pips
    #   On M1, a 1-pip spread consumes 16% of a 6-pip target.
    #   Hard-blocking above 0.8 pips is essential for M1 edge preservation.
    #
    # Scan interval: 10 seconds
    #   M1 candle = 60 seconds.  10s gives 6 scans per candle — sufficient
    #   to catch signal entries without overloading CPU.
    #
    # Signal score: 4/9
    #   Same minimum confluence as M5.  Higher cost-per-trade ratio on M1
    #   means strict confluence is even more important than on M5.
    #
    # Confirm TF: M5
    #   M5 is the natural higher-timeframe confirmation for M1 signals.
    #
    SCALPER_M1_TF_PRIMARY:   int   = 1      # M1
    SCALPER_M1_TF_CONFIRM:   int   = 5      # M5 confirmation
    SCALPER_M1_RSI_PERIOD:     int   = 7
    SCALPER_M1_RSI_OVERBOUGHT: int   = 80
    SCALPER_M1_RSI_OVERSOLD:   int   = 20
    SCALPER_M1_EMA_FAST:       int   = 5
    SCALPER_M1_EMA_SLOW:       int   = 13
    SCALPER_M1_EMA_TREND:      int   = 34
    SCALPER_M1_MACD_FAST:      int   = 5
    SCALPER_M1_MACD_SLOW:      int   = 13
    SCALPER_M1_MACD_SIGNAL:    int   = 4
    SCALPER_M1_BB_PERIOD:      int   = 10
    SCALPER_M1_BB_STD:         float = 2.0
    SCALPER_M1_ATR_PERIOD:     int   = 7
    SCALPER_M1_ADX_PERIOD:     int   = 7
    SCALPER_M1_ADX_THRESHOLD:  int   = 18
    SCALPER_M1_STOCH_K:        int   = 3
    SCALPER_M1_STOCH_D:        int   = 3
    SCALPER_M1_TP_PIPS:        float = 6.0
    SCALPER_M1_SL_PIPS:        float = 3.0
    SCALPER_M1_MAX_SPREAD:     float = 0.8
    SCALPER_M1_SCAN_SECS:      int   = 10
    SCALPER_M1_SIGNAL_SCORE:   int   = 4
    SCALPER_M1_TRAILING_STOP:  int   = 5    # pips — tighter on M1

    # ── Scalper M5 Profile ────────────────────────────────────────────────────
    #
    # RSI: period 9, OB/OS 75/25
    #   45-minute lookback — professional M5 sweet spot per MC² Finance /
    #   ePlanet Brokers M5 RSI research 2025.  75/25 filters M5 noise
    #   better than the standard 70/30 levels.
    #
    # EMA: 8/21/50
    #   Standard M5 scalp triple-EMA.  8 for entry timing, 21 for short-term
    #   trend, 50 for directional bias.
    #
    # MACD: 8/21/5
    #   Professional M5 scalping MACD.  Reduces slow-EMA lookback from
    #   130 min (12/26/9) to 105 min — more responsive on M5.
    #
    # BB: period 15, std 2.0
    #   75-minute midline aligns with M5 scalp trade duration.
    #
    # ATR: period 10
    #   50-minute ATR — current M5 volatility snapshot.
    #
    # ADX: period 10, threshold 20
    #   M5 directional moves rarely sustain ADX above 25.  Threshold 20
    #   correctly identifies M5 trend strength without over-filtering.
    #
    # Stochastic: K=5, D=3  (5,3,3 industry standard for M5)
    #   25-minute lookback — matches the pace of M5 scalp setups.
    #
    # TP/SL: 12/6 pips
    #   Within the 10–15 pip professional M5 TP range.
    #   6-pip SL within the 5–8 pip professional M5 SL range.
    #
    # Max spread: 1.2 pips
    #   Pepperstone EURUSD typical: 0.6–1.0 pips during London/Overlap.
    #   Hard-blocks spread widening events that would consume >10% of TP.
    #
    # Scan interval: 20 seconds
    #   M5 candle = 300 seconds.  20s gives ~15 scans per candle — tight
    #   enough to catch fast signal setups, light enough for CPU.
    #
    # Signal score: 4/9
    #   Stricter confluence reduces trade frequency ~30–40% but eliminates
    #   the weakest signals that barely cover spread cost.
    #
    # Confirm TF: M15
    #   M15 is the natural higher-timeframe confirmation for M5 signals.
    #
    SCALPER_M5_TF_PRIMARY:   int   = 5      # M5
    SCALPER_M5_TF_CONFIRM:   int   = 15     # M15 confirmation
    SCALPER_M5_RSI_PERIOD:     int   = 9
    SCALPER_M5_RSI_OVERBOUGHT: int   = 75
    SCALPER_M5_RSI_OVERSOLD:   int   = 25
    SCALPER_M5_EMA_FAST:       int   = 8
    SCALPER_M5_EMA_SLOW:       int   = 21
    SCALPER_M5_EMA_TREND:      int   = 50
    SCALPER_M5_MACD_FAST:      int   = 8
    SCALPER_M5_MACD_SLOW:      int   = 21
    SCALPER_M5_MACD_SIGNAL:    int   = 5
    SCALPER_M5_BB_PERIOD:      int   = 15
    SCALPER_M5_BB_STD:         float = 2.0
    SCALPER_M5_ATR_PERIOD:     int   = 10
    SCALPER_M5_ADX_PERIOD:     int   = 10
    SCALPER_M5_ADX_THRESHOLD:  int   = 20
    SCALPER_M5_STOCH_K:        int   = 5
    SCALPER_M5_STOCH_D:        int   = 3
    SCALPER_M5_TP_PIPS:        float = 12.0
    SCALPER_M5_SL_PIPS:        float = 6.0
    SCALPER_M5_MAX_SPREAD:     float = 1.2
    SCALPER_M5_SCAN_SECS:      int   = 20
    SCALPER_M5_SIGNAL_SCORE:   int   = 4
    SCALPER_M5_TRAILING_STOP:  int   = 8    # pips — slightly wider on M5

    # ── Day Trader Style ──────────────────────────────────────────────────────
    # FIXED: DAYTRADER_TF_PRIMARY was 900 (invalid MT5 int) → 15 (M15)
    DAYTRADER_TF_PRIMARY:   int   = 15      # M15
    DAYTRADER_TF_CONFIRM:   int   = 16385   # H1
    DAYTRADER_TP_PIPS:      float = 30.0
    DAYTRADER_SL_PIPS:      float = 15.0
    DAYTRADER_MAX_SPREAD:   float = 1.5
    DAYTRADER_SCAN_SECS:    int   = 60
    DAYTRADER_SIGNAL_SCORE: int   = 4

    # ── Session Times (CET hour) ──────────────────────────────────────────────
    LONDON_OPEN_CET:  int = 8
    LONDON_CLOSE_CET: int = 12
    NY_OPEN_CET:      int = 14
    NY_CLOSE_CET:     int = 23
    OVERLAP_START:    int = 14
    OVERLAP_END:      int = 17

    # ── Profile File ──────────────────────────────────────────────────────────
    PROFILE_FILE: str = "data/profile.json"

    # ══════════════════════════════════════════════════════════════════════════
    #  get_scalper_profile()
    #
    #  Returns a single flat dictionary containing every setting for the
    #  currently selected scalper timeframe (M1 or M5).
    #
    #  Usage in signal_engine.py, main.py, indicators.py:
    #
    #      from config.settings import CONFIG
    #      p = CONFIG.get_scalper_profile()
    #
    #      # Then access any setting by key:
    #      rsi_period     = p["rsi_period"]
    #      adx_threshold  = p["adx_threshold"]
    #      tp_pips        = p["tp_pips"]
    #      scan_secs      = p["scan_secs"]
    #      tf_primary     = p["tf_primary"]
    #
    #  main.py sets SCALPER_TF_SELECTED before the scan loop starts:
    #      CONFIG.SCALPER_TF_SELECTED = 1   # user chose M1
    #      CONFIG.SCALPER_TF_SELECTED = 5   # user chose M5
    # ══════════════════════════════════════════════════════════════════════════

    def get_scalper_profile(self) -> Dict[str, Any]:
        """
        Return the complete scalper settings dictionary for the currently
        selected timeframe (SCALPER_TF_SELECTED = 1 for M1, 5 for M5).

        All keys are lowercase strings so callers use clean dict access
        rather than long CONFIG.SCALPER_M5_RSI_PERIOD-style attribute names.

        Falls back to M5 for any unrecognised timeframe value.
        """
        if self.SCALPER_TF_SELECTED == 1:
            return {
                # ── Timeframes ────────────────────────────────────────────
                "tf_primary":    self.SCALPER_M1_TF_PRIMARY,
                "tf_confirm":    self.SCALPER_M1_TF_CONFIRM,
                "tf_label":      "M1",
                # ── RSI ───────────────────────────────────────────────────
                "rsi_period":     self.SCALPER_M1_RSI_PERIOD,
                "rsi_overbought": self.SCALPER_M1_RSI_OVERBOUGHT,
                "rsi_oversold":   self.SCALPER_M1_RSI_OVERSOLD,
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
                "stoch_k": self.SCALPER_M1_STOCH_K,
                "stoch_d": self.SCALPER_M1_STOCH_D,
                # ── Trade sizing ──────────────────────────────────────────
                "tp_pips":       self.SCALPER_M1_TP_PIPS,
                "sl_pips":       self.SCALPER_M1_SL_PIPS,
                "trailing_stop": self.SCALPER_M1_TRAILING_STOP,
                # ── Execution ─────────────────────────────────────────────
                "max_spread":   self.SCALPER_M1_MAX_SPREAD,
                "scan_secs":    self.SCALPER_M1_SCAN_SECS,
                "signal_score": self.SCALPER_M1_SIGNAL_SCORE,
            }

        # Default — M5 (also handles any unrecognised TF value)
        return {
            # ── Timeframes ────────────────────────────────────────────────
            "tf_primary":    self.SCALPER_M5_TF_PRIMARY,
            "tf_confirm":    self.SCALPER_M5_TF_CONFIRM,
            "tf_label":      "M5",
            # ── RSI ───────────────────────────────────────────────────────
            "rsi_period":     self.SCALPER_M5_RSI_PERIOD,
            "rsi_overbought": self.SCALPER_M5_RSI_OVERBOUGHT,
            "rsi_oversold":   self.SCALPER_M5_RSI_OVERSOLD,
            # ── EMA ───────────────────────────────────────────────────────
            "ema_fast":  self.SCALPER_M5_EMA_FAST,
            "ema_slow":  self.SCALPER_M5_EMA_SLOW,
            "ema_trend": self.SCALPER_M5_EMA_TREND,
            # ── MACD ──────────────────────────────────────────────────────
            "macd_fast":   self.SCALPER_M5_MACD_FAST,
            "macd_slow":   self.SCALPER_M5_MACD_SLOW,
            "macd_signal": self.SCALPER_M5_MACD_SIGNAL,
            # ── Bollinger Bands ───────────────────────────────────────────
            "bb_period": self.SCALPER_M5_BB_PERIOD,
            "bb_std":    self.SCALPER_M5_BB_STD,
            # ── ATR ───────────────────────────────────────────────────────
            "atr_period": self.SCALPER_M5_ATR_PERIOD,
            # ── ADX ───────────────────────────────────────────────────────
            "adx_period":    self.SCALPER_M5_ADX_PERIOD,
            "adx_threshold": self.SCALPER_M5_ADX_THRESHOLD,
            # ── Stochastic ────────────────────────────────────────────────
            "stoch_k": self.SCALPER_M5_STOCH_K,
            "stoch_d": self.SCALPER_M5_STOCH_D,
            # ── Trade sizing ──────────────────────────────────────────────
            "tp_pips":       self.SCALPER_M5_TP_PIPS,
            "sl_pips":       self.SCALPER_M5_SL_PIPS,
            "trailing_stop": self.SCALPER_M5_TRAILING_STOP,
            # ── Execution ─────────────────────────────────────────────────
            "max_spread":   self.SCALPER_M5_MAX_SPREAD,
            "scan_secs":    self.SCALPER_M5_SCAN_SECS,
            "signal_score": self.SCALPER_M5_SIGNAL_SCORE,
        }


# ── Single global instance ────────────────────────────────────────────────────
CONFIG = TradingConfig()

# ── Module-level aliases ──────────────────────────────────────────────────────
PROFILE_FILE           = CONFIG.PROFILE_FILE
TRADER_NAME            = CONFIG.TRADER_NAME
TRADER_TIMEZONE        = CONFIG.TRADER_TIMEZONE
GEMINI_ENABLED         = CONFIG.GEMINI_ENABLED
GEMINI_API_KEY         = CONFIG.GEMINI_API_KEY
SCALPER_SCAN_SECS      = CONFIG.SCALPER_M5_SCAN_SECS      # default M5
DAYTRADER_SCAN_SECS    = CONFIG.DAYTRADER_SCAN_SECS
SCALPER_TF_PRIMARY     = CONFIG.SCALPER_M5_TF_PRIMARY     # default M5
SCALPER_TF_CONFIRM     = CONFIG.SCALPER_M5_TF_CONFIRM     # default M5
DAYTRADER_TF_PRIMARY   = CONFIG.DAYTRADER_TF_PRIMARY
DAYTRADER_TF_CONFIRM   = CONFIG.DAYTRADER_TF_CONFIRM
SCALPER_MAX_SPREAD     = CONFIG.SCALPER_M5_MAX_SPREAD      # default M5
DAYTRADER_MAX_SPREAD   = CONFIG.DAYTRADER_MAX_SPREAD
SCALPER_SIGNAL_SCORE   = CONFIG.SCALPER_M5_SIGNAL_SCORE    # default M5
DAYTRADER_SIGNAL_SCORE = CONFIG.DAYTRADER_SIGNAL_SCORE
SCALPER_SL_PIPS        = CONFIG.SCALPER_M5_SL_PIPS         # default M5
DAYTRADER_SL_PIPS      = CONFIG.DAYTRADER_SL_PIPS
SCALPER_TP_PIPS        = CONFIG.SCALPER_M5_TP_PIPS         # default M5
DAYTRADER_TP_PIPS      = CONFIG.DAYTRADER_TP_PIPS
LONDON_OPEN_CET        = CONFIG.LONDON_OPEN_CET
LONDON_CLOSE_CET       = CONFIG.LONDON_CLOSE_CET
NY_OPEN_CET            = CONFIG.NY_OPEN_CET
NY_CLOSE_CET           = CONFIG.NY_CLOSE_CET
OVERLAP_START          = CONFIG.OVERLAP_START
OVERLAP_END            = CONFIG.OVERLAP_END
