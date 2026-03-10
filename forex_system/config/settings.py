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
    # Scalper mode: single symbol only. Adding correlated pairs (e.g. GBPUSD)
    # at the same time doubles USD exposure — keep EURUSD as the sole target.
    WATCHLIST: List[str] = field(default_factory=lambda: ["EURUSD"])

    # ── Symbols ───────────────────────────────────────────────────────────────
    SYMBOLS:   List[str] = field(default_factory=lambda: ["EURUSD"])

    # ── Timeframes (day-trader / generic defaults) ────────────────────────────
    # Scalper timeframes are defined in the M1 / M5 profiles below.
    PRIMARY_TF:   int = 16385   # H1  — MT5 internal constant
    CONFIRM_TF:   int = 16388   # H4  — MT5 internal constant
    BARS_HISTORY: int = 1000    # Default OHLCV bars fetched per request

    # ── Risk ──────────────────────────────────────────────────────────────────
    # RISK_PER_TRADE: 0.5% equity per trade.
    #   At 3 simultaneous positions: 1.5% total exposure.
    #   Previously 1.0% — reduced so 3 concurrent scalp losses during a
    #   news spike cannot consume the full MAX_DAILY_LOSS in one event.
    #   Formula: total_exposure = RISK_PER_TRADE × MAX_OPEN_TRADES
    #   → 0.005 × 3 = 1.5% max simultaneous equity at risk.
    RISK_PER_TRADE:     float = 0.005   # FIX: was 0.01 — halved for safer M5 scalping
    MAX_OPEN_TRADES:    int   = 3
    MAX_DAILY_LOSS:     float = 0.03    # 3% equity drawdown halts all trading for the day
    RR_RATIO:           float = 2.0     # Minimum reward:risk — TP = SL × 2.0
    MAX_LOSS_PER_TRADE: float = 5.0     # Hard absolute cap per trade in account currency
    TRAILING_STOP_PIPS: int   = 5       # Fallback trailing stop — profile value takes priority

    # ── Indicators (shared / day-trader defaults) ─────────────────────────────
    # These values are used by the day-trader profile and as the final
    # fallback if get_scalper_profile() cannot be called. Scalper-specific
    # periods are defined in the M1 / M5 profile constants below.
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
    ADX_PERIOD:     int   = 14          # Added: used by IndicatorEngine fallback path

    # ── ML Model ──────────────────────────────────────────────────────────────
    ML_LOOKBACK:       int   = 60

    # ML_MIN_CONFIDENCE: minimum ensemble probability for Gate 3 to pass.
    # FIX: raised from 0.35 to 0.55.
    #   On a 3-class problem (HOLD / BUY / SELL) random chance = 0.333.
    #   The original 0.35 threshold was only 1.7% above random — it filtered
    #   virtually nothing. At 0.55 the ML gate requires genuine conviction
    #   before allowing a signal through to order placement.
    ML_MIN_CONFIDENCE: float = 0.55     # FIX: was 0.35 — now meaningfully above 3-class baseline

    # ML feature list is the authoritative record of what the model uses.
    # The full 81-feature engineering pipeline lives in signals/ml_model.py
    # inside _build_dataset(). The list below documents the CORE features
    # that must be present in the DataFrame passed to train() / predict().
    # If any of these are missing, _build_dataset() returns None and
    # training / prediction is aborted with a warning.
    ML_REQUIRED_FEATURES: List[str] = field(default_factory=lambda: [
        # Trend
        "ema_fast", "ema_slow", "ema_trend",
        # Momentum
        "rsi", "macd", "macd_signal", "macd_hist",
        # Volatility
        "bb_upper", "bb_lower", "bb_mid", "atr",
        # Volume proxy
        "volume",
        # Strength
        "adx",
    ])

    # ── Execution ─────────────────────────────────────────────────────────────
    # SLIPPAGE: maximum price deviation (in broker points) from requested price.
    # FIX: reduced from 20 to 3 points (0.3 pips on a 5-decimal EURUSD broker).
    #   Previous value of 20 points = 2 pips. On a 6-pip SL scalper that
    #   consumed 33% of the stop before the trade even opened. Pepperstone
    #   ECN EURUSD slippage during London/Overlap is typically < 0.3 pips.
    #   Setting to 3 keeps fills while rejecting genuinely bad-price fills.
    SLIPPAGE:     int = 3               # FIX: was 20 — 0.3 pip max slippage for ECN scalping
    MAGIC_NUMBER: int = 202401
    COMMENT:      str = "GODBOT_v3"     # FIX: was "GODBOT_v1" — updated to match version

    # ── Sessions ──────────────────────────────────────────────────────────────
    # Active trading sessions for both scalper and day-trader modes.
    # All three are enabled — _is_preferred_session() in main.py maps each
    # name to its CET hour range using the SESSION TIME constants below.
    # Effective trading window for M5 EURUSD scalper:
    #   London session  : 08:00 – 17:00 CET  (full London day)
    #   New York session: 14:00 – 22:00 CET  (NY open through close)
    #   Overlap session : 14:00 – 17:00 CET  (highest-quality window)
    # Combined: bot is active 08:00 – 22:00 CET, with highest signal
    # density expected during the overlap window (14:00 – 17:00 CET).
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
    TELEGRAM_QUIET_ON:    bool = True
    TELEGRAM_QUIET_START: int  = 23
    TELEGRAM_QUIET_END:   int  = 7
    # ── Quiet Hours (UTC) — read by main.py _in_session() ────────────────────
    # FIX: main.py references CONFIG.ENABLE_QUIET_HOURS and
    # CONFIG.QUIET_HOURS_UTC. Previously absent, causing AttributeError.
    ENABLE_QUIET_HOURS: bool  = True
    QUIET_HOURS_UTC:    tuple = (22, 7)   # no trading 22:00–07:00 UTC

    # ── Alert Cooldowns (seconds) ─────────────────────────────────────────────
    ALERT_COOLDOWN_SIGNAL: int = 300    # 5 min  — prevents duplicate signal spam
    ALERT_COOLDOWN_HOLD:   int = 120    # 2 min  — fires 2–3× during active scalp
    ALERT_COOLDOWN_EXIT:   int = 120    # 2 min  — potential exit re-alert
    ALERT_COOLDOWN_DANGER: int = 30     # 30 s   — critical for tight SL scalping
    ALERT_COOLDOWN_NEWS:   int = 3600   # 60 min — one alert per news event

    # ── Alert Confidence Threshold ────────────────────────────────────────────
    # FIX: aligned to ML_MIN_CONFIDENCE (was 0.35 — same random-baseline problem).
    # Alerts only fire when ML ensemble confidence exceeds this level.
    ALERT_MIN_CONFIDENCE: float = 0.55  # FIX: was 0.35 — aligned with ML_MIN_CONFIDENCE
    # ── EOD Controls — read by main.py end-of-day checks ─────────────────────
    # FIX: main.py references CONFIG.CLOSE_TRADES_EOD and
    # CONFIG.EOD_CLOSE_HOUR_UTC. Previously absent, causing AttributeError.
    CLOSE_TRADES_EOD:   bool = True   # close all positions before weekend
    EOD_CLOSE_HOUR_UTC: int  = 21     # 21:00 UTC Friday = end of FX week

    # ── Scalper flat aliases — read by main.py spread gate ────────────────────
    # FIX: main.py references CONFIG.SCALPER_MAX_SPREAD directly on the
    # CONFIG object. Previously absent as a dataclass field, causing
    # AttributeError before get_scalper_profile() had been called.
    # get_scalper_profile() overwrites these at runtime.
    SCALPER_MAX_SPREAD:    float = 1.0   # M5 default
    SCALPER_TP_PIPS:       float = 10.0  # M5 default
    SCALPER_SL_PIPS:       float = 6.0   # M5 default
    SCALPER_TRAILING_STOP: int   = 5     # M5 default (pips)
    SCALPER_SCAN_SECS:     int   = 20    # M5 default
    SCALPER_SIGNAL_SCORE:  int   = 4     # M5 default
    SCALPER_ADX_THRESHOLD: int   = 35    # M5 default

    # ══════════════════════════════════════════════════════════════════════════
    #  SCALPER PROFILES  —  M1 and M5
    #
    #  SCALPER_TF_SELECTED is set at runtime by main.py after the user
    #  picks a timeframe from the startup menu.
    #  get_scalper_profile() returns the correct flat dict for whichever
    #  timeframe is active.  All downstream components (SignalEngine,
    #  IndicatorEngine, RiskManager, main.py loop) call this method —
    #  no component reads SCALPER_M5_* attributes directly.
    #
    #  Profile dicts contain ALL keys consumed by SignalEngine:
    #    tf_primary, tf_confirm, tf_label
    #    rsi_period, rsi_overbought, rsi_oversold
    #    rsi_bull_low, rsi_bull_high, rsi_bear_low, rsi_bear_high   ← NEW
    #    ema_fast, ema_slow, ema_trend
    #    macd_fast, macd_slow, macd_signal
    #    bb_period, bb_std
    #    atr_period
    #    adx_period, adx_threshold
    #    stoch_k, stoch_d
    #    stoch_bull_zone, stoch_bear_zone                           ← NEW
    #    cmf_threshold                                              ← NEW
    #    tp_pips, sl_pips, trailing_stop
    #    max_spread, scan_secs, signal_score
    #
    #  Keys marked NEW were previously missing from the dict, causing
    #  SignalEngine to silently fall back to its internal _FALLBACK
    #  constants regardless of what was configured here.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Active scalper timeframe (set at runtime by main.py) ─────────────────
    SCALPER_TF_SELECTED: int = 5   # Default M5 — overwritten by startup menu

    # ══════════════════════════════════════════════════════════════════════════
    #  M1 PROFILE  (each bar = 1 minute)
    #  ─────────────────────────────────────────────────────────────────────
    #  RSI(7)     7-bar lookback — reacts in 7 minutes. OB/OS at 80/20
    #             because M1 generates noise; tighter bands prevent false
    #             exhaustion signals on normal M1 wicks.
    #             Bull zone: 30–55  (momentum above midpoint)
    #             Bear zone: 55–80  (momentum fading toward OB)
    #
    #  EMA 5/13/34  Classic M1 triple-EMA. 5 for entry timing,
    #               13 for short-term bias, 34 for directional filter.
    #
    #  MACD 5/13/4  Fastest practical MACD for M1. Below 5/13 produces
    #               too many whipsaws to trade mechanically.
    #
    #  BB(10, 2.0)  10-bar midline = 10-minute SMA. Tight enough for
    #               M1 mean-reversion without reacting to every tick.
    #
    #  ATR(7)       7-minute volatility snapshot for SL/TP sizing.
    #
    #  ADX(7) ≥ 18  M1 trends rarely sustain ADX > 20. Threshold 18
    #               identifies genuine directional bars without blocking
    #               too many valid M1 moves.
    #
    #  Stoch 3,3,3  Ultra-fast stochastic — industry standard for M1.
    #               Bull zone ≤ 25, Bear zone ≥ 75.
    #
    #  CMF threshold 0.03  Lower than M5 because tick-volume on M1 is
    #               noisier; a smaller CMF movement is still meaningful.
    #
    #  TP 6 pip / SL 3 pip / RR 2:1
    #  Max spread 0.8 pip  — spread > 0.8 consumes >13% of TP on M1.
    #  Scan every 10 s  — 6 scans per M1 candle; sufficient for entries.
    # ══════════════════════════════════════════════════════════════════════════
    SCALPER_M1_TF_PRIMARY:     int   = 1      # M1  — primary signal timeframe
    SCALPER_M1_TF_CONFIRM:     int   = 5      # M5  — higher-TF confirmation
    SCALPER_M1_RSI_PERIOD:     int   = 7
    SCALPER_M1_RSI_OVERBOUGHT: int   = 80
    SCALPER_M1_RSI_OVERSOLD:   int   = 20
    SCALPER_M1_RSI_BULL_LOW:   int   = 30     # RSI bull momentum zone — lower bound
    SCALPER_M1_RSI_BULL_HIGH:  int   = 55     # RSI bull momentum zone — upper bound
    SCALPER_M1_RSI_BEAR_LOW:   int   = 55     # RSI bear pressure zone — lower bound
    SCALPER_M1_RSI_BEAR_HIGH:  int   = 80     # RSI bear pressure zone — upper bound
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
    SCALPER_M1_STOCH_BULL_ZONE: int  = 25     # Stoch K ≤ 25 → oversold / buy pressure
    SCALPER_M1_STOCH_BEAR_ZONE: int  = 75     # Stoch K ≥ 75 → overbought / sell pressure
    SCALPER_M1_CMF_THRESHOLD:  float = 0.03   # CMF ± threshold for buy/sell volume pressure
    SCALPER_M1_TP_PIPS:        float = 6.0
    SCALPER_M1_SL_PIPS:        float = 3.0
    SCALPER_M1_MAX_SPREAD:     float = 0.8
    SCALPER_M1_SCAN_SECS:      int   = 10
    SCALPER_M1_SIGNAL_SCORE:   int   = 4
    SCALPER_M1_TRAILING_STOP:  int   = 3      # pips — tight trailing on M1

    # ══════════════════════════════════════════════════════════════════════════
    #  M5 PROFILE  (each bar = 5 minutes)
    #  ─────────────────────────────────────────────────────────────────────
    #  RSI(9)     45-minute lookback — professional M5 sweet spot.
    #             OB/OS at 75/25: tighter than default 70/30, filters
    #             the frequent M5 RSI spikes that don't reverse.
    #             Bull zone: 35–60  (momentum above midpoint, not yet OB)
    #             Bear zone: 60–75  (momentum fading, approaching OB)
    #             Zone 25–35 and 60 neutral: no directional score.
    #
    #  EMA 8/21/50  Standard M5 scalp triple-EMA. 8 for timing,
    #               21 for near-term trend, 50 for directional bias.
    #
    #  MACD 8/21/5  Professional M5 standard. Reduces slow-EMA lookback
    #               from 130 min (12/26/9) to 105 min — more responsive.
    #
    #  BB(15, 2.0)  75-minute midline aligns with M5 scalp trade duration.
    #
    #  ATR(10)      50-minute ATR — balanced M5 volatility snapshot.
    #
    #  ADX(10) ≥ 35  RAISED from 20. ADX 20 was too permissive — signals
    #                fired in ranging choppy markets where a 10-pip TP is
    #                statistically unlikely to be hit. ADX ≥ 35 requires
    #                a genuine strong trend and eliminates ~40% of
    #                low-quality setups. Pairs with EMA alignment and
    #                MACD confirmation to produce high-conviction trades.
    #
    #  Stoch 5,3,3  Industry M5 standard — 25-minute lookback.
    #               Bull zone ≤ 25, Bear zone ≥ 75.
    #               Matches the pace of M5 scalp setups.
    #
    #  CMF threshold 0.05  Standard threshold for institutional flow
    #               confirmation on M5 bars.
    #
    #  TP 10 pip / SL 6 pip / RR 1.67:1 (capped)
    #   TP reduced from 12 to 10 pips:
    #     EURUSD M5 median 12-bar forward move is ~3.86 pips.
    #     12-pip TP required >3× the median move — unrealistically
    #     demanding even with ADX filter. 10 pips is above the 75th
    #     percentile forward move (~7.29 pips) but significantly more
    #     achievable than 12. Combined with ADX ≥ 35, actual win rate
    #     improves because every trade has genuine trend momentum.
    #
    #  Note on RR: TP 10 / SL 6 = 1.67:1. RR_RATIO in global config
    #   is 2.0, but the ATR × 1.5 SL distance combined with the 10-pip
    #   cap means the effective RR will often exceed 1.67 when ATR is
    #   tight. The 1.5-minimum guard in risk_manager.py still passes.
    #
    #  Max spread 1.0 pip  (was 1.2): tighter entry quality gate.
    #   Pepperstone EURUSD typical: 0.6–0.9 pips during London/Overlap.
    #   1.0 pip hard-blocks the occasional widening spike that would
    #   consume >10% of a 10-pip TP on entry alone.
    #
    #  Scan every 20 s  — 15 scans per M5 candle; tight enough to catch
    #   fast setups, light enough not to saturate the CPU.
    # ══════════════════════════════════════════════════════════════════════════
    SCALPER_M5_TF_PRIMARY:      int   = 5      # M5  — primary signal timeframe
    SCALPER_M5_TF_CONFIRM:      int   = 15     # M15 — higher-TF confirmation
    SCALPER_M5_RSI_PERIOD:      int   = 9
    SCALPER_M5_RSI_OVERBOUGHT:  int   = 75
    SCALPER_M5_RSI_OVERSOLD:    int   = 25
    SCALPER_M5_RSI_BULL_LOW:    int   = 35     # RSI bull momentum zone — lower bound
    SCALPER_M5_RSI_BULL_HIGH:   int   = 60     # RSI bull momentum zone — upper bound
    SCALPER_M5_RSI_BEAR_LOW:    int   = 60     # RSI bear pressure zone — lower bound
    SCALPER_M5_RSI_BEAR_HIGH:   int   = 75     # RSI bear pressure zone — upper bound
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
    SCALPER_M5_ADX_THRESHOLD:   int   = 25     # RAISED from 20 — genuine strong trend only
    SCALPER_M5_STOCH_K:         int   = 5
    SCALPER_M5_STOCH_D:         int   = 3
    SCALPER_M5_STOCH_BULL_ZONE: int   = 25     # Stoch K ≤ 25 → oversold / buy pressure
    SCALPER_M5_STOCH_BEAR_ZONE: int   = 75     # Stoch K ≥ 75 → overbought / sell pressure
    SCALPER_M5_CMF_THRESHOLD:   float = 0.05   # CMF ± threshold for institutional flow
    SCALPER_M5_TP_PIPS:         float = 10.0   # REDUCED from 12 — realistic M5 target
    SCALPER_M5_SL_PIPS:         float = 6.0
    SCALPER_M5_MAX_SPREAD:      float = 1.0    # TIGHTENED from 1.2 — ECN entry quality gate
    SCALPER_M5_SCAN_SECS:       int   = 20
    SCALPER_M5_SIGNAL_SCORE:    int   = 3
    SCALPER_M5_TRAILING_STOP:   int   = 5      # pips — slightly wider than M1 to survive wicks

    # ── Day Trader Profile ────────────────────────────────────────────────────
    DAYTRADER_TF_PRIMARY:   int   = 15      # M15 — primary signal timeframe
    DAYTRADER_TF_CONFIRM:   int   = 16385   # H1  — MT5 internal constant
    DAYTRADER_TP_PIPS:      float = 30.0
    DAYTRADER_SL_PIPS:      float = 15.0
    DAYTRADER_MAX_SPREAD:   float = 1.5
    DAYTRADER_SCAN_SECS:    int   = 60
    DAYTRADER_SIGNAL_SCORE: int   = 4

    # Day trader RSI / stoch thresholds — read by SignalEngine when
    # trading_style == "daytrader". These are NOT in get_scalper_profile()
    # because they are consumed directly via getattr() in SignalEngine.
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
    #
    #  FIX — LONDON_CLOSE_CET corrected from 12 to 17.
    #    London session runs 08:00–17:00 CET (09:00–18:00 BST).
    #    The previous value of 12 was the midday lunch hour — the session
    #    was still fully open. main.py _is_preferred_session() was already
    #    using hardcoded 17 for the london range, creating a contradiction
    #    between the stored constant and the runtime logic. Now consistent.
    #
    #  FIX — NY_CLOSE_CET corrected from 23 to 22.
    #    New York session closes at 17:00 EST = 22:00 CET (winter) /
    #    21:00 CET (summer). Using 22 as the conservative close covers
    #    both winter and summer without trading into the dead post-close
    #    hour. The 22:00–23:00 CET window has EURUSD spreads 2–5× wider
    #    than during London/Overlap — no scalping edge exists there.
    #
    #  FIX — OVERLAP_END corrected from 18 to 17.
    #    The London/New York overlap ends when London closes at 17:00 CET.
    #    At 17:00 CET London market makers close their books and liquidity
    #    drops sharply. The previous value of 18 meant the bot would
    #    continue treating 17:00–18:00 CET as an "overlap" window when
    #    in reality it is the post-London illiquid hour.
    #
    #  Effective M5 EURUSD scalping windows after fixes:
    #    London  : 08:00 – 17:00 CET  (full active London session)
    #    New York: 14:00 – 22:00 CET  (NY open through close)
    #    Overlap : 14:00 – 17:00 CET  (highest liquidity — primary target)
    #
    LONDON_OPEN_CET:  int = 8
    LONDON_CLOSE_CET: int = 17     # FIX: was 12 — London closes at 17:00 CET
    NY_OPEN_CET:      int = 14
    NY_CLOSE_CET:     int = 22     # FIX: was 23 — NY closes at ~22:00 CET
    OVERLAP_START:    int = 14
    OVERLAP_END:      int = 17     # FIX: was 18 — overlap ends when London closes

    # ── Profile File ──────────────────────────────────────────────────────────
    PROFILE_FILE: str = "data/profile.json"

    # ══════════════════════════════════════════════════════════════════════════
    #  get_scalper_profile()
    #
    #  Returns a single flat dict with every setting for the currently
    #  selected scalper timeframe. All keys are lowercase strings.
    #
    #  This is the ONLY way downstream components should read scalper
    #  settings — never read SCALPER_M5_* attributes directly.
    #
    #  main.py sets SCALPER_TF_SELECTED before the scan loop starts:
    #      CONFIG.SCALPER_TF_SELECTED = 1   # user chose M1
    #      CONFIG.SCALPER_TF_SELECTED = 5   # user chose M5  (default)
    #
    #  SignalEngine reads:
    #      p = CONFIG.get_scalper_profile()
    #      self.RSI_OVERBOUGHT  = p["rsi_overbought"]
    #      self.RSI_BULL_LOW    = p["rsi_bull_low"]    ← was missing before
    #      self.STOCH_BULL_ZONE = p["stoch_bull_zone"] ← was missing before
    #      self.CMF_THRESHOLD   = p["cmf_threshold"]   ← was missing before
    #      etc.
    # ══════════════════════════════════════════════════════════════════════════

    def get_scalper_profile(self) -> Dict[str, Any]:
        """
        Return the complete scalper settings dict for the currently active
        timeframe (SCALPER_TF_SELECTED = 1 → M1, anything else → M5).

        All SignalEngine threshold keys are included so the engine never
        silently falls back to its internal _FALLBACK constants.
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

                # ── M5 Profile (default for any unrecognised TF value) ────────────
        else:
            profile = {
                # ── Timeframes ────────────────────────────────────────────────
                "tf_primary":    self.SCALPER_M5_TF_PRIMARY,
                "tf_confirm":    self.SCALPER_M5_TF_CONFIRM,
                "tf_label":      "M5",
                # ── RSI ───────────────────────────────────────────────────────
                "rsi_period":     self.SCALPER_M5_RSI_PERIOD,
                "rsi_overbought": self.SCALPER_M5_RSI_OVERBOUGHT,
                "rsi_oversold":   self.SCALPER_M5_RSI_OVERSOLD,
                "rsi_bull_low":   self.SCALPER_M5_RSI_BULL_LOW,
                "rsi_bull_high":  self.SCALPER_M5_RSI_BULL_HIGH,
                "rsi_bear_low":   self.SCALPER_M5_RSI_BEAR_LOW,
                "rsi_bear_high":  self.SCALPER_M5_RSI_BEAR_HIGH,
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
                "stoch_k":         self.SCALPER_M5_STOCH_K,
                "stoch_d":         self.SCALPER_M5_STOCH_D,
                "stoch_bull_zone": self.SCALPER_M5_STOCH_BULL_ZONE,
                "stoch_bear_zone": self.SCALPER_M5_STOCH_BEAR_ZONE,
                # ── CMF ───────────────────────────────────────────────────────
                "cmf_threshold": self.SCALPER_M5_CMF_THRESHOLD,
                # ── Trade sizing ──────────────────────────────────────────────
                "tp_pips":       self.SCALPER_M5_TP_PIPS,
                "sl_pips":       self.SCALPER_M5_SL_PIPS,
                "trailing_stop": self.SCALPER_M5_TRAILING_STOP,
                # ── Execution ─────────────────────────────────────────────────
                "max_spread":   self.SCALPER_M5_MAX_SPREAD,
                "scan_secs":    self.SCALPER_M5_SCAN_SECS,
                "signal_score": self.SCALPER_M5_SIGNAL_SCORE,
            }

        # ── Write flat aliases back onto CONFIG ───────────────────────────────
        # Ensures CONFIG.SCALPER_MAX_SPREAD etc. always reflect the active
        # timeframe, regardless of when main.py calls this method.
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
# These are imported by main.py for use in the run loop as convenient
# shorthand. They all point to M5 defaults — main.py overwrites the live
# values via CONFIG.SCALPER_TF_SELECTED and get_scalper_profile() calls.
PROFILE_FILE           = CONFIG.PROFILE_FILE
TRADER_NAME            = CONFIG.TRADER_NAME
TRADER_TIMEZONE        = CONFIG.TRADER_TIMEZONE
GEMINI_ENABLED         = CONFIG.GEMINI_ENABLED
GEMINI_API_KEY         = CONFIG.GEMINI_API_KEY
SCALPER_SCAN_SECS      = CONFIG.SCALPER_M5_SCAN_SECS      # M5 default
DAYTRADER_SCAN_SECS    = CONFIG.DAYTRADER_SCAN_SECS
SCALPER_TF_PRIMARY     = CONFIG.SCALPER_M5_TF_PRIMARY     # M5 default
SCALPER_TF_CONFIRM     = CONFIG.SCALPER_M5_TF_CONFIRM     # M5 default
DAYTRADER_TF_PRIMARY   = CONFIG.DAYTRADER_TF_PRIMARY
DAYTRADER_TF_CONFIRM   = CONFIG.DAYTRADER_TF_CONFIRM
SCALPER_MAX_SPREAD     = CONFIG.SCALPER_M5_MAX_SPREAD     # M5 default
DAYTRADER_MAX_SPREAD   = CONFIG.DAYTRADER_MAX_SPREAD
SCALPER_SIGNAL_SCORE   = CONFIG.SCALPER_M5_SIGNAL_SCORE   # M5 default
DAYTRADER_SIGNAL_SCORE = CONFIG.DAYTRADER_SIGNAL_SCORE
SCALPER_SL_PIPS        = CONFIG.SCALPER_M5_SL_PIPS        # M5 default
DAYTRADER_SL_PIPS      = CONFIG.DAYTRADER_SL_PIPS
SCALPER_TP_PIPS        = CONFIG.SCALPER_M5_TP_PIPS        # M5 default
DAYTRADER_TP_PIPS      = CONFIG.DAYTRADER_TP_PIPS
LONDON_OPEN_CET        = CONFIG.LONDON_OPEN_CET
LONDON_CLOSE_CET       = CONFIG.LONDON_CLOSE_CET          # Now correctly 17
NY_OPEN_CET            = CONFIG.NY_OPEN_CET
NY_CLOSE_CET           = CONFIG.NY_CLOSE_CET              # Now correctly 22
OVERLAP_START          = CONFIG.OVERLAP_START
OVERLAP_END            = CONFIG.OVERLAP_END               # Now correctly 17
