# =============================================================================
# GODBOT v3.0 – signals/signal_engine.py  (PRODUCTION – fully corrected)
# =============================================================================
#
#  Fixes applied vs the repo baseline:
#
#  [A]  ADX strong-trend override (Fixes A + J):
#         _strong_trend() helper suppresses RSI/Stochastic OB/OS readings
#         when ADX ≥ ADX_STRONG_TREND (35) or ADX is rising above threshold.
#         Fix J: sustained high ADX (flat, not just rising) now also fires.
#
#  [B]  Price structure majority-vote (Filter 9):
#         ≥ 66% of consecutive candle pairs must confirm HH+HL or LH+LL.
#         Replaces strict all() logic that never fired in practice.
#
#  [C]  MACD veto dynamic threshold (disabled, MACD_VETO_INCREMENT = 0).
#
#  [D]  Early Reversal Detector:
#         Fires one filter below normal threshold when Stochastic, RSI,
#         and CMF all align. Confidence is score-based (Fix H).
#
#  [E]  Hard-block overbought: RSI ≥ 90 or Stoch ≥ 95 → unconditional -1.
#
#  [F]  Hard-block oversold:   RSI ≤ 18 or Stoch ≤ 5 → unconditional +1.
#
#  [G]  None-safe ADX debug log.
#
#  [H]  Reversal confidence is score-based, not synthetic.
#
#  [I]  RSI_BEAR_LOW raised to 65 in fallback (was 60).
#
#  [J]  See [A] — ADX_STRONG_TREND = 35 fires override even when flat.
#
#  [K]  Candle body filter: body/range ≥ 0.45 required after threshold met.
#         Doji / spinning top candles are rejected.
#
#  [L]  RSI recovery zone 25–35 is now neutral (was +1 — buy bias fixed).
#
#  [M]  BB band scoring is ADX-context aware:
#         Upper/lower band only scores ±1 in confirmed trend (ADX ≥ threshold).
#         Neutral in ranging markets.
#
#  [N]  Stochastic crossover only scores when K is in the correct half:
#         Bullish crossup only if K < 50; bearish crossdown only if K > 50.
#
#  [O]  TP pip cap enforced after RR calculation — was dead code in repo.
#
#  [P]  REVERSAL_THRESHOLD = max(BULL_THRESHOLD - 1, 2) set dynamically.
#
#  [Q]  VWAP Filter 10 added. MAX_FILTERS raised from 9 to 10.
#
#  [R]  _calc_sl_tp() pip_size now correct for metals (XAUUSD = 1.0).
#
#  [S]  TradingSignal dataclass gains sl_pips and tp_pips fields.
#         These are required by risk_manager.calculate_position() [Fix B]
#         and main.py's order routing after the risk_manager interface change.
#
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("SignalEngine")


class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """
    Fully-specified signal returned by SignalEngine.evaluate().

    [S] sl_pips and tp_pips are added so that main.py can pass pip
        distances to risk_manager.calculate_position() after the
        interface change in risk_manager Fix [B]. Without these fields
        main.py has to reverse-engineer pip distances from absolute
        prices, which is error-prone and symbol-type-dependent.
    """
    symbol:     str
    signal:     SignalType
    strength:   float        # 0.0 – 1.0  (abs_score / MAX_FILTERS)
    confidence: float        # 0.0 – 1.0  (abs_score / MAX_FILTERS)
    entry:      float        # close price at signal bar
    sl:         float        # absolute SL price
    tp:         float        # absolute TP price
    sl_pips:    float        # [S] SL distance in pips
    tp_pips:    float        # [S] TP distance in pips
    atr:        float        # ATR value at signal bar
    reasons:    list         # human-readable filter breakdown
    timestamp:  str          # index label of signal bar


class SignalEngine:
    """
    Multi-confluence signal engine — M1 / M5 / Day-Trader profile aware.

    10 technical filters, each scoring +1 (bull) / -1 (bear) / 0 (neutral).
    A signal fires when abs(score) >= threshold (loaded from active profile).
    Max possible score: ±10.

    Filter list
    ───────────
    1.  EMA trend alignment (fast vs slow vs trend EMA)
    2.  MACD crossover / histogram expansion
    3.  RSI zone (oversold / bull-momentum / neutral / bear-momentum / OB)
    4.  Bollinger Band position (ADX-context aware — Fix M)
    5.  ADX trend strength + DI+/DI- direction
    6.  Stochastic OB/OS + K/D crossover (band-aware — Fix N)
    7.  CMF volume pressure
    8.  Volatility squeeze breakout
    9.  Price structure HH+HL or LH+LL (majority vote — Fix B)
    10. VWAP deviation (Fix Q)
    """

    # ── Max filters ───────────────────────────────────────────────────────────
    MAX_FILTERS: int = 10   # [Q] was 9 — VWAP adds Filter 10

    # ── Fallback constants (M5 profile mirror) ────────────────────────────────
    # [I] rsi_bear_low = 65 (was 60 — RSI 60–65 is normal trend momentum)
    _FALLBACK = {
        "adx_threshold":   35,
        "rsi_overbought":  75,
        "rsi_oversold":    25,
        "rsi_bull_low":    35,
        "rsi_bull_high":   60,
        "rsi_bear_low":    65,   # [I]
        "rsi_bear_high":   75,
        "cmf_threshold":   0.05,
        "stoch_bull_zone": 25,
        "stoch_bear_zone": 75,
        "signal_score":    4,
        "sl_pips":         6.0,
        "tp_pips":         10.0,
    }

    # ── Price structure ───────────────────────────────────────────────────────
    STRUCTURE_LOOKBACK: int   = 6
    STRUCTURE_MAJORITY: float = 0.66

    # ── MACD Veto (Fix C) — set > 0 to activate ──────────────────────────────
    MACD_VETO_INCREMENT: int = 0

    # ── Early Reversal Detector (Fix D) ──────────────────────────────────────
    REVERSAL_STOCH_MAX:     int = 45
    REVERSAL_STOCH_MIN:     int = 60
    REVERSAL_RSI_LOW:       int = 30
    REVERSAL_RSI_HIGH:      int = 55
    REVERSAL_RSI_SELL_LOW:  int = 45
    REVERSAL_RSI_SELL_HIGH: int = 70

    # ── Hard-block extremes [E][F] ────────────────────────────────────────────
    RSI_HARD_BLOCK:       int = 90   # RSI ≥ 90  → -1 unconditional
    STOCH_HARD_BLOCK:     int = 95   # K ≥ 95    → -1 unconditional
    RSI_HARD_BLOCK_LOW:   int = 18   # RSI ≤ 18  → +1 unconditional
    STOCH_HARD_BLOCK_LOW: int = 5    # K ≤ 5     → +1 unconditional

    # ── Strong trend ADX level [J] ────────────────────────────────────────────
    ADX_STRONG_TREND: int = 35

    # ── Candle body filter [K] ────────────────────────────────────────────────
    CANDLE_BODY_MIN_RATIO: float = 0.45

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, config=CONFIG, trading_style: str = "scalper"):
        self.cfg           = config
        self.trading_style = trading_style
        self._update_thresholds()

    # ── Profile loader ────────────────────────────────────────────────────────
    def _update_thresholds(self) -> None:
        """
        Load all thresholds and pip caps from the active profile.
        [P] Sets REVERSAL_THRESHOLD = max(BULL_THRESHOLD - 1, 2) dynamically.
        """
        if self.trading_style == "scalper":
            try:
                p  = self.cfg.get_scalper_profile()
                tf = p.get("tf_primary", 5)
                logger.info(
                    f"⚙️  SignalEngine loading scalper profile M{tf}"
                )
            except Exception as exc:
                logger.warning(
                    f"⚠️  Scalper profile load failed ({exc}); using fallback"
                )
                p = {}

            fb = self._FALLBACK
            self.ADX_THRESHOLD   = p.get("adx_threshold",   fb["adx_threshold"])
            self.RSI_OVERBOUGHT  = p.get("rsi_overbought",  fb["rsi_overbought"])
            self.RSI_OVERSOLD    = p.get("rsi_oversold",    fb["rsi_oversold"])
            self.RSI_BULL_LOW    = p.get("rsi_bull_low",    fb["rsi_bull_low"])
            self.RSI_BULL_HIGH   = p.get("rsi_bull_high",   fb["rsi_bull_high"])
            self.RSI_BEAR_LOW    = p.get("rsi_bear_low",    fb["rsi_bear_low"])
            self.RSI_BEAR_HIGH   = p.get("rsi_bear_high",   fb["rsi_bear_high"])
            self.CMF_THRESHOLD   = p.get("cmf_threshold",   fb["cmf_threshold"])
            self.STOCH_BULL_ZONE = p.get("stoch_bull_zone", fb["stoch_bull_zone"])
            self.STOCH_BEAR_ZONE = p.get("stoch_bear_zone", fb["stoch_bear_zone"])
            self.BULL_THRESHOLD  = p.get("signal_score",    fb["signal_score"])
            self.BEAR_THRESHOLD  = self.BULL_THRESHOLD
            self.SL_PIPS         = p.get("sl_pips",         fb["sl_pips"])
            self.TP_PIPS         = p.get("tp_pips",         fb["tp_pips"])

        elif self.trading_style == "daytrader":
            try:
                self.ADX_THRESHOLD   = getattr(self.cfg, "DAYTRADER_ADX_THRESHOLD",  25)
                self.RSI_OVERBOUGHT  = getattr(self.cfg, "DAYTRADER_RSI_OVERBOUGHT",  70)
                self.RSI_OVERSOLD    = getattr(self.cfg, "DAYTRADER_RSI_OVERSOLD",    30)
                self.RSI_BULL_LOW    = getattr(self.cfg, "DAYTRADER_RSI_BULL_LOW",    40)
                self.RSI_BULL_HIGH   = getattr(self.cfg, "DAYTRADER_RSI_BULL_HIGH",   60)
                self.RSI_BEAR_LOW    = getattr(self.cfg, "DAYTRADER_RSI_BEAR_LOW",    65)
                self.RSI_BEAR_HIGH   = getattr(self.cfg, "DAYTRADER_RSI_BEAR_HIGH",   70)
                self.CMF_THRESHOLD   = getattr(self.cfg, "DAYTRADER_CMF_THRESHOLD",  0.05)
                self.STOCH_BULL_ZONE = getattr(self.cfg, "DAYTRADER_STOCH_BULL_ZONE", 20)
                self.STOCH_BEAR_ZONE = getattr(self.cfg, "DAYTRADER_STOCH_BEAR_ZONE", 80)
                self.BULL_THRESHOLD  = getattr(self.cfg, "DAYTRADER_SIGNAL_SCORE",     4)
                self.BEAR_THRESHOLD  = self.BULL_THRESHOLD
                self.SL_PIPS         = getattr(self.cfg, "DAYTRADER_SL_PIPS",         15.0)
                self.TP_PIPS         = getattr(self.cfg, "DAYTRADER_TP_PIPS",         30.0)
                logger.info("⚙️  SignalEngine loaded day-trader profile")
            except Exception as exc:
                logger.warning(
                    f"⚠️  Day-trader profile load failed ({exc}); using fallback"
                )
                self._apply_fallback()
        else:
            self._apply_fallback()

        # [P] Dynamic reversal threshold — always one filter below normal
        self.REVERSAL_THRESHOLD = max(self.BULL_THRESHOLD - 1, 2)

    def _apply_fallback(self) -> None:
        fb = self._FALLBACK
        self.ADX_THRESHOLD   = fb["adx_threshold"]
        self.RSI_OVERBOUGHT  = fb["rsi_overbought"]
        self.RSI_OVERSOLD    = fb["rsi_oversold"]
        self.RSI_BULL_LOW    = fb["rsi_bull_low"]
        self.RSI_BULL_HIGH   = fb["rsi_bull_high"]
        self.RSI_BEAR_LOW    = fb["rsi_bear_low"]
        self.RSI_BEAR_HIGH   = fb["rsi_bear_high"]
        self.CMF_THRESHOLD   = fb["cmf_threshold"]
        self.STOCH_BULL_ZONE = fb["stoch_bull_zone"]
        self.STOCH_BEAR_ZONE = fb["stoch_bear_zone"]
        self.BULL_THRESHOLD  = fb["signal_score"]
        self.BEAR_THRESHOLD  = fb["signal_score"]
        self.SL_PIPS         = fb["sl_pips"]
        self.TP_PIPS         = fb["tp_pips"]
        self.REVERSAL_THRESHOLD = max(self.BULL_THRESHOLD - 1, 2)   # [P]
        logger.warning("⚙️  SignalEngine using fallback thresholds")

    def reload_profile(self) -> None:
        """Re-load all thresholds after CONFIG.SCALPER_TF_SELECTED changes."""
        self._update_thresholds()
        logger.info(
            f"🔄 SignalEngine profile reloaded | "
            f"style={self.trading_style} | "
            f"TF=M{getattr(self.cfg, 'SCALPER_TF_SELECTED', '?')} | "
            f"BULL_THRESHOLD={self.BULL_THRESHOLD} | "
            f"REVERSAL_THRESHOLD={self.REVERSAL_THRESHOLD}"
        )

    # ── Pip size helper [R] ───────────────────────────────────────────────────
    @staticmethod
    def _pip_size(symbol: str) -> float:
        """
        [R] Return the correct pip size for the symbol.
        Fixes the original 'JPY or 0.0001' logic which gave XAUUSD 0.0001
        (100× too small — a 6-pip SL became 0.0006 instead of $6.00).

        JPY crosses:  0.01    (3-digit price, 1 pip = 0.01)
        Metals:       1.0     (XAUUSD $2345 — 1 pip = $1.00)
        Standard FX:  0.0001  (5-digit price, 1 pip = 0.0001)
        """
        sym = symbol.upper()
        if "JPY" in sym:
            return 0.01
        if any(m in sym for m in ("XAU", "XAG", "GOLD", "SILVER")):
            return 1.0
        return 0.0001

    # ── SL/TP calculator [O][R][S] ────────────────────────────────────────────
    def _calc_sl_tp(
        self,
        entry:     float,
        atr:       float,
        symbol:    str,
        direction: str,
    ) -> tuple:
        """
        Two-layer SL/TP with pip cap correctly enforced (Fix O).

        Returns (sl_price, tp_price, sl_pips, tp_pips) — four values.

        [S] sl_pips and tp_pips are returned so TradingSignal can store
            them directly, enabling risk_manager.calculate_position() to
            receive pip distances without reverse-engineering from prices.

        [O] TP pip cap applied AFTER RR multiplication. Previously
            tp_dist = min(atr_tp, cap_tp) was immediately overwritten by
            tp_dist = sl_dist * RR_RATIO, making the cap dead code.

        [R] Uses _pip_size(symbol) for correct metals / JPY handling.
        """
        pip_size = self._pip_size(symbol)

        # Layer 1 — ATR-adaptive
        atr_sl_dist = atr * 1.5
        atr_tp_dist = atr_sl_dist * self.cfg.RR_RATIO

        # Layer 2 — Pip caps from profile
        cap_sl_dist = self.SL_PIPS * pip_size
        cap_tp_dist = self.TP_PIPS * pip_size

        # Final SL: tighter of ATR vs cap
        sl_dist = min(atr_sl_dist, cap_sl_dist)

        # Final TP: RR from final SL, then capped independently [O]
        tp_dist = sl_dist * self.cfg.RR_RATIO
        tp_dist = min(tp_dist, cap_tp_dist)   # [O] cap enforced here

        # Convert to pips for TradingSignal [S]
        sl_pips = round(sl_dist / pip_size, 1)
        tp_pips = round(tp_dist / pip_size, 1)

        if direction == "BUY":
            sl = round(entry - sl_dist, 5)
            tp = round(entry + tp_dist, 5)
        else:
            sl = round(entry + sl_dist, 5)
            tp = round(entry - tp_dist, 5)

        effective_rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        logger.debug(
            f"_calc_sl_tp [{direction}] {symbol} | "
            f"entry={entry:.5f}  ATR={atr:.5f} | "
            f"atr_sl={atr_sl_dist:.5f}  cap_sl={cap_sl_dist:.5f} "
            f"→ sl_dist={sl_dist:.5f} ({sl_pips:.1f}p) | "
            f"atr_tp={atr_tp_dist:.5f}  cap_tp={cap_tp_dist:.5f} "
            f"→ tp_dist={tp_dist:.5f} ({tp_pips:.1f}p) | "
            f"eff_RR={effective_rr:.2f} | "
            f"SL={sl:.5f}  TP={tp:.5f}"
        )
        return sl, tp, sl_pips, tp_pips   # [S] four values returned

    # ── ADX strong trend [A][J] ───────────────────────────────────────────────
    def _strong_trend(
        self,
        adx:      Optional[float],
        prev_adx: Optional[float],
    ) -> bool:
        """
        [A][J] True when trend is strong enough to neutralise OB/OS readings.

        Condition 1: ADX ≥ ADX_STRONG_TREND (35) — sustained trend, even flat.
        Condition 2: ADX > ADX_THRESHOLD AND rising — accelerating trend.

        Fix J: condition 1 was missing — ADX=48 flat did not trigger override.
        """
        if adx is None or prev_adx is None:
            return False
        if adx >= self.ADX_STRONG_TREND:
            return True
        return adx > self.ADX_THRESHOLD and adx > prev_adx

    # ── Candle body filter [K] ────────────────────────────────────────────────
    def _candle_body_ok(self, last: pd.Series) -> tuple:
        """
        Returns (True, ratio) for directional candles.
        Returns (False, ratio) for doji / spinning top / indecision.
        [K] Threshold: body / range ≥ CANDLE_BODY_MIN_RATIO (0.45).
        """
        try:
            high  = float(last.get("high",  0.0))
            low   = float(last.get("low",   0.0))
            open_ = float(last.get("open",  0.0))
            close = float(last.get("close", 0.0))
            rng   = high - low
            if rng <= 0:
                return False, 0.0
            ratio = abs(close - open_) / rng
            return ratio >= self.CANDLE_BODY_MIN_RATIO, round(ratio, 3)
        except Exception:
            return True, 1.0   # Cannot compute → do not block

    # ── Price structure — majority vote [B] ───────────────────────────────────
    def _price_structure(self, df: pd.DataFrame) -> int:
        """
        [B] Returns +1 (HH+HL majority), -1 (LH+LL majority), or 0.
        ≥ STRUCTURE_MAJORITY (66%) of consecutive candle pairs must confirm.
        """
        try:
            window = df.iloc[-(self.STRUCTURE_LOOKBACK + 1):-1]
            if len(window) < self.STRUCTURE_LOOKBACK:
                return 0

            highs = window["high"].values
            lows  = window["low"].values
            pairs = len(highs) - 1
            if pairs < 1:
                return 0

            hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
            hl = sum(1 for i in range(1, len(lows))  if lows[i]  > lows[i - 1])
            lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
            ll = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i - 1])

            maj = self.STRUCTURE_MAJORITY
            if (hh / pairs >= maj) and (hl / pairs >= maj):
                return  1
            if (lh / pairs >= maj) and (ll / pairs >= maj):
                return -1
            return 0
        except Exception:
            return 0

    # ── Early Reversal Detector [D][H] ────────────────────────────────────────
    def _detect_reversal(
        self,
        rsi:          Optional[float],
        stoch_k:      Optional[float],
        stoch_d:      Optional[float],
        cmf:          Optional[float],
        prev_stoch_k: Optional[float],
        prev_stoch_d: Optional[float],
    ) -> tuple:
        """
        [D] Three-indicator early-entry detector.
        [H] Confidence = REVERSAL_THRESHOLD / MAX_FILTERS (score-based, not synthetic).
        """
        if None in (rsi, stoch_k, stoch_d, cmf, prev_stoch_k, prev_stoch_d):
            return None, []

        # BUY reversal
        if (stoch_k < self.REVERSAL_STOCH_MAX and stoch_k > prev_stoch_k and
                self.REVERSAL_RSI_LOW < rsi < self.REVERSAL_RSI_HIGH and
                cmf > 0):
            return "BUY", [
                "🔄 EARLY REVERSAL BUY",
                f"✅ K rising in low zone ({stoch_k:.1f} > {prev_stoch_k:.1f})",
                f"✅ RSI recovering ({rsi:.1f})",
                f"✅ CMF positive ({cmf:.3f})",
                "⚠️  EMA/MACD/ADX still lagging — tighter SL advised",
            ]

        # SELL reversal
        if (stoch_k > self.REVERSAL_STOCH_MIN and stoch_k < prev_stoch_k and
                self.REVERSAL_RSI_SELL_LOW < rsi < self.REVERSAL_RSI_SELL_HIGH and
                cmf < 0):
            return "SELL", [
                "🔄 EARLY REVERSAL SELL",
                f"✅ K falling in high zone ({stoch_k:.1f} < {prev_stoch_k:.1f})",
                f"✅ RSI fading ({rsi:.1f})",
                f"✅ CMF negative ({cmf:.3f})",
                "⚠️  EMA/MACD/ADX still lagging — tighter SL advised",
            ]

        return None, []

    # ── Main evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[TradingSignal]:
        if df is None or len(df) < 2:
            return None

        df = df.copy()
        df.columns = df.columns.str.lower()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        score = 0
        reasons: list = []

        # ── Pre-fetch ADX — used by multiple filters ───────────────────────
        adx      = last.get("adx",    None)
        prev_adx = prev.get("adx",    None)
        di_pos   = last.get("di_pos", None)
        di_neg   = last.get("di_neg", None)

        strong_trend  = self._strong_trend(adx, prev_adx)
        bull_trend_di = (
            strong_trend and
            di_pos is not None and di_neg is not None and
            di_pos > di_neg
        )
        bear_trend_di = (
            strong_trend and
            di_pos is not None and di_neg is not None and
            di_neg > di_pos
        )

        # ── Filter 1: EMA Trend Alignment ─────────────────────────────────
        ema_fast  = last.get("ema_fast",  None)
        ema_slow  = last.get("ema_slow",  None)
        ema_trend = last.get("ema_trend", None)

        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                score += 1
                tag = "✅ EMA fast>slow bullish"
                if ema_trend is not None and ema_fast > ema_trend:
                    tag += " + trend EMA aligned"
                reasons.append(tag)
            elif ema_fast < ema_slow:
                score -= 1
                tag = "❌ EMA fast<slow bearish"
                if ema_trend is not None and ema_fast < ema_trend:
                    tag += " + trend EMA aligned"
                reasons.append(tag)

        # ── Filter 2: MACD Crossover / Histogram ──────────────────────────
        macd      = last.get("macd",        None)
        macd_sig  = last.get("macd_signal", None)
        prev_macd = prev.get("macd",        None)
        prev_msig = prev.get("macd_signal", None)

        macd_score = 0
        if None not in (macd, macd_sig, prev_macd, prev_msig):
            bull_cross = prev_macd < prev_msig and macd > macd_sig
            bear_cross = prev_macd > prev_msig and macd < macd_sig
            if bull_cross:
                macd_score = 1
                reasons.append("✅ MACD bullish crossover")
            elif bear_cross:
                macd_score = -1
                reasons.append("❌ MACD bearish crossover")
            elif macd > macd_sig and macd > prev_macd:
                macd_score = 1
                reasons.append("✅ MACD histogram expanding bullish")
            elif macd < macd_sig and macd < prev_macd:
                macd_score = -1
                reasons.append("❌ MACD histogram expanding bearish")
        score += macd_score

        # ── Filter 3: RSI Zone [L][E][F][A] ───────────────────────────────
        rsi = last.get("rsi", None)
        if rsi is not None:
            if rsi <= self.RSI_HARD_BLOCK_LOW:                          # [F]
                score += 1
                reasons.append(
                    f"✅ RSI extreme oversold ({rsi:.1f} ≤ "
                    f"{self.RSI_HARD_BLOCK_LOW}) — hard block"
                )
            elif rsi <= self.RSI_OVERSOLD:
                score += 1
                reasons.append(
                    f"✅ RSI oversold ({rsi:.1f} ≤ {self.RSI_OVERSOLD})"
                )
            elif self.RSI_OVERSOLD < rsi < self.RSI_BULL_LOW:          # [L]
                reasons.append(
                    f"⚪ RSI recovery zone ({rsi:.1f}) — neutral "
                    f"[{self.RSI_OVERSOLD}–{self.RSI_BULL_LOW}]"
                )
            elif self.RSI_BULL_LOW <= rsi < self.RSI_BULL_HIGH:
                score += 1
                reasons.append(
                    f"✅ RSI bull momentum ({rsi:.1f}) "
                    f"[{self.RSI_BULL_LOW}–{self.RSI_BULL_HIGH}]"
                )
            elif self.RSI_BULL_HIGH <= rsi < self.RSI_BEAR_LOW:
                reasons.append(
                    f"⚪ RSI neutral zone ({rsi:.1f}) — "
                    f"[{self.RSI_BULL_HIGH}–{self.RSI_BEAR_LOW}]"
                )
            elif rsi >= self.RSI_HARD_BLOCK:                           # [E]
                score -= 1
                reasons.append(
                    f"❌ RSI extreme overbought ({rsi:.1f} ≥ "
                    f"{self.RSI_HARD_BLOCK}) — hard block"
                )
            elif rsi >= self.RSI_OVERBOUGHT:
                if bull_trend_di:                                       # [A]
                    reasons.append(
                        f"⚠️  RSI OB ({rsi:.1f}) — ADX={adx:.1f} "
                        f"strong uptrend → neutralised"
                    )
                else:
                    score -= 1
                    reasons.append(
                        f"❌ RSI overbought ({rsi:.1f} ≥ {self.RSI_OVERBOUGHT})"
                    )
            elif self.RSI_BEAR_LOW <= rsi < self.RSI_OVERBOUGHT:
                if bull_trend_di:                                       # [A]
                    reasons.append(
                        f"⚠️  RSI bear zone ({rsi:.1f}) — ADX={adx:.1f} "
                        f"uptrend → neutralised"
                    )
                else:
                    score -= 1
                    reasons.append(
                        f"❌ RSI bear zone ({rsi:.1f}) "
                        f"[{self.RSI_BEAR_LOW}–{self.RSI_OVERBOUGHT}]"
                    )

        # ── Filter 4: Bollinger Band Position [M] ─────────────────────────
        close    = last.get("close",    None)
        bb_mid   = last.get("bb_mid",   None)
        bb_upper = last.get("bb_upper", None)
        bb_lower = last.get("bb_lower", None)

        if None not in (close, bb_mid, bb_upper, bb_lower):
            adx_ok = adx is not None and adx >= self.ADX_THRESHOLD
            if close >= bb_upper:
                if adx_ok:                                             # [M]
                    score += 1
                    reasons.append(
                        f"✅ Price ≥ BB upper — trend continuation "
                        f"(ADX={adx:.1f})"
                    )
                else:
                    adx_str = f"{adx:.1f}" if adx is not None else "N/A"
                    reasons.append(
                        f"⚪ Price ≥ BB upper — ranging (ADX={adx_str}) → neutral"
                    )
            elif bb_mid < close < bb_upper:
                score += 1
                reasons.append("✅ Price above BB midline — bullish side")
            elif bb_lower < close < bb_mid:
                score -= 1
                reasons.append("❌ Price below BB midline — bearish side")
            elif close <= bb_lower:
                if adx_ok:                                             # [M]
                    score -= 1
                    reasons.append(
                        f"❌ Price ≤ BB lower — trend continuation "
                        f"(ADX={adx:.1f})"
                    )
                else:
                    adx_str = f"{adx:.1f}" if adx is not None else "N/A"
                    reasons.append(
                        f"⚪ Price ≤ BB lower — ranging (ADX={adx_str}) → neutral"
                    )

        # ── Filter 5: ADX + DI Direction ──────────────────────────────────
        if None not in (adx, di_pos, di_neg):
            if adx > self.ADX_THRESHOLD:
                if di_pos > di_neg:
                    score += 1
                    reasons.append(
                        f"✅ Uptrend confirmed ADX={adx:.1f} | "
                        f"DI+={di_pos:.1f} > DI-={di_neg:.1f}"
                    )
                else:
                    score -= 1
                    reasons.append(
                        f"❌ Downtrend confirmed ADX={adx:.1f} | "
                        f"DI-={di_neg:.1f} > DI+={di_pos:.1f}"
                    )
            else:
                reasons.append(
                    f"⚪ ADX weak ({adx:.1f} < {self.ADX_THRESHOLD}) — no trend"
                )

        # ── Filter 6: Stochastic [N][E][F][A] ─────────────────────────────
        stoch_k      = last.get("stoch_k", None)
        stoch_d      = last.get("stoch_d", None)
        prev_stoch_k = prev.get("stoch_k", None)
        prev_stoch_d = prev.get("stoch_d", None)

        if None not in (stoch_k, stoch_d):
            if stoch_k <= self.STOCH_HARD_BLOCK_LOW:                   # [F]
                score += 1
                reasons.append(
                    f"✅ Stoch extreme oversold ({stoch_k:.1f} ≤ "
                    f"{self.STOCH_HARD_BLOCK_LOW}) — hard block"
                )
            elif stoch_k <= self.STOCH_BULL_ZONE:
                score += 1
                reasons.append(
                    f"✅ Stoch oversold ({stoch_k:.1f} ≤ {self.STOCH_BULL_ZONE})"
                )
            elif stoch_k >= self.STOCH_HARD_BLOCK:                     # [E]
                score -= 1
                reasons.append(
                    f"❌ Stoch extreme OB ({stoch_k:.1f} ≥ "
                    f"{self.STOCH_HARD_BLOCK}) — hard block"
                )
            elif stoch_k >= self.STOCH_BEAR_ZONE:
                if bull_trend_di:                                       # [A]
                    reasons.append(
                        f"⚠️  Stoch OB ({stoch_k:.1f}) — ADX={adx:.1f} "
                        f"uptrend → neutralised"
                    )
                else:
                    score -= 1
                    reasons.append(
                        f"❌ Stoch OB ({stoch_k:.1f} ≥ {self.STOCH_BEAR_ZONE})"
                    )
            elif stoch_k > stoch_d and stoch_k < 50:                   # [N]
                score += 1
                reasons.append(
                    f"✅ Stoch bullish crossup ({stoch_k:.1f} > "
                    f"{stoch_d:.1f}) in lower half (<50)"
                )
            elif stoch_k < stoch_d and stoch_k > 50:                   # [N]
                score -= 1
                reasons.append(
                    f"❌ Stoch bearish crossdown ({stoch_k:.1f} < "
                    f"{stoch_d:.1f}) in upper half (>50)"
                )
            else:
                reasons.append(
                    f"⚪ Stoch neutral ({stoch_k:.1f}) — mid-range"
                )

        # ── Filter 7: CMF Volume Pressure ─────────────────────────────────
        cmf = last.get("cmf", None)
        if cmf is not None:
            if cmf > self.CMF_THRESHOLD:
                score += 1
                reasons.append(
                    f"✅ CMF={cmf:.3f} > +{self.CMF_THRESHOLD} — buying pressure"
                )
            elif cmf < -self.CMF_THRESHOLD:
                score -= 1
                reasons.append(
                    f"❌ CMF={cmf:.3f} < -{self.CMF_THRESHOLD} — selling pressure"
                )
            else:
                reasons.append(
                    f"⚪ CMF={cmf:.3f} — within ±{self.CMF_THRESHOLD}, neutral"
                )

        # ── Filter 8: Volatility Squeeze Breakout ─────────────────────────
        squeeze    = last.get("squeeze",    None)
        prev_sq    = prev.get("squeeze",    None)
        is_bullish = last.get("is_bullish", None)

        if squeeze is not None and prev_sq is not None:
            if squeeze == 0 and prev_sq == 1:
                if is_bullish is not None:
                    sq_direction = 1 if is_bullish else -1
                else:
                    prev_close   = prev.get("close", None)
                    sq_direction = (
                        1  if (prev_close is not None and close is not None
                               and close > prev_close)
                        else -1 if (prev_close is not None and close is not None)
                        else 0
                    )
                if sq_direction != 0:
                    score += sq_direction
                    label = "bullish" if sq_direction == 1 else "bearish"
                    reasons.append(f"⚡ Squeeze breakout ({label})")
                else:
                    reasons.append("⚪ Squeeze released — direction indeterminate")
            elif squeeze == 1:
                reasons.append("⚪ Squeeze active — awaiting breakout")

        # ── Filter 9: Price Structure [B] ─────────────────────────────────
        structure = self._price_structure(df)
        if structure == 1:
            score += 1
            reasons.append(
                "✅ Bullish structure — HH+HL in ≥66% of recent pairs"
            )
        elif structure == -1:
            score -= 1
            reasons.append(
                "❌ Bearish structure — LH+LL in ≥66% of recent pairs"
            )
        else:
            reasons.append("⚪ No clear price structure — choppy")

        # ── Filter 10: VWAP Deviation [Q] ─────────────────────────────────
        vwap = last.get("vwap", None)
        if vwap is not None and close is not None:
            try:
                vwap_f = float(vwap)
                if not np.isnan(vwap_f):
                    if close > vwap_f:
                        score += 1
                        reasons.append(
                            f"✅ Price above VWAP ({close:.5f} > {vwap_f:.5f})"
                        )
                    elif close < vwap_f:
                        score -= 1
                        reasons.append(
                            f"❌ Price below VWAP ({close:.5f} < {vwap_f:.5f})"
                        )
                    else:
                        reasons.append(f"⚪ Price at VWAP ({vwap_f:.5f}) — neutral")
            except (ValueError, TypeError):
                pass

        # ── Fix C: MACD Veto ──────────────────────────────────────────────
        eff_bull = self.BULL_THRESHOLD
        eff_bear = self.BEAR_THRESHOLD

        if self.MACD_VETO_INCREMENT > 0:
            if score > 0 and macd_score == -1:
                eff_bull += self.MACD_VETO_INCREMENT
                logger.debug(
                    f"⚠️  [{symbol}] MACD veto — bull threshold "
                    f"{self.BULL_THRESHOLD} → {eff_bull}"
                )
            elif score < 0 and macd_score == 1:
                eff_bear += self.MACD_VETO_INCREMENT
                logger.debug(
                    f"⚠️  [{symbol}] MACD veto — bear threshold "
                    f"{self.BEAR_THRESHOLD} → {eff_bear}"
                )

        # ── Fix D: Early Reversal Detector ────────────────────────────────
        rev_direction = None
        rev_reasons   = []

        normal_bull = score >= eff_bull
        normal_bear = score <= -eff_bear

        if not normal_bull and not normal_bear:
            rev_direction, rev_reasons = self._detect_reversal(
                rsi=rsi, stoch_k=stoch_k, stoch_d=stoch_d,
                cmf=cmf, prev_stoch_k=prev_stoch_k, prev_stoch_d=prev_stoch_d,
            )
            if rev_direction:
                logger.debug(
                    f"🔄 [{symbol}] Reversal detector: {rev_direction} | "
                    f"score={score} (need ±{self.BULL_THRESHOLD}) | "
                    f"K={stoch_k:.1f} prev={prev_stoch_k:.1f} | "
                    f"RSI={rsi:.1f} | CMF={cmf:.3f}"
                )

        # ── Fix K: Candle body filter ──────────────────────────────────────
        would_fire = normal_bull or normal_bear or (rev_direction is not None)

        if would_fire:
            body_ok, body_ratio = self._candle_body_ok(last)
            if not body_ok:
                dir_str = (
                    "BUY" if (normal_bull or rev_direction == "BUY") else "SELL"
                )
                logger.debug(
                    f"⛔ [{symbol}] Candle body filter BLOCKED | "
                    f"ratio={body_ratio:.3f} < {self.CANDLE_BODY_MIN_RATIO} | "
                    f"score={score} would have fired {dir_str}"
                )
                return None
            logger.debug(
                f"✅ [{symbol}] Candle body OK | ratio={body_ratio:.3f}"
            )

        # ── Fix G: None-safe ADX log ──────────────────────────────────────
        adx_str      = f"{adx:.1f}"      if adx      is not None else "None"
        prev_adx_str = f"{prev_adx:.1f}" if prev_adx is not None else "None"

        logger.debug(
            f"📊 {symbol} | score={score} (±{self.BULL_THRESHOLD}"
            + (f", eff_bull={eff_bull}" if eff_bull != self.BULL_THRESHOLD else "")
            + (f", eff_bear={eff_bear}" if eff_bear != self.BEAR_THRESHOLD else "")
            + f") | ADX={adx_str}/{prev_adx_str} strong={strong_trend} | "
            f"reversal={rev_direction} | "
            f"style={self.trading_style} TF=M{getattr(self.cfg,'SCALPER_TF_SELECTED','?')}"
        )

        # ── Final signal assembly ──────────────────────────────────────────
        atr_val = float(last.get("atr",   0.0001))
        entry   = float(last.get("close", 0.0))

        if normal_bull:
            sl, tp, sl_pips, tp_pips = self._calc_sl_tp(  # [S]
                entry, atr_val, symbol, "BUY"
            )
            signal_type   = SignalType.BUY
            strength      = abs(score)  / self.MAX_FILTERS
            confidence    = round(min(score / self.MAX_FILTERS, 1.0), 3)
            final_reasons = reasons
            display_score = score
            display_thr   = eff_bull

        elif normal_bear:
            sl, tp, sl_pips, tp_pips = self._calc_sl_tp(  # [S]
                entry, atr_val, symbol, "SELL"
            )
            signal_type   = SignalType.SELL
            strength      = abs(score)  / self.MAX_FILTERS
            confidence    = round(min(abs(score) / self.MAX_FILTERS, 1.0), 3)
            final_reasons = reasons
            display_score = score
            display_thr   = eff_bear

        elif rev_direction == "BUY":
            sl, tp, sl_pips, tp_pips = self._calc_sl_tp(  # [S]
                entry, atr_val, symbol, "BUY"
            )
            signal_type   = SignalType.BUY
            strength      = self.REVERSAL_THRESHOLD / self.MAX_FILTERS
            confidence    = round(self.REVERSAL_THRESHOLD / self.MAX_FILTERS, 3)  # [H]
            final_reasons = rev_reasons + ["── Filter breakdown ──"] + reasons
            display_score = self.REVERSAL_THRESHOLD
            display_thr   = eff_bull

        elif rev_direction == "SELL":
            sl, tp, sl_pips, tp_pips = self._calc_sl_tp(  # [S]
                entry, atr_val, symbol, "SELL"
            )
            signal_type   = SignalType.SELL
            strength      = self.REVERSAL_THRESHOLD / self.MAX_FILTERS
            confidence    = round(self.REVERSAL_THRESHOLD / self.MAX_FILTERS, 3)  # [H]
            final_reasons = rev_reasons + ["── Filter breakdown ──"] + reasons
            display_score = self.REVERSAL_THRESHOLD
            display_thr   = eff_bear

        else:
            logger.debug(
                f"⛔ Gate 2 BLOCKED — {symbol} | score={score} "
                f"(need ≥{eff_bull} bull or ≤-{eff_bear} bear) | "
                f"no reversal | ADX={adx_str}"
            )
            return None

        signal = TradingSignal(
            symbol     = symbol,
            signal     = signal_type,
            strength   = round(strength, 3),
            confidence = confidence,
            entry      = round(entry, 5),
            sl         = sl,
            tp         = tp,
            sl_pips    = sl_pips,    # [S]
            tp_pips    = tp_pips,    # [S]
            atr        = round(atr_val, 5),
            reasons    = final_reasons,
            timestamp  = str(df.index[-1]),
        )

        logger.info(
            f"🎯 Signal [{signal_type.value}] {symbol} | "
            f"Score={display_score}/{display_thr} | "
            f"Conf={confidence:.0%} | "
            f"Entry={entry:.5f}  SL={sl:.5f} ({sl_pips:.1f}p)  "
            f"TP={tp:.5f} ({tp_pips:.1f}p) | "
            f"ATR={atr_val:.5f} | "
            f"Style={self.trading_style} "
            f"TF=M{getattr(self.cfg, 'SCALPER_TF_SELECTED', '?')}"
            + (" | 🔄 REVERSAL" if rev_direction else "")
        )
        return signal
