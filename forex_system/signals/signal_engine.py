# signals/signal_engine.py
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
    symbol:     str
    signal:     SignalType
    strength:   float        # 0.0 – 1.0
    confidence: float        # 0.0 – 1.0
    entry:      float
    sl:         float
    tp:         float
    atr:        float
    reasons:    list
    timestamp:  str


class SignalEngine:
    """
    Multi-confluence signal engine with dynamic M1 / M5 profile support.

    Each sub-filter scores +1 (bull) / -1 (bear) / 0 (neutral).
    A signal fires when confluence score >= threshold (profile-dependent).
    9 filters total — max possible score ±9.

    SL / TP calculation (updated):
      - Base distance = ATR × 1.5  (adaptive to current volatility)
      - Hard cap      = profile sl_pips / tp_pips  (from settings.py)
      - Final distance = min(ATR-based, pip cap)
      - This means SL/TP tighten during low volatility and are always
        capped at the configured pip maximums during high volatility.
      - JPY pairs use pip_size = 0.01; all others use 0.0001.
      - RR ratio applied to TP: tp_dist = sl_dist × RR_RATIO.

    Profile is loaded from CONFIG.get_scalper_profile() at init and can be
    reloaded at any time by calling reload_profile().

    Filter list
    -----------
    1. EMA trend alignment (fast vs slow vs trend)
    2. MACD crossover / histogram expansion
    3. RSI zone (oversold / overbought / momentum zone)
    4. Bollinger Band position
    5. ADX trend strength + DI direction
    6. Stochastic oversold / overbought / crossover
    7. CMF volume pressure
    8. Volatility squeeze breakout
    9. Price structure — HH/HL (bull) or LH/LL (bear)
    """

    # ── Fallback constants (used only if profile load fails) ──────────────────
    _FALLBACK = {
        "adx_threshold":   20,
        "rsi_overbought":  75,
        "rsi_oversold":    25,
        "rsi_bull_low":    35,
        "rsi_bull_high":   60,
        "rsi_bear_low":    60,
        "rsi_bear_high":   75,
        "cmf_threshold":   0.05,
        "stoch_bull_zone": 25,
        "stoch_bear_zone": 75,
        "signal_score":    4,
        "sl_pips":         6.0,
        "tp_pips":         12.0,
    }

    STRUCTURE_LOOKBACK = 6   # candles used by Filter 9

    def __init__(self, config=CONFIG, trading_style: str = "scalper"):
        self.cfg           = config
        self.trading_style = trading_style
        self._update_thresholds()

    # ── Profile loader ────────────────────────────────────────────────────────
    def _update_thresholds(self) -> None:
        """
        Load indicator thresholds and SL/TP pip caps from the active profile.

        - scalper   → CONFIG.get_scalper_profile() (M1 or M5)
        - daytrader → CONFIG day-trader fields
        - fallback  → _FALLBACK dict
        """
        if self.trading_style == "scalper":
            try:
                p  = self.cfg.get_scalper_profile()
                tf = p.get("tf_primary", 5)
                logger.info(
                    f"⚙️  SignalEngine loading scalper profile "
                    f"M{tf} from CONFIG.get_scalper_profile()"
                )
            except Exception as exc:
                logger.warning(
                    f"⚠️  Could not load scalper profile ({exc}); "
                    "using fallback thresholds"
                )
                p = {}

            fb = self._FALLBACK
            self.ADX_THRESHOLD     = p.get("adx_threshold",   fb["adx_threshold"])
            self.RSI_OVERBOUGHT    = p.get("rsi_overbought",  fb["rsi_overbought"])
            self.RSI_OVERSOLD      = p.get("rsi_oversold",    fb["rsi_oversold"])
            self.RSI_BULL_LOW      = p.get("rsi_bull_low",    fb["rsi_bull_low"])
            self.RSI_BULL_HIGH     = p.get("rsi_bull_high",   fb["rsi_bull_high"])
            self.RSI_BEAR_LOW      = p.get("rsi_bear_low",    fb["rsi_bear_low"])
            self.RSI_BEAR_HIGH     = p.get("rsi_bear_high",   fb["rsi_bear_high"])
            self.CMF_THRESHOLD     = p.get("cmf_threshold",   fb["cmf_threshold"])
            self.STOCH_BULL_ZONE   = p.get("stoch_bull_zone", fb["stoch_bull_zone"])
            self.STOCH_BEAR_ZONE   = p.get("stoch_bear_zone", fb["stoch_bear_zone"])
            self.BULL_THRESHOLD    = p.get("signal_score",    fb["signal_score"])
            self.BEAR_THRESHOLD    = self.BULL_THRESHOLD
            # ── SL/TP pip caps from profile ───────────────────────────────
            self.SL_PIPS           = p.get("sl_pips",         fb["sl_pips"])
            self.TP_PIPS           = p.get("tp_pips",         fb["tp_pips"])

        elif self.trading_style == "daytrader":
            try:
                self.ADX_THRESHOLD   = getattr(self.cfg, "DAYTRADER_ADX_THRESHOLD",  25)
                self.RSI_OVERBOUGHT  = getattr(self.cfg, "DAYTRADER_RSI_OVERBOUGHT",  70)
                self.RSI_OVERSOLD    = getattr(self.cfg, "DAYTRADER_RSI_OVERSOLD",    30)
                self.RSI_BULL_LOW    = getattr(self.cfg, "DAYTRADER_RSI_BULL_LOW",    40)
                self.RSI_BULL_HIGH   = getattr(self.cfg, "DAYTRADER_RSI_BULL_HIGH",   60)
                self.RSI_BEAR_LOW    = getattr(self.cfg, "DAYTRADER_RSI_BEAR_LOW",    60)
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

    def _apply_fallback(self) -> None:
        fb = self._FALLBACK
        self.ADX_THRESHOLD     = fb["adx_threshold"]
        self.RSI_OVERBOUGHT    = fb["rsi_overbought"]
        self.RSI_OVERSOLD      = fb["rsi_oversold"]
        self.RSI_BULL_LOW      = fb["rsi_bull_low"]
        self.RSI_BULL_HIGH     = fb["rsi_bull_high"]
        self.RSI_BEAR_LOW      = fb["rsi_bear_low"]
        self.RSI_BEAR_HIGH     = fb["rsi_bear_high"]
        self.CMF_THRESHOLD     = fb["cmf_threshold"]
        self.STOCH_BULL_ZONE   = fb["stoch_bull_zone"]
        self.STOCH_BEAR_ZONE   = fb["stoch_bear_zone"]
        self.BULL_THRESHOLD    = fb["signal_score"]
        self.BEAR_THRESHOLD    = fb["signal_score"]
        self.SL_PIPS           = fb["sl_pips"]
        self.TP_PIPS           = fb["tp_pips"]
        logger.warning("⚙️  SignalEngine using fallback thresholds")

    def reload_profile(self) -> None:
        """
        Call this after changing CONFIG.SCALPER_TF_SELECTED at runtime
        so all filter thresholds and SL/TP caps are immediately updated.
        """
        self._update_thresholds()
        logger.info(
            f"🔄 SignalEngine profile reloaded "
            f"(style={self.trading_style}, "
            f"TF={getattr(self.cfg, 'SCALPER_TF_SELECTED', '?')})"
        )

    # ── SL/TP calculator ──────────────────────────────────────────────────────
    def _calc_sl_tp(
        self,
        entry:     float,
        atr:       float,
        symbol:    str,
        direction: str,          # "BUY" or "SELL"
    ) -> tuple:
        """
        Calculate SL and TP with two-layer logic:

        Layer 1 — ATR-based (adaptive):
            sl_dist = ATR × 1.5
            tp_dist = sl_dist × RR_RATIO

        Layer 2 — Pip cap (from active profile):
            sl_dist = min(ATR-based, SL_PIPS × pip_size)
            tp_dist = min(ATR-based TP, TP_PIPS × pip_size)

        The smaller of the two distances is used so the SL/TP tightens
        with low volatility but never exceeds the configured pip maximum.
        TP is then recomputed as sl_dist × RR_RATIO to preserve the
        risk-reward ratio regardless of which layer wins.

        JPY pairs: pip_size = 0.01
        All others: pip_size = 0.0001

        Returns (sl: float, tp: float).
        """
        pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001

        # Layer 1 — ATR-based distances
        atr_sl_dist = atr * 1.5
        atr_tp_dist = atr_sl_dist * self.cfg.RR_RATIO

        # Layer 2 — Pip cap distances
        cap_sl_dist = self.SL_PIPS * pip_size
        cap_tp_dist = self.TP_PIPS * pip_size

        # Final distances — take the tighter of ATR vs pip cap
        sl_dist = min(atr_sl_dist, cap_sl_dist)
        tp_dist = min(atr_tp_dist, cap_tp_dist)

        # Recompute TP from final SL distance to maintain RR ratio
        # (ensures TP is always SL_dist × RR even if ATR cap was applied)
        tp_dist = sl_dist * self.cfg.RR_RATIO

        if direction == "BUY":
            sl = round(entry - sl_dist, 5)
            tp = round(entry + tp_dist, 5)
        else:  # SELL
            sl = round(entry + sl_dist, 5)
            tp = round(entry - tp_dist, 5)

        logger.debug(
            f"_calc_sl_tp [{direction}] {symbol} | "
            f"entry={entry:.5f} ATR={atr:.5f} | "
            f"atr_sl={atr_sl_dist:.5f} cap_sl={cap_sl_dist:.5f} → "
            f"sl_dist={sl_dist:.5f} ({sl_dist/pip_size:.1f} pips) | "
            f"tp_dist={tp_dist:.5f} ({tp_dist/pip_size:.1f} pips) | "
            f"SL={sl:.5f} TP={tp:.5f} | RR={self.cfg.RR_RATIO}"
        )

        return sl, tp

    # ── Filter 9: Price Structure ─────────────────────────────────────────────
    def _price_structure(self, df: pd.DataFrame) -> int:
        """
        Detects HH+HL (bull, +1) or LH+LL (bear, -1) over the last
        STRUCTURE_LOOKBACK candles.  Returns 0 for choppy structure.
        """
        try:
            window = df.iloc[-(self.STRUCTURE_LOOKBACK + 1):-1]
            if len(window) < self.STRUCTURE_LOOKBACK:
                return 0

            highs = window["high"].values
            lows  = window["low"].values

            hh = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
            hl = all(lows[i]  > lows[i - 1]  for i in range(1, len(lows)))
            lh = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
            ll = all(lows[i]  < lows[i - 1]  for i in range(1, len(lows)))

            if hh and hl:
                return  1
            elif lh and ll:
                return -1
            else:
                return  0
        except Exception:
            return 0

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

        # ── Filter 1: Trend Alignment (EMA fast / slow / trend) ───────────────
        ema_fast  = last.get("ema_fast",  None)
        ema_slow  = last.get("ema_slow",  None)
        ema_trend = last.get("ema_trend", None)

        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                score += 1
                tag = "✅ EMA fast>slow bullish"
                if ema_trend is not None and ema_fast > ema_trend:
                    tag += " + trend aligned"
                reasons.append(tag)
            elif ema_fast < ema_slow:
                score -= 1
                tag = "❌ EMA fast<slow bearish"
                if ema_trend is not None and ema_fast < ema_trend:
                    tag += " + trend aligned"
                reasons.append(tag)

        # ── Filter 2: MACD Crossover / Histogram ──────────────────────────────
        macd      = last.get("macd",        None)
        macd_sig  = last.get("macd_signal", None)
        prev_macd = prev.get("macd",        None)
        prev_msig = prev.get("macd_signal", None)

        if None not in (macd, macd_sig, prev_macd, prev_msig):
            bull_cross = prev_macd < prev_msig and macd > macd_sig
            bear_cross = prev_macd > prev_msig and macd < macd_sig
            if bull_cross:
                score += 1
                reasons.append("✅ MACD bullish crossover")
            elif bear_cross:
                score -= 1
                reasons.append("❌ MACD bearish crossover")
            elif macd > macd_sig and macd > prev_macd:
                score += 1
                reasons.append("✅ MACD histogram expanding bullish")
            elif macd < macd_sig and macd < prev_macd:
                score -= 1
                reasons.append("❌ MACD histogram expanding bearish")

        # ── Filter 3: RSI Zone ────────────────────────────────────────────────
        rsi = last.get("rsi", None)
        if rsi is not None:
            if rsi <= self.RSI_OVERSOLD:
                score += 1
                reasons.append(f"✅ RSI oversold ({rsi:.1f} ≤ {self.RSI_OVERSOLD})")
            elif self.RSI_OVERSOLD < rsi < self.RSI_BULL_HIGH:
                score += 1
                reasons.append(f"✅ RSI bullish zone ({rsi:.1f})")
            elif rsi >= self.RSI_OVERBOUGHT:
                score -= 1
                reasons.append(f"❌ RSI overbought ({rsi:.1f} ≥ {self.RSI_OVERBOUGHT})")
            elif self.RSI_BEAR_LOW < rsi < self.RSI_OVERBOUGHT:
                score -= 1
                reasons.append(f"❌ RSI bearish zone ({rsi:.1f})")

        # ── Filter 4: Bollinger Band Position ─────────────────────────────────
        close    = last.get("close",    None)
        bb_mid   = last.get("bb_mid",   None)
        bb_upper = last.get("bb_upper", None)
        bb_lower = last.get("bb_lower", None)

        if None not in (close, bb_mid, bb_upper, bb_lower):
            if bb_mid < close < bb_upper:
                score += 1
                reasons.append("✅ Price above BB midline")
            elif bb_lower < close < bb_mid:
                score -= 1
                reasons.append("❌ Price below BB midline")
            elif close >= bb_upper:
                score += 1
                reasons.append("✅ Price at/above BB upper (strong bull momentum)")
            elif close <= bb_lower:
                score -= 1
                reasons.append("❌ Price at/below BB lower (strong bear momentum)")

        # ── Filter 5: ADX Trend Strength ──────────────────────────────────────
        adx    = last.get("adx",    None)
        di_pos = last.get("di_pos", None)
        di_neg = last.get("di_neg", None)

        if None not in (adx, di_pos, di_neg):
            if adx > self.ADX_THRESHOLD:
                if di_pos > di_neg:
                    score += 1
                    reasons.append(
                        f"✅ Uptrend confirmed ADX={adx:.1f} "
                        f"(threshold {self.ADX_THRESHOLD})"
                    )
                else:
                    score -= 1
                    reasons.append(
                        f"❌ Downtrend confirmed ADX={adx:.1f} "
                        f"(threshold {self.ADX_THRESHOLD})"
                    )
            else:
                reasons.append(
                    f"⚠️  ADX weak ({adx:.1f} < {self.ADX_THRESHOLD}) "
                    "— no trend confirmation"
                )

        # ── Filter 6: Stochastic ──────────────────────────────────────────────
        stoch_k = last.get("stoch_k", None)
        stoch_d = last.get("stoch_d", None)

        if None not in (stoch_k, stoch_d):
            if stoch_k <= self.STOCH_BULL_ZONE:
                score += 1
                reasons.append(
                    f"✅ Stochastic oversold ({stoch_k:.1f} ≤ {self.STOCH_BULL_ZONE})"
                )
            elif stoch_k >= self.STOCH_BEAR_ZONE:
                score -= 1
                reasons.append(
                    f"❌ Stochastic overbought ({stoch_k:.1f} ≥ {self.STOCH_BEAR_ZONE})"
                )
            elif stoch_k > stoch_d and stoch_k < self.STOCH_BEAR_ZONE:
                score += 1
                reasons.append(
                    f"✅ Stochastic bullish crossup ({stoch_k:.1f})"
                )
            elif stoch_k < stoch_d and stoch_k > self.STOCH_BULL_ZONE:
                score -= 1
                reasons.append(
                    f"❌ Stochastic bearish crossdown ({stoch_k:.1f})"
                )

        # ── Filter 7: CMF Volume ──────────────────────────────────────────────
        cmf = last.get("cmf", None)
        if cmf is not None:
            if cmf > self.CMF_THRESHOLD:
                score += 1
                reasons.append(
                    f"✅ Positive CMF={cmf:.3f} "
                    f"(> {self.CMF_THRESHOLD} buying pressure)"
                )
            elif cmf < -self.CMF_THRESHOLD:
                score -= 1
                reasons.append(
                    f"❌ Negative CMF={cmf:.3f} "
                    f"(< -{self.CMF_THRESHOLD} selling pressure)"
                )

        # ── Filter 8: Volatility Squeeze Breakout ─────────────────────────────
        squeeze    = last.get("squeeze",    None)
        prev_sq    = prev.get("squeeze",    None)
        is_bullish = last.get("is_bullish", None)

        if squeeze is not None and prev_sq is not None:
            if squeeze == 0 and prev_sq == 1:
                if is_bullish is not None:
                    direction = 1 if is_bullish else -1
                else:
                    prev_close = prev.get("close", None)
                    direction  = (
                        1 if (prev_close is not None and close is not None
                              and close > prev_close)
                        else -1 if (prev_close is not None and close is not None)
                        else 0
                    )
                if direction != 0:
                    score += direction
                    label = "bullish" if direction == 1 else "bearish"
                    reasons.append(f"⚡ Squeeze breakout ({label})")

        # ── Filter 9: Price Structure (HH/HL or LH/LL) ───────────────────────
        structure = self._price_structure(df)
        if structure == 1:
            score += 1
            reasons.append("✅ Bullish structure (HH + HL confirmed)")
        elif structure == -1:
            score -= 1
            reasons.append("❌ Bearish structure (LH + LL confirmed)")
        else:
            reasons.append("⚠️  No clear price structure")

        # ── Determine Signal ──────────────────────────────────────────────────
        strength = abs(score) / 9.0
        atr      = float(last.get("atr",   0.0001))
        entry    = float(last.get("close", 0.0))

        logger.debug(
            f"📊 {symbol} score={score} "
            f"(threshold ±{self.BULL_THRESHOLD}) | "
            f"style={self.trading_style} | "
            f"TF=M{getattr(self.cfg, 'SCALPER_TF_SELECTED', '?')} | "
            f"SL_PIPS={self.SL_PIPS} TP_PIPS={self.TP_PIPS} | "
            f"filters: {' | '.join(reasons)}"
        )

        if score >= self.BULL_THRESHOLD:
            sl, tp      = self._calc_sl_tp(entry, atr, symbol, "BUY")
            signal_type = SignalType.BUY
            confidence  = round(min(score / 9.0, 1.0), 3)

        elif score <= -self.BEAR_THRESHOLD:
            sl, tp      = self._calc_sl_tp(entry, atr, symbol, "SELL")
            signal_type = SignalType.SELL
            confidence  = round(min(abs(score) / 9.0, 1.0), 3)

        else:
            logger.debug(
                f"⛔ Gate 2 BLOCKED — {symbol} score={score} "
                f"below threshold ±{self.BULL_THRESHOLD}"
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
            atr        = round(atr, 5),
            reasons    = reasons,
            timestamp  = str(df.index[-1]),
        )

        logger.info(
            f"🎯 Signal [{signal_type.value}] on {symbol} | "
            f"Score={score}/{self.BULL_THRESHOLD} | "
            f"Conf={confidence:.0%} | "
            f"Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} | "
            f"Style={self.trading_style} "
            f"TF=M{getattr(self.cfg, 'SCALPER_TF_SELECTED', '?')} | "
            f"SL={self.SL_PIPS}pip cap TP={self.TP_PIPS}pip cap"
        )
        return signal
