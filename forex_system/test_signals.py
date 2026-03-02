# test_signals.py — Test 6: Signal Engine + Risk Manager
import os
os.environ["NUMBA_CACHE_DIR"] = r"C:\Users\micha\.numba_cache"

import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime

print("=" * 60)
print("   TEST 6 — SIGNAL ENGINE & RISK MANAGER")
print("=" * 60)

# ── 1. Connect to MT5 ─────────────────────────────────────
print("\n📡 1. CONNECTING TO MT5...")
if not mt5.initialize():
    if not mt5.initialize(
        path=r"C:\Program Files\StoneX Europe MT5 Terminal\terminal64.exe",
        login=5046802311,
        password="-8LblzDe",
        server="MetaQuotes-Demo"
    ):
        print(f"❌ MT5 connection failed: {mt5.last_error()}")
        quit()
print("✅ MT5 Connected")

# ── 2. Fetch EURUSD Data ──────────────────────────────────
print("\n📊 2. FETCHING EURUSD DATA (500 H1 bars)...")
try:
    utc_now = datetime.now(pytz.utc)
    mt5.symbol_select("EURUSD", True)
    rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_H1, utc_now, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close",
        "tick_volume": "Volume"
    }, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    print(f"✅ {len(df)} bars loaded | Last close: {df['Close'].iloc[-1]:.5f}")
except Exception as e:
    print(f"❌ Data fetch error: {e}")
    mt5.shutdown(); quit()

# ── 3. Compute Indicators ─────────────────────────────────
print("\n📐 3. COMPUTING INDICATORS...")
try:
    from core.indicators import IndicatorEngine
    ie     = IndicatorEngine()
    df_ind = ie.compute_all(df)
    last   = df_ind.iloc[-1]
    print(f"✅ {df_ind.shape[1]} indicator features computed")
    print(f"   EMA9:  {last.get('ema_fast', 0):.5f}")
    print(f"   EMA21: {last.get('ema_slow', 0):.5f}")
    print(f"   RSI:   {last.get('rsi', 0):.1f}")
    print(f"   MACD:  {last.get('macd', 0):.5f}")
    print(f"   ATR:   {last.get('atr', 0):.5f}")
except Exception as e:
    print(f"❌ Indicator error: {e}")
    mt5.shutdown(); quit()

# ── 4. Run Signal Engine ──────────────────────────────────
print("\n🚦 4. RUNNING SIGNAL ENGINE...")
trade_signal = None
try:
    from signals.signal_engine import SignalEngine

    se = SignalEngine()

    # ✅ Correct method name: evaluate(df, symbol)
    # ✅ Correct arg order:   df first, symbol second
    trade_signal = se.evaluate(df_ind, "EURUSD")

    if trade_signal is not None:
        # ✅ TradingSignal is a dataclass — use dot notation
        sig_type   = trade_signal.signal.value        # "BUY" or "SELL"
        score      = round(trade_signal.strength * 8) # reverse-engineer score
        strength   = trade_signal.strength
        confidence = trade_signal.confidence
        entry      = trade_signal.entry
        sl         = trade_signal.sl
        tp         = trade_signal.tp
        reasons    = trade_signal.reasons

        icon = "🟢" if sig_type == "BUY" else "🔴"
        print(f"✅ Signal found!")
        print(f"   {icon} Type:        {sig_type}")
        print(f"   📊 Score:       {score}/8")
        print(f"   💪 Strength:    {strength:.0%}")
        print(f"   🎯 Confidence:  {confidence:.0%}")
        print(f"   📍 Entry:       {entry:.5f}")
        print(f"   🛑 Stop Loss:   {sl:.5f}")
        print(f"   🎯 Take Profit: {tp:.5f}")
        print(f"   📝 Reasons ({len(reasons)}):")
        for r in reasons:
            print(f"      {r}")
    else:
        print("⚪ No signal right now — confluence score below threshold (need ≥4/8)")
        print("   This is normal — the engine only fires on high-confidence setups")

        # Show current score even if no trade
        last = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        print("\n   📊 Current indicator snapshot:")
        print(f"      EMA stack: {'Bullish ✅' if last['ema_fast'] > last['ema_slow'] > last['ema_trend'] else 'Bearish ❌' if last['ema_fast'] < last['ema_slow'] < last['ema_trend'] else 'Mixed ⚪'}")
        print(f"      RSI:       {last['rsi']:.1f}")
        print(f"      ADX:       {last['adx']:.1f} ({'Strong' if last['adx'] > 25 else 'Weak'})")
        macd_cross = "↑ Bullish" if prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'] else \
                     "↓ Bearish" if prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal'] else "No crossover"
        print(f"      MACD:      {macd_cross}")

except Exception as e:
    print(f"❌ Signal engine error: {e}")

# ── 5. Test Risk Manager ──────────────────────────────────
print("\n💰 5. TESTING RISK MANAGER...")
try:
    from risk.risk_manager import RiskManager
    rm = RiskManager()

    tick        = mt5.symbol_info_tick("EURUSD")
    info        = mt5.symbol_info("EURUSD")
    entry_price = tick.ask if tick else 1.18127
    atr_value   = float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else 0.0010

    # Use actual signal levels if we got a signal, otherwise simulate
    if trade_signal is not None:
        sl_price = trade_signal.sl
        tp_price = trade_signal.tp
        direction = trade_signal.signal.value
        print(f"   (Using levels from live signal above)")
    else:
        sl_price  = round(entry_price - (atr_value * 1.5), 5)
        tp_price  = round(entry_price + (atr_value * 3.0), 5)
        direction = "BUY"
        print(f"   (Simulating BUY — no live signal active)")

    # ✅ No 'balance' param — RiskManager reads it from MT5 internally
    result = rm.calculate_position(
        symbol    = "EURUSD",
        direction = direction,
        entry     = entry_price,
        sl        = sl_price,
        tp        = tp_price,
    )

    if result is not None:
        # ✅ PositionSpec is a dataclass — use dot notation
        account  = mt5.account_info()
        balance  = account.balance if account else 1
        risk_pct = (result.risk_usd / balance) * 100
        sl_pips  = round(abs(entry_price - result.sl) / info.point / 10, 1) if info else 0
        tp_pips  = round(abs(result.tp - entry_price) / info.point / 10, 1) if info else 0
        print(f"✅ Risk Manager approved:")
        print(f"   💼 Volume:      {result.volume} lots")
        print(f"   💸 Risk:        €{result.risk_usd:.2f}  ({risk_pct:.1f}% of balance)")
        print(f"   📏 R:R Ratio:   1:{result.rr_ratio}")
        print(f"   📍 Entry:       {result.entry:.5f}")
        print(f"   🛑 SL:          {result.sl:.5f}  ({sl_pips} pips)")
        print(f"   🎯 TP:          {result.tp:.5f}  ({tp_pips} pips)")
    else:
        print("⛔ Risk Manager rejected — check logs above (daily limit / margin / duplicate)")

except Exception as e:
    print(f"❌ Risk manager error: {e}")

# ── Done ──────────────────────────────────────────────────
mt5.shutdown()
print("\n" + "=" * 60)
print("   TEST 6 COMPLETE")
print("=" * 60)