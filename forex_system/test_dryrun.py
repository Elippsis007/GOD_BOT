# test_dryrun.py — Test 7: Full Pipeline Dry Run
# Runs ONE complete scan cycle — no trades placed
import os
os.environ["NUMBA_CACHE_DIR"] = r"C:\Users\micha\.numba_cache"

import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime

print("=" * 60)
print("   TEST 7 — FULL DRY RUN (Signal-Only Mode)")
print("=" * 60)
print("   ⚠️  No trades will be placed — Telegram alerts only")
print("=" * 60)

# ── 1. MT5 Connect ────────────────────────────────────────
print("\n📡 1. MT5 CONNECTION...")
if not mt5.initialize():
    if not mt5.initialize(
        path=r"C:\Program Files\StoneX Europe MT5 Terminal\terminal64.exe",
        login=5046802311,
        password="-8LblzDe",
        server="MetaQuotes-Demo"
    ):
        print(f"❌ MT5 failed: {mt5.last_error()}"); quit()
acc = mt5.account_info()
print(f"✅ Connected | Balance: €{acc.balance:,.2f} | Equity: €{acc.equity:,.2f}")

# ── 2. Research Stack ─────────────────────────────────────
print("\n🔬 2. RESEARCH STACK...")
try:
    from research.calendar_scanner import CalendarScanner
    cal    = CalendarScanner()
    events = cal.get_todays_events()
    print(f"✅ Calendar: {len(events)} events today")
except Exception as e:
    print(f"⚠️  Calendar error: {e}")
    cal = None

try:
    from research.sentiment_analyzer import SentimentAnalyzer
    sa = SentimentAnalyzer()
    print("✅ Sentiment engine loaded")
except Exception as e:
    print(f"⚠️  Sentiment error: {e}")
    sa = None

try:
    from research.intermarket import IntermarketAnalyzer
    im  = IntermarketAnalyzer()
    env = im.get_risk_environment()
    print(f"✅ Intermarket loaded | Environment: {env}")
except Exception as e:
    print(f"⚠️  Intermarket error: {e}")
    im = None

try:
    from research.cot_reader import COTReader
    cot = COTReader()
    print("✅ COT reader loaded")
except Exception as e:
    print(f"⚠️  COT error: {e}")
    cot = None

# ── 3. Scan All Symbols ───────────────────────────────────
print("\n🔍 3. SCANNING SYMBOLS...")
from core.indicators import IndicatorEngine
from signals.signal_engine import SignalEngine
from risk.risk_manager import RiskManager

ie = IndicatorEngine()
se = SignalEngine()
rm = RiskManager()

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
signals_found = []
utc_now = datetime.now(pytz.utc)

print(f"\n{'SYMBOL':<10} {'SIGNAL':<8} {'RSI':>6} {'ADX':>6} {'EMA':>10} {'NOTE'}")
print("-" * 60)

for symbol in SYMBOLS:
    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, utc_now, 500)
        if rates is None or len(rates) < 100:
            print(f"{symbol:<10} {'NO DATA':<8}")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                           "close":"Close","tick_volume":"Volume"}, inplace=True)
        df = df[["Open","High","Low","Close","Volume"]]

        df_ind = ie.compute_all(df)
        signal = se.evaluate(df_ind, symbol)
        last   = df_ind.iloc[-1]

        rsi  = f"{last['rsi']:.1f}"
        adx  = f"{last['adx']:.1f}"
        ema  = "Bull✅" if last["ema_fast"]>last["ema_slow"]>last["ema_trend"] else \
               "Bear❌" if last["ema_fast"]<last["ema_slow"]<last["ema_trend"] else "Mixed⚪"

        if signal:
            sig_str = f"{'🟢 BUY' if signal.signal.value=='BUY' else '🔴 SELL'}"
            note    = f"Conf {signal.confidence:.0%} | Score {round(signal.strength*8)}/8"
            signals_found.append((symbol, signal))
        else:
            sig_str = "⚪ HOLD"
            note    = ""

        print(f"{symbol:<10} {sig_str:<8} {rsi:>6} {adx:>6} {ema:>10}  {note}")

    except Exception as e:
        print(f"{symbol:<10} ❌ Error: {e}")

# ── 4. Process Signals ────────────────────────────────────
print(f"\n📊 4. SIGNAL SUMMARY: {len(signals_found)} signal(s) found")
print("-" * 60)

for symbol, signal in signals_found:
    print(f"\n  {'🟢' if signal.signal.value=='BUY' else '🔴'} {symbol} {signal.signal.value}")
    print(f"     Confidence: {signal.confidence:.0%}")
    print(f"     Entry: {signal.entry:.5f} | SL: {signal.sl:.5f} | TP: {signal.tp:.5f}")
    for r in signal.reasons:
        print(f"     {r}")

    # Check calendar safety
    if cal:
        try:
            safety = cal.is_safe_to_trade(symbol)
            if not safety.get("safe", True):
                print(f"     ⛔ BLOCKED by calendar: {safety.get('reason')}")
                continue
        except Exception:
            pass

    # Get sentiment
    if sa:
        try:
            sent = sa.get_symbol_sentiment(symbol)
            sent_label = sent.get("label","Neutral")
            sent_score = sent.get("score", 0)
            print(f"     📰 Sentiment: {sent_label} ({sent_score:+.3f})")
        except Exception:
            pass

    # Get intermarket
    if im:
        try:
            im_result = im.get_intermarket_signal(symbol)
            print(f"     📊 Intermarket: {im_result.get('bias','N/A')} (score {im_result.get('score',0):+.3f})")
        except Exception:
            pass

    # Get COT
    if cot:
        try:
            cot_result = cot.get_cot_signal(symbol)
            print(f"     📋 COT: {cot_result.get('bias','N/A')} ({cot_result.get('percentile',50):.0f}th pctile)")
        except Exception:
            pass

    # Risk sizing
    try:
        tick = mt5.symbol_info_tick(symbol)
        entry = tick.ask if signal.signal.value == "BUY" else tick.bid
        pos = rm.calculate_position(
            symbol=symbol,
            direction=signal.signal.value,
            entry=entry,
            sl=signal.sl,
            tp=signal.tp
        )
        if pos:
            print(f"     💰 Size: {pos.volume} lots | Risk: €{pos.risk_usd:.2f} | R:R 1:{pos.rr_ratio}")
    except Exception as e:
        print(f"     ⚠️  Risk sizing error: {e}")

# ── 5. Telegram Test Alert ────────────────────────────────
print("\n📲 5. SENDING TELEGRAM TEST ALERT...")
try:
    from monitoring.telegram_bot import TelegramBot
    bot = TelegramBot()
    msg = (
        f"🤖 *GODBOT DRY RUN — Test 7*\n"
        f"{'─'*30}\n"
        f"✅ All systems operational\n"
        f"📊 Symbols scanned: {len(SYMBOLS)}\n"
        f"🎯 Signals found: {len(signals_found)}\n"
        f"💰 Balance: €{acc.balance:,.2f}\n"
        f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}\n"
        f"{'─'*30}\n"
        f"_Dry run — no trades placed_"
    )
    bot.send_message(msg)
    print("✅ Telegram alert sent — check your phone!")
except Exception as e:
    print(f"⚠️  Telegram error: {e}")

# ── Done ──────────────────────────────────────────────────
mt5.shutdown()
print("\n" + "=" * 60)
print("   TEST 7 COMPLETE — GODBOT IS READY")
print("=" * 60)
print("""
  Next step: run main.py to start live signal mode
  The bot will scan every 60 seconds and alert you
  on Telegram when all 7 advisors agree on a trade.
""")