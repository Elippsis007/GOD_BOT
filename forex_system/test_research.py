# test_research.py
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  🧪 GODBOT RESEARCH STACK TEST")
print("=" * 55)

# ── Test 1: Calendar ──────────────────────────────────────
print("\n1️⃣  ECONOMIC CALENDAR")
print("   Sources: MT5 Built-in + FXStreet RSS")
print("   No scraping — never gets blocked")
print("-" * 55)
from research.calendar_scanner import CalendarScanner
cal = CalendarScanner()
cal.print_todays_events()

for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
    safety = cal.is_safe_to_trade(symbol)
    icon   = "✅" if safety["safe"] else "⛔"
    print(f"  {icon} {symbol:<8} — {safety['reason']}")

# ── Test 2: Sentiment ─────────────────────────────────────
print("\n2️⃣  SENTIMENT ANALYSIS")
print("   Model: FinBERT (financial AI) or VADER backup")
print("   Sources: Central Banks + Reuters + FT")
print("   No Reddit — removed (too noisy)")
print("-" * 55)
from research.sentiment_analyzer import SentimentAnalyzer
sent = SentimentAnalyzer()
sent.print_sentiment_table()

# ── Test 3: Intermarket ───────────────────────────────────
print("\n3️⃣  INTERMARKET ANALYSIS")
print("   Source: Yahoo Finance (yfinance) — free")
print("-" * 55)
from research.intermarket import IntermarketAnalyzer
inter = IntermarketAnalyzer()
inter.print_market_summary()

# ── Test 4: COT ───────────────────────────────────────────
print("\n4️⃣  COT INSTITUTIONAL DATA")
print("   Source: CFTC.gov — US Government, free")
print("-" * 55)
from research.cot_reader import COTReader
cot = COTReader()
cot.print_cot_summary()

print("\n" + "=" * 55)
print("  ✅ Research stack test complete")
print("=" * 55)