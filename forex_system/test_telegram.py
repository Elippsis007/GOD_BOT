# test_telegram.py
import requests
from datetime import datetime
import pytz

TOKEN   = "8693437372:AAHQm4RwedYLkgKkYTlD5XZACHqMnvHVUBE"
CHAT_ID = "7688107635"
TZ      = pytz.timezone("Europe/Madrid")

def send(message: str):
    url  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id":    CHAT_ID,
        "text":       message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=data, timeout=10)
    return response.json()

print("Testing Telegram connection...")

# Test 1 — Basic connection
result = send(
    f"🤖 <b>GODBOT TEST</b>\n"
    f"⏰ {datetime.now(TZ).strftime('%H:%M — %d %b')}\n\n"
    f"✅ Telegram connected successfully!\n"
    f"📱 You will receive all alerts here\n\n"
    f"Alert types coming:\n"
    f"🟢 BUY signals\n"
    f"🔴 SELL signals\n"
    f"⏸️ HOLD alerts\n"
    f"🟡 Potential exits\n"
    f"🚨 Danger exits\n"
    f"📰 News warnings\n"
    f"📊 Daily summaries"
)

if result.get("ok"):
    print("✅ Telegram working — check your phone!")
else:
    print(f"❌ Failed: {result}")

# Test 2 — BUY Signal simulation
print("Sending simulated BUY signal...")
send(
    f"🟢 <b>SCALP BUY — EURUSD</b>\n"
    f"⏰ {datetime.now(TZ).strftime('%H:%M — %d %b')}\n\n"
    f"Entry  : <b>1.08432</b>\n"
    f"SL     : 1.08385 (4.7 pips)\n"
    f"TP     : 1.08505 (7.3 pips)\n"
    f"Spread : 0.8 pips ✅\n"
    f"Conf   : 87%\n\n"
    f"Reasons:\n"
    f"  ✅ EMA bullish stack\n"
    f"  ✅ MACD crossover\n"
    f"  ✅ RSI healthy 54.2\n"
    f"  ✅ Sentiment bullish\n\n"
    f"⚡ <b>Act within 60 seconds</b>\n\n"
    f"<i>This is a test message</i>"
)

# Test 3 — Danger exit simulation
print("Sending simulated DANGER alert...")
send(
    f"🚨 <b>⚠️ DANGER — EURUSD BUY</b>\n"
    f"⏰ {datetime.now(TZ).strftime('%H:%M — %d %b')}\n\n"
    f"Opened  : 1.08432\n"
    f"Current : 1.08390\n"
    f"P&L     : 🔴 -4.2 pips (€-42)\n"
    f"SL      : 1.08385 (0.5 pips away!)\n\n"
    f"Danger signals:\n"
    f"  ❌ Trend reversing\n"
    f"  ❌ SL almost hit\n\n"
    f"⚡ <b>URGENT — Consider closing NOW</b>\n\n"
    f"<i>This is a test message</i>"
)

# Test 4 — News warning simulation
print("Sending simulated NEWS warning...")
send(
    f"📰 <b>⚠️ NEWS WARNING</b>\n"
    f"⏰ {datetime.now(TZ).strftime('%H:%M — %d %b')}\n\n"
    f"Event    : <b>EUR CPI Release</b>\n"
    f"Currency : EUR\n"
    f"Time     : 14:30 your time\n"
    f"In       : 25 minutes\n"
    f"Impact   : 🔴 HIGH\n\n"
    f"Affects  : EURUSD directly\n\n"
    f"⚡ Signals paused until 14:45\n"
    f"💡 Consider closing open trades\n\n"
    f"<i>This is a test message</i>"
)

print("\n✅ All test messages sent!")
print("📱 Check your Telegram now")
print("   You should have 4 messages")