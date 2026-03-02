import MetaTrader5 as mt5

# Attempt connection
print("Connecting to MT5...")

if not mt5.initialize(
    login=5046802311,
    password="-8LbIzDe",
    server="MetaQuotes-Demo"
):
    print(f"❌ Connection FAILED: {mt5.last_error()}")
else:
    info = mt5.account_info()
    print(f"✅ Connected Successfully!")
    print(f"   Account  : {info.login}")
    print(f"   Balance  : {info.balance} {info.currency}")
    print(f"   Server   : {info.server}")
    print(f"   Leverage : 1:{info.leverage}")

mt5.shutdown()
print("Connection closed.")