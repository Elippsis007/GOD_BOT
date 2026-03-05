# test_mt5.py
import MetaTrader5 as mt5

print("Attempting bare initialize...")
if mt5.initialize():
    print("✅ Connected!")
    info = mt5.account_info()
    if info:
        print(f"   Login   : {info.login}")
        print(f"   Server  : {info.server}")
        print(f"   Balance : {info.balance}")
        print(f"   Company : {info.company}")
    term = mt5.terminal_info()
    if term:
        print(f"   Path    : {term.path}")
    mt5.shutdown()
else:
    print(f"❌ Failed: {mt5.last_error()}")
