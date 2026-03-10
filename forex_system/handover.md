GODBOT v3.0 — Handover Document
Date: 2026-03-10 | Owner: Michael (Murcia, Spain) | Account: 62111571 PepperstoneUK-Demo | Balance: 180.61 EUR

Project Structure:
C:\Users\micha\Desktop\godbot\forex_system\
├── config/settings.py
├── core/
│   ├── mt5_connector.py
│   ├── data_handler.py
│   └── risk_manager.py
├── indicators/indicators_engine.py
├── signals/signal_engine.py
├── notifications/
│   ├── telegram_alerts.py
│   └── sound_alerts.py
├── monitoring/
│   ├── dashboard.py
│   └── logger.py
├── research/
│   └── calendar_scanner.py
├── reports/
└── main.py
Current Status: WORKING BUT NOT TRADING
The bot starts, connects, scans, computes indicators, and displays the dashboard correctly. It is not generating trades because of one unfixed bug described below.

The One Critical Bug — Fix This First:
File: main.py Method: _process_symbol() Problem: compute_all is called without passing htf_df:

Copy# CURRENT (broken) — produces ADX=None in SignalEngine
ind = self.indicators.compute_all(df)
Copy# CORRECT — fixes ADX=None and restores all scoring
tf_confirm = self._scalper_profile.get("tf_confirm", 15)
htf_df = self.data.get_data(symbol, tf_confirm)
ind = self.indicators.compute_all(df, htf_df=htf_df)
Why this matters: IndicatorEngine.compute_all() signature is (df, for_prediction=False, htf_df=None). Without htf_df, the HTF EMA columns (htf_ema_fast, htf_ema_slow, htf_ema_bull, htf_ema_bear) are all NaN. More critically, ADX also comes back as None inside SignalEngine.evaluate(), causing every filter that touches ADX to score 0. With score=0 on every bar, no signal ever fires.

Verified: Running compute_all(df, htf_df=htf) manually in isolation returns valid ADX values (e.g., 16.6, di_pos=27.5, di_neg=17.0). The signal engine reads them correctly when passed properly.

Second Fix Required — Signal Score Threshold:
File: config/settings.py Field: SCALPER_M5_SIGNAL_SCORE Change:

CopySCALPER_M5_SIGNAL_SCORE: int = 3  # was 4
Why: With 10 filters where ADX, Squeeze, and Price Structure frequently output 0 in ranging conditions, requiring 4/10 means the bot rarely fires even in valid trending setups. Median ADX on EURUSD M5 is ~26, meaning ADX filter contributes 0 on roughly 50% of bars. Threshold of 3 means 3 independent indicators must agree — solid confluence for a scalper.

Flow: settings.py → get_scalper_profile() → profile["signal_score"] → SignalEngine.BULL_THRESHOLD

Confirmed Working Components:
MT5Connector — connects, fetches OHLCV, tick data, positions, account info
DataHandler — fetches and caches M5 and M15 bars correctly
IndicatorEngine — computes all 50+ columns including HTF EMAs when htf_df is passed
SignalEngine — scores correctly when ADX is not None; _update_thresholds() loads from scalper profile
CalendarScanner — is_safe_to_trade(symbol) returns {"safe": bool, "reason": str, "events": list}
TelegramAlerts — connected (@GodBot_alerts_bot), lazy currency fetch fixed
Dashboard — renders live, update_scan_status(str) signature confirmed, get_positions(magic=) confirmed
RiskManager — calculates position size from sl_pips, tp_pips, confidence
ProfileManager — loads/saves JSON trader profiles
Confirmed Method Signatures — Do Not Get Wrong Again:
CopySignalEngine.evaluate(df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]
IndicatorEngine.compute_all(df, for_prediction=False, htf_df=None) -> Optional[pd.DataFrame]
Dashboard.update_scan_status(status: str) -> None
MT5Connector.get_positions(symbol=None, magic=None) -> List[Dict]
CalendarScanner.is_safe_to_trade(symbol, minutes_before=None, minutes_after=15) -> dict
CONFIG Settings That Matter:
CopySCALPER_M5_SIGNAL_SCORE: int = 3        # signal threshold (was 4, too high)
SCALPER_M5_ADX_THRESHOLD: int = 25      # was 35, lowered — median ADX ~26
SCALPER_M5_MAX_SPREAD: float = 1.0      # pips
SCALPER_M5_TP_PIPS: float = 10.0
SCALPER_M5_SL_PIPS: float = 6.0
SCALPER_M5_TRAILING_STOP: int = 5
SCALPER_M5_SCAN_SECS: int = 20
CLOSE_TRADES_EOD: bool = True
EOD_CLOSE_HOUR_UTC: int = 21
ENABLE_QUIET_HOURS: bool = True
QUIET_HOURS_UTC: tuple = (22, 7)
SYMBOLS: List[str] = ["EURUSD"]         # Michael only wants EURUSD
WATCHLIST: List[str] = ["EURUSD"]
Performance So Far (2026-03-10):
3 trades executed, 1 win / 2 losses, P&L -1.70 EUR
Win rate 33.3%, Profit Factor 0.124
Close reasons: Danger exit, Broker closed (positions closed externally)
7-day: 13 trades, 30.8% win rate, -0.22 EUR expectancy per trade
Note: Performance data is from before the htf_df fix — the bot was effectively trading blind without proper indicator data
Known Issues Still Open:
Duplicate compute_all calls — IndicatorEngine is called multiple times per scan cycle (3 calls logged per bar). Needs deduplication in _process_symbol.
FinBERT loads slowly on startup (~8 HTTP requests to HuggingFace). Not harmful but annoying. _setup_logging() silences the noise.
UnicodeEncodeError on Windows cp1252 terminal when log messages contain emoji. Fix by adding encoding="utf-8" to logging.basicConfig or use sys.stdout.reconfigure(encoding="utf-8") at top of main.py.
"Broker closed" exit reason appearing frequently — positions being closed externally by broker (likely margin/demo account behaviour). Not a code bug.
reports/report_2026-03-10.csv shows 3 trades from earlier sessions that were executed before the signal engine was properly fixed.
Michael's Preferences — Important:
No .env files — credentials stored directly in config/settings.py
Only EURUSD — remove all other symbols from SYMBOLS and WATCHLIST
No verbose explanations — give exact file, exact line, exact replacement
No partial code snippets — always provide complete file or complete method
Trading style: Scalper, M5, fully automated
Location: Murcia, Spain (CET = UTC+1)
Broker: Pepperstone UK Demo, MT5
How To Start The Bot:
Copycd C:\Users\micha\Desktop\godbot\forex_system
python main.py
# Select: scalper → fully_automated → M5 → Mike
Next Agent — Do This In Order:
Apply the htf_df fix in main.py → _process_symbol() — this is the only thing blocking trades
Confirm SCALPER_M5_SIGNAL_SCORE = 3 in settings.py
Run python test_signal.py (file exists on Desktop) to verify signal fires before starting full bot
Restart bot during London session (07:00–16:00 UTC) or NY overlap (13:00–17:00 UTC) for best results
Monitor for 🎯 Signal [BUY] lines in logs — should appear within first few scan cycles on a trending market