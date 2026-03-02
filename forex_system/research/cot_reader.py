# research/cot_reader.py
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from monitoring.logger import get_logger

logger = get_logger("COTReader")

class COTReader:
    """
    Downloads and parses CFTC Commitments of Traders data.
    Source: CFTC.gov — free US government data, updated weekly.
    """

    COT_URL   = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    CACHE_DIR = "data/cot"

    CURRENCY_MAP = {
        "EURUSD": "EURO FX",
        "GBPUSD": "BRITISH POUND",
        "USDJPY": "JAPANESE YEN",
        "AUDUSD": "AUSTRALIAN DOLLAR",
        "USDCAD": "CANADIAN DOLLAR",
        "USDCHF": "SWISS FRANC",
        "NZDUSD": "NZ DOLLAR",
        "XAUUSD": "GOLD",
    }

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._data = None

    # ── Should Update? ────────────────────────────────────
    def should_update(self) -> bool:
        cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
        if not os.path.exists(cache_file):
            return True
        modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - modified > timedelta(days=7)

    # ── Download ──────────────────────────────────────────
    def download_cot_data(self) -> bool:
        try:
            year = datetime.now().year
            url  = self.COT_URL.format(year=year)
            logger.info(f"📥 Downloading COT data from CFTC: {year}")

            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            with zipfile.ZipFile(BytesIO(resp.content)) as z:
                csv_name = [n for n in z.namelist() if n.endswith(".txt")][0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, low_memory=False)

            # Auto-detect and standardise date column
            date_col = self._find_date_column(df)
            if date_col:
                df["report_date"] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                df["report_date"] = pd.NaT

            cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
            df.to_csv(cache_file, index=False)
            self._data = df

            logger.info(f"✅ COT data downloaded: {len(df)} records")
            return True

        except Exception as e:
            logger.error(f"COT download error: {e}")
            return False

    # ── Column Finders ────────────────────────────────────
    def _find_date_column(self, df: pd.DataFrame) -> str:
        candidates = [
            "Report_Date_as_YYYY-MM-DD",
            "Report_Date_as_MM_DD_YYYY",
            "As_of_Date_In_Form_YYMMDD",
            "report_date",
            "Date",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        date_cols = [c for c in df.columns if "date" in c.lower()]
        return date_cols[0] if date_cols else None

    def _find_name_column(self, df: pd.DataFrame) -> str:
        candidates = [
            "Market_and_Exchange_Names",
            "Market and Exchange Names",
            "market_name",
            "Name",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        name_cols = [c for c in df.columns
                     if "market" in c.lower() or "name" in c.lower()]
        return name_cols[0] if name_cols else None

    def _find_column(self, df: pd.DataFrame, candidates: list) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    # ── Load Cache ────────────────────────────────────────
    def _load_data(self) -> bool:
        if self._data is not None:
            return True
        cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
        if not os.path.exists(cache_file):
            return False
        try:
            self._data = pd.read_csv(cache_file, low_memory=False)
            return True
        except Exception as e:
            logger.error(f"COT load error: {e}")
            return False

    # ── Get Signal ────────────────────────────────────────
    def get_cot_signal(self, symbol: str) -> dict:
        default = {
            "bias":         "Neutral",
            "net_position": 0,
            "percentile":   50.0,
            "signal":       "HOLD",
            "note":         "No data"
        }

        market_name = self.CURRENCY_MAP.get(symbol)
        if not market_name:
            return default

        if not self._load_data():
            if self.should_update():
                self.download_cot_data()
            if not self._load_data():
                return default

        try:
            df = self._data.copy()

            # ── Find market name column ───────────────────
            name_col = self._find_name_column(df)
            if not name_col:
                logger.error("COT: Cannot find market name column")
                return default

            # ── Filter to this currency ───────────────────
            mask      = df[name_col].str.upper().str.contains(
                            market_name.upper(), na=False)
            market_df = df[mask].copy()

            if market_df.empty:
                logger.warning(f"COT: No data found for {symbol} ({market_name})")
                return default

            # ── Find long/short columns ───────────────────
            long_col = self._find_column(df, [
                "Lev_Money_Positions_Long_All",   # Financial format ✅
                "NonComm_Positions_Long_All",     # Legacy format
                "Asset_Mgr_Positions_Long_All",   # Alternative
            ])
            short_col = self._find_column(df, [
                "Lev_Money_Positions_Short_All",  # Financial format ✅
                "NonComm_Positions_Short_All",    # Legacy format
                "Asset_Mgr_Positions_Short_All",  # Alternative
            ])

            if not long_col or not short_col:
                logger.error("COT: Cannot find long/short columns")
                return default

            # ── Calculate net position ────────────────────
            market_df = market_df.copy()
            market_df[long_col]  = pd.to_numeric(market_df[long_col],  errors="coerce")
            market_df[short_col] = pd.to_numeric(market_df[short_col], errors="coerce")
            market_df["net"]     = market_df[long_col] - market_df[short_col]
            market_df            = market_df.dropna(subset=["net"])

            if market_df.empty:
                return default

            latest_net = float(market_df["net"].iloc[-1])
            net_series = market_df["net"]

            # ── Percentile rank ───────────────────────────
            pct = float((net_series < latest_net).mean() * 100)

            # ── Bias & signal ─────────────────────────────
            if pct >= 65:
                bias, signal = "Bullish", "BUY"
            elif pct <= 35:
                bias, signal = "Bearish", "SELL"
            else:
                bias, signal = "Neutral", "HOLD"

            return {
                "bias":         bias,
                "net_position": int(latest_net),
                "percentile":   round(pct, 1),
                "signal":       signal,
                "note":         f"{len(market_df)} weeks of data"
            }

        except Exception as e:
            logger.error(f"COT signal error for {symbol}: {e}")
            return default

    # ── Print Summary ─────────────────────────────────────
    def print_cot_summary(self):
        from config.settings import CONFIG

        print("\n📋 COT INSTITUTIONAL POSITIONING")
        print("=" * 55)

        for symbol in CONFIG.SYMBOLS:
            result = self.get_cot_signal(symbol)
            bias   = result["bias"]
            pct    = result["percentile"]
            note   = result["note"]
            icon   = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
            print(f"  {icon} {symbol:<8} {bias:<10} Pctile: {pct:5.1f}% | {note}")

        print("=" * 55)