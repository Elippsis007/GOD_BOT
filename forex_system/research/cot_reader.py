# research/cot_reader.py
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional
from monitoring.logger import logger


class COTReader:
    """
    Downloads and parses CFTC Commitments of Traders data.
    Source: CFTC.gov — free US government data, updated every Tuesday.
    """

    COT_URL   = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    CACHE_DIR = "data/cot"

    UPDATE_THRESHOLD_DAYS = 2

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
        self._data: Optional[pd.DataFrame] = None

    def should_update(self) -> bool:
        cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
        if not os.path.exists(cache_file):
            return True
        modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age = datetime.now() - modified
        return age > timedelta(days=self.UPDATE_THRESHOLD_DAYS)

    def download_cot_data(self) -> bool:
        year = datetime.now().year

        for attempt_year in (year, year - 1):
            url = self.COT_URL.format(year=attempt_year)
            logger.info(f"📥 Downloading COT data from CFTC ({attempt_year})…")
            try:
                resp = requests.get(url, timeout=30)

                if resp.status_code == 404:
                    logger.warning(
                        f"COT {attempt_year} file not found (404) — "
                        f"trying {attempt_year - 1}"
                    )
                    continue

                resp.raise_for_status()

                with zipfile.ZipFile(BytesIO(resp.content)) as z:
                    txt_files = [n for n in z.namelist() if n.endswith(".txt")]
                    if not txt_files:
                        logger.error("COT ZIP contains no .txt files")
                        return False
                    with z.open(txt_files[0]) as f:
                        df = pd.read_csv(f, low_memory=False)

                date_col = self._find_date_column(df)
                if date_col:
                    df["report_date"] = pd.to_datetime(
                        df[date_col], errors="coerce"
                    )
                else:
                    logger.warning(
                        "COT: no date column found — report_date will be NaT"
                    )
                    df["report_date"] = pd.NaT

                if "report_date" in df.columns:
                    df.sort_values("report_date", ascending=True, inplace=True)
                    df.reset_index(drop=True, inplace=True)

                cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
                df.to_csv(cache_file, index=False)
                self._data = df

                logger.info(
                    f"✅ COT data downloaded ({attempt_year}): {len(df)} records"
                )
                return True

            except Exception as e:
                logger.error(f"COT download error ({attempt_year}): {e}")
                continue

        logger.error("COT download failed for both current and previous year")
        return False

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
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

    def _find_name_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = [
            "Market_and_Exchange_Names",
            "Market and Exchange Names",
            "market_name",
            "Name",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        name_cols = [
            c for c in df.columns
            if "market" in c.lower() or "name" in c.lower()
        ]
        return name_cols[0] if name_cols else None

    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _load_data(self) -> bool:
        if self._data is not None:
            return True
        cache_file = os.path.join(self.CACHE_DIR, "cot_latest.csv")
        if not os.path.exists(cache_file):
            return False
        try:
            self._data = pd.read_csv(cache_file, low_memory=False)

            date_col = self._find_date_column(self._data)
            if date_col:
                self._data["report_date"] = pd.to_datetime(
                    self._data[date_col], errors="coerce"
                )
                self._data.sort_values(
                    "report_date", ascending=True, inplace=True
                )
                self._data.reset_index(drop=True, inplace=True)

            logger.debug(f"COT cache loaded: {len(self._data)} records")
            return True

        except Exception as e:
            logger.error(f"COT load error: {e}")
            return False

    def get_cot_signal(self, symbol: str) -> dict:
        default = {
            "bias":         "Neutral",
            "net_position": 0,
            "percentile":   50.0,
            "signal":       "HOLD",
            "note":         "No data",
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

            name_col = self._find_name_column(df)
            if not name_col:
                logger.error("COT: cannot find market name column")
                return default

            mask      = df[name_col].str.upper().str.contains(
                market_name.upper(), na=False
            )
            market_df = df[mask].copy()

            if market_df.empty:
                logger.warning(
                    f"COT: no data found for {symbol} ({market_name})"
                )
                return default

            long_col = self._find_column(market_df, [
                "Lev_Money_Positions_Long_All",
                "NonComm_Positions_Long_All",
                "Asset_Mgr_Positions_Long_All",
            ])
            short_col = self._find_column(market_df, [
                "Lev_Money_Positions_Short_All",
                "NonComm_Positions_Short_All",
                "Asset_Mgr_Positions_Short_All",
            ])

            if not long_col or not short_col:
                logger.error(
                    f"COT: cannot find long/short columns. "
                    f"Available: {list(market_df.columns[:10])}"
                )
                return default

            market_df[long_col]  = pd.to_numeric(
                market_df[long_col],  errors="coerce"
            )
            market_df[short_col] = pd.to_numeric(
                market_df[short_col], errors="coerce"
            )
            market_df["net"] = market_df[long_col] - market_df[short_col]
            market_df        = market_df.dropna(subset=["net"])

            if market_df.empty:
                return default

            latest_net = float(market_df["net"].iloc[-1])
            net_series = market_df["net"]

            historical = net_series.iloc[:-1]
            if len(historical) == 0:
                pct = 50.0
            else:
                pct = float((historical < latest_net).mean() * 100)

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
                "note":         f"{len(market_df)} weeks of data",
            }

        except Exception as e:
            logger.error(f"COT signal error for {symbol}: {e}")
            return default

    def print_cot_summary(self) -> None:
        from config.settings import CONFIG

        print("\n📋 COT INSTITUTIONAL POSITIONING")
        print("=" * 55)
        for symbol in CONFIG.SYMBOLS:
            result = self.get_cot_signal(symbol)
            bias   = result["bias"]
            pct    = result["percentile"]
            note   = result["note"]
            icon   = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
            print(
                f"  {icon} {symbol:<8} {bias:<10} "
                f"Pctile: {pct:5.1f}% | {note}"
            )
        print("=" * 55)
