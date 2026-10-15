"""
data_feed.py - Market Data Downloader & Offline Storage for Gold (XAUUSD)
"""

import logging
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
import config

logger = logging.getLogger("DataFeed")

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def get_mt5_timeframe(tf_name: str):
    return TIMEFRAME_MAP.get(tf_name.upper(), mt5.TIMEFRAME_H1)


def fetch_historical_candles(symbol: str, timeframe_str: str = config.TIMEFRAME_NAME, num_bars: int = config.CANDLE_COUNT_TRAIN) -> pd.DataFrame:
    """
    Downloads historical rates from MetaTrader 5 terminal.
    Returns a clean DataFrame with columns: ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
    """
    mt5_tf = get_mt5_timeframe(timeframe_str)

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise RuntimeError(f"Failed to fetch rates for {symbol} on {timeframe_str}: {error}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].copy()

    # Drop any nulls or invalid rows
    df.dropna(inplace=True)
    df.sort_values(by="time", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe_str}) from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


def save_candles_to_cache(df: pd.DataFrame, symbol: str, timeframe_str: str) -> Path:
    """Saves candle DataFrame to CSV in data/ directory."""
    filename = config.DATA_DIR / f"{symbol}_{timeframe_str}_history.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Saved candle data to: {filename}")
    return filename


def load_cached_candles(symbol: str, timeframe_str: str) -> pd.DataFrame:
    """Loads cached candles from CSV file if available."""
    filename = config.DATA_DIR / f"{symbol}_{timeframe_str}_history.csv"
    if not filename.exists():
        raise FileNotFoundError(f"No cached file found at {filename}")

    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    logger.info(f"Loaded {len(df)} candles from cache: {filename}")
    return df
