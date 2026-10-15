"""
indicators.py - Technical Indicators and Volatility Metrics
"""

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Computes key technical indicators:
    - EMA 20 & EMA 50 (Trend direction & crossover)
    - RSI 14 (Momentum & overbought/oversold levels)
    - ATR 14 (Market volatility for dynamic stop-loss & take-profit)
    - Returns & Volatility features for Machine Learning model
    """
    df = df.copy()

    # 1. Trend: Exponential Moving Averages (EMA)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = (df["ema20"] - df["ema50"]) / df["close"]

    # 2. Momentum: Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # 3. Volatility: Average True Range (ATR)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # 4. Bollinger Bands (20 periods, 2 std dev)
    rolling_mean = df["close"].rolling(window=20).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # 5. Price Action & Momentum Features for Machine Learning
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["candle_body"] = (df["close"] - df["open"]) / df["atr"].replace(0, np.nan)
    df["candle_range"] = (df["high"] - df["low"]) / df["atr"].replace(0, np.nan)

    # Fill warmup NaN values
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)

    return df
