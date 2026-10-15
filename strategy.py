"""
strategy.py - Trading Strategy Logic with AI Model Confirmation Filter
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
from indicators import add_technical_indicators
from ml_model import predict_setup_probability

logger = logging.getLogger("Strategy")


def evaluate_market_signal(
    df: pd.DataFrame,
    ai_model: Optional[RandomForestClassifier] = None,
    use_ai: bool = True
) -> Dict[str, Any]:
    """
    Evaluates the most recently CLOSED candle (df.iloc[-2]) to generate signals.
    Note: df.iloc[-1] is the currently forming, unfinished candle and must NOT be traded on!

    Returns dictionary with:
    - signal: 'BUY', 'SELL', or 'HOLD'
    - entry_price: float
    - sl: suggested stop loss price
    - tp: suggested take profit price
    - atr: ATR value
    - ai_confidence: model probability or None
    - reason: explanation of the decision
    """
    if len(df) < 60:
        return {"signal": "HOLD", "reason": "Insufficient candles for indicators"}

    # Calculate indicators if not already present
    if "ema20" not in df.columns or "atr" not in df.columns:
        df = add_technical_indicators(df)

    # Use the last completed bar
    closed_bar = df.iloc[-2]
    prev_bar = df.iloc[-3]

    close = closed_bar["close"]
    ema20 = closed_bar["ema20"]
    ema50 = closed_bar["ema50"]
    rsi = closed_bar["rsi"]
    atr = closed_bar["atr"]

    # Base Technical Hypotheses
    is_bullish_trend = ema20 > ema50 and close > ema20
    is_bearish_trend = ema20 < ema50 and close < ema20

    # RSI momentum confirmation (avoid entering at extreme tops or bottoms)
    rsi_bullish = 50.0 <= rsi <= 68.0
    rsi_bearish = 32.0 <= rsi <= 50.0

    raw_signal = "HOLD"
    if is_bullish_trend and rsi_bullish:
        raw_signal = "BUY"
    elif is_bearish_trend and rsi_bearish:
        raw_signal = "SELL"

    # AI Prediction Score (calculated on every closed bar for complete transparency)
    ai_confidence = None
    if use_ai and ai_model is not None:
        try:
            ai_confidence = predict_setup_probability(ai_model, closed_bar)
        except Exception as e:
            logger.warning(f"AI prediction error: {e}")

    if raw_signal == "HOLD":
        ai_desc = f" | AI Bullish Prob: {ai_confidence:.1%}" if ai_confidence is not None else ""
        return {
            "signal": "HOLD",
            "entry_price": close,
            "sl": 0.0,
            "tp": 0.0,
            "atr": atr,
            "ai_confidence": ai_confidence,
            "reason": f"No baseline setup (EMA20={ema20:.2f}, EMA50={ema50:.2f}, RSI={rsi:.1f}){ai_desc}"
        }

    # AI Confirmation Filter for actionable signals
    if use_ai and ai_model is not None and ai_confidence is not None:
        if ai_confidence < config.ML_MIN_CONFIDENCE:
            return {
                "signal": "HOLD",
                "entry_price": close,
                "sl": 0.0,
                "tp": 0.0,
                "atr": atr,
                "ai_confidence": ai_confidence,
                "reason": f"Technical {raw_signal} rejected by AI: Confidence {ai_confidence:.1%} < threshold {config.ML_MIN_CONFIDENCE:.1%}"
            }

    # Calculate Dynamic ATR-based Stop-Loss & Take-Profit
    sl_dist = atr * config.SL_ATR_MULTIPLIER
    tp_dist = atr * config.TP_ATR_MULTIPLIER

    if raw_signal == "BUY":
        sl = round(close - sl_dist, 2)
        tp = round(close + tp_dist, 2)
    else:
        sl = round(close + sl_dist, 2)
        tp = round(close - tp_dist, 2)

    return {
        "signal": raw_signal,
        "entry_price": close,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "ai_confidence": ai_confidence,
        "reason": f"Confirmed {raw_signal} signal (AI Confidence: {ai_confidence:.1%})" if ai_confidence else f"Confirmed {raw_signal} (Technical)"
    }
