"""
backtester.py - Rigorous Historical Backtesting Engine with Baseline vs AI Comparison
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
from indicators import add_technical_indicators
from ml_model import predict_setup_probability

logger = logging.getLogger("Backtester")


class Trade:
    def __init__(self, entry_bar: int, entry_time: Any, direction: str, entry_price: float, sl: float, tp: float, risk_usd: float):
        self.entry_bar = entry_bar
        self.entry_time = entry_time
        self.direction = direction  # 'BUY' or 'SELL'
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.risk_usd = risk_usd
        self.exit_bar = None
        self.exit_time = None
        self.exit_price = 0.0
        self.pnl_usd = 0.0
        self.return_pct = 0.0
        self.is_closed = False
        self.exit_reason = ""


def run_backtest(
    df: pd.DataFrame,
    ai_model: Optional[RandomForestClassifier] = None,
    use_ai: bool = True,
    initial_balance: float = 10000.0,
    spread_points: float = 30.0  # $0.30 spread on Gold
) -> Dict[str, Any]:
    """
    Executes a realistic bar-by-bar backtest:
    - Signal evaluated at candle close
    - Trade enters at next candle OPEN
    - Spread cost deducted from entry
    - Stops & Targets evaluated against subsequent High/Low prices
    """
    df = add_technical_indicators(df)
    balance = initial_balance
    equity_curve = [initial_balance]
    trades: List[Trade] = []
    active_trade: Optional[Trade] = None

    spread_cost = spread_points * 0.01

    for i in range(55, len(df) - 1):
        current_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]

        # 1. Manage Active Trade if any
        if active_trade is not None:
            high = next_bar["high"]
            low = next_bar["low"]

            hit_sl = False
            hit_tp = False

            if active_trade.direction == "BUY":
                if low <= active_trade.sl:
                    hit_sl = True
                elif high >= active_trade.tp:
                    hit_tp = True
            else:  # SELL
                if high >= active_trade.sl:
                    hit_sl = True
                elif low <= active_trade.tp:
                    hit_tp = True

            # If both touched in same candle, take conservative approach (assume SL hit)
            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                active_trade.exit_bar = i + 1
                active_trade.exit_time = next_bar["time"]
                active_trade.exit_price = active_trade.sl
                active_trade.pnl_usd = -active_trade.risk_usd
                active_trade.is_closed = True
                active_trade.exit_reason = "Stop Loss"
                balance += active_trade.pnl_usd
                active_trade = None

            elif hit_tp:
                active_trade.exit_bar = i + 1
                active_trade.exit_time = next_bar["time"]
                active_trade.exit_price = active_trade.tp
                # R:R ratio based on TP vs SL multiplier
                reward_ratio = config.TP_ATR_MULTIPLIER / config.SL_ATR_MULTIPLIER
                active_trade.pnl_usd = active_trade.risk_usd * reward_ratio
                active_trade.is_closed = True
                active_trade.exit_reason = "Take Profit"
                balance += active_trade.pnl_usd
                active_trade = None

        equity_curve.append(balance)

        # 2. Check for New Entry if no active trade
        if active_trade is None:
            close = current_bar["close"]
            ema20 = current_bar["ema20"]
            ema50 = current_bar["ema50"]
            rsi = current_bar["rsi"]
            atr = current_bar["atr"]

            # Technical Baseline Signals
            is_buy = (ema20 > ema50) and (close > ema20) and (50.0 <= rsi <= 68.0)
            is_sell = (ema20 < ema50) and (close < ema20) and (32.0 <= rsi <= 50.0)

            signal = None
            if is_buy:
                signal = "BUY"
            elif is_sell:
                signal = "SELL"

            if signal is not None:
                # Apply AI Confirmation Filter if enabled
                approved_by_ai = True
                if use_ai and ai_model is not None:
                    try:
                        prob = predict_setup_probability(ai_model, current_bar)
                        if prob < config.ML_MIN_CONFIDENCE:
                            approved_by_ai = False
                    except Exception:
                        approved_by_ai = True

                if approved_by_ai:
                    risk_amount = balance * (config.RISK_PER_TRADE_PERCENT / 100.0)
                    sl_dist = atr * config.SL_ATR_MULTIPLIER
                    tp_dist = atr * config.TP_ATR_MULTIPLIER

                    # Enter at next bar's OPEN with spread adjustment
                    if signal == "BUY":
                        entry_price = next_bar["open"] + spread_cost
                        sl = entry_price - sl_dist
                        tp = entry_price + tp_dist
                    else:
                        entry_price = next_bar["open"] - spread_cost
                        sl = entry_price + sl_dist
                        tp = entry_price - tp_dist

                    new_trade = Trade(
                        entry_bar=i + 1,
                        entry_time=next_bar["time"],
                        direction=signal,
                        entry_price=entry_price,
                        sl=sl,
                        tp=tp,
                        risk_usd=risk_amount
                    )
                    trades.append(new_trade)
                    active_trade = new_trade

    # Performance Analytics Calculation
    closed_trades = [t for t in trades if t.is_closed]
    total_trades = len(closed_trades)

    if total_trades == 0:
        return {
            "name": "AI Filtered" if use_ai else "Baseline Rules",
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "profit_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "final_balance": balance,
        }

    wins = [t for t in closed_trades if t.pnl_usd > 0]
    losses = [t for t in closed_trades if t.pnl_usd < 0]

    win_rate = (len(wins) / total_trades) * 100.0
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    net_profit = balance - initial_balance
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    # Calculate Maximum Drawdown %
    equity_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_arr)
    drawdowns = (peaks - equity_arr) / peaks * 100.0
    max_drawdown_pct = np.max(drawdowns)

    return {
        "name": "AI Filtered" if use_ai else "Baseline Rules",
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "profit_pct": (net_profit / initial_balance) * 100.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "final_balance": balance,
    }


def compare_strategies(df: pd.DataFrame, ai_model: Optional[RandomForestClassifier] = None):
    """Runs and prints a formatted comparative analysis between Baseline and AI."""
    print("\n" + "=" * 65)
    print("        STRATEGY BACKTEST RESULTS (GOLD / XAUUSD)")
    print("=" * 65)

    res_baseline = run_backtest(df, ai_model=None, use_ai=False)
    res_ai = run_backtest(df, ai_model=ai_model, use_ai=True) if ai_model else None

    results = [res_baseline]
    if res_ai:
        results.append(res_ai)

    table = pd.DataFrame(results)
    table = table[[
        "name", "total_trades", "win_rate", "profit_pct",
        "net_profit", "profit_factor", "max_drawdown_pct", "final_balance"
    ]]
    table.columns = [
        "Strategy", "Trades", "Win Rate %", "Return %",
        "Net Profit ($)", "Profit Factor", "Max Drawdown %", "Final Equity ($)"
    ]

    print(table.to_string(index=False))
    print("=" * 65 + "\n")
    return results
