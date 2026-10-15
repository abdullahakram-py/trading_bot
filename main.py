"""
main.py - Master Controller for MT5 AI Gold Trading Bot
Run from VS Code Terminal:
    python main.py
Or pass arguments:
    python main.py --test
    python main.py --fetch
    python main.py --train
    python main.py --backtest
    python main.py --run
"""

import argparse
import sys
import time
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

import config
from mt5_connector import initialize_mt5, discover_gold_symbol, get_symbol_spec, shutdown_mt5
from data_feed import fetch_historical_candles, save_candles_to_cache, load_cached_candles, get_mt5_timeframe
from indicators import add_technical_indicators
from ml_model import train_ai_model, load_ai_model
from strategy import evaluate_market_signal
from risk_manager import calculate_lot_size, check_trade_eligibility
from backtester import compare_strategies
from execution import get_open_bot_positions, execute_market_order


def generate_synthetic_gold_candles(num_bars: int = 1500) -> pd.DataFrame:
    """
    Generates realistic synthetic Gold (XAUUSD) candles for offline learning and
    backtesting when MT5 desktop is closed or on market weekends.
    """
    import numpy as np

    np.random.seed(42)
    start_price = 2650.0  # Representative Gold price
    returns = np.random.normal(0.0001, 0.0035, num_bars)
    price_series = start_price * np.cumprod(1 + returns)

    dates = pd.date_range(end=datetime.utcnow(), periods=num_bars, freq="1h")
    records = []

    for i in range(num_bars):
        close_p = price_series[i]
        volatility = close_p * np.random.uniform(0.001, 0.004)
        open_p = close_p + np.random.uniform(-volatility * 0.5, volatility * 0.5)
        high_p = max(open_p, close_p) + np.random.uniform(0, volatility)
        low_p = min(open_p, close_p) - np.random.uniform(0, volatility)

        records.append({
            "time": dates[i],
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "tick_volume": int(np.random.uniform(500, 4000)),
            "spread": 25
        })

    df = pd.DataFrame(records)
    return df


def cmd_test_connection():
    """Tests MT5 terminal connection and prints account & symbol details."""
    print("\n--- [Step 1] Testing MT5 Connection ---")
    if not initialize_mt5():
        print("[!] Connection to MT5 could not be established.")
        print("    -> Make sure MetaTrader 5 desktop is open and logged into your Demo account.")
        return False

    symbol = discover_gold_symbol()
    spec = get_symbol_spec(symbol)
    print(f"\n[+] Active Symbol: {symbol}")
    print(f"    - Bid / Ask: {spec['bid']} / {spec['ask']}")
    print(f"    - Spread: {spec['spread_points']} points (${spec['spread_price']:.2f})")
    print(f"    - Min Lot: {spec['min_volume']} | Max Lot: {spec['max_volume']} | Step: {spec['volume_step']}")
    print(f"    - Contract Size: {spec['contract_size']} oz\n")
    shutdown_mt5()
    return True


def cmd_fetch_data(force_live: bool = False) -> pd.DataFrame:
    """Downloads historical candles from MT5 or uses offline cached data."""
    print("\n--- [Step 2] Obtaining Historical Gold Candles ---")
    if not force_live:
        try:
            df = load_cached_candles(config.DEFAULT_SYMBOL, config.TIMEFRAME_NAME)
            print(f"[+] Loaded {len(df)} historical candles from local cache.")
            return df
        except FileNotFoundError:
            pass

    if initialize_mt5():
        symbol = discover_gold_symbol()
        df = fetch_historical_candles(symbol, config.TIMEFRAME_NAME, config.CANDLE_COUNT_TRAIN)
        save_candles_to_cache(df, symbol, config.TIMEFRAME_NAME)
        shutdown_mt5()
        return df
    else:
        print("[*] MT5 terminal not active. Generating realistic Gold historical candles for offline learning & backtesting...")
        df = generate_synthetic_gold_candles(num_bars=2500)
        save_candles_to_cache(df, config.DEFAULT_SYMBOL, config.TIMEFRAME_NAME)
        return df


def cmd_train_ai():
    """Trains the Random Forest model on historical market data."""
    print("\n--- [Step 3] Training AI Machine Learning Model ---")
    df = cmd_fetch_data()
    model = train_ai_model(df)
    print("[+] AI Training Complete! Model is ready for inference.")
    return model


def cmd_backtest():
    """Runs the backtest comparing Baseline Rules against AI Filtered Strategy."""
    print("\n--- [Step 4] Running Historical Backtest & Strategy Evaluation ---")
    df = cmd_fetch_data()

    # Load or train AI model
    try:
        model = load_ai_model()
    except FileNotFoundError:
        print("[*] No existing model found. Training fresh model first...")
        model = train_ai_model(df)

    compare_strategies(df, ai_model=model)


def cmd_run_live_demo():
    """Runs the real-time demo trading loop against MetaTrader 5."""
    print("\n--- [Step 5] Starting Demo Trading Engine ---")
    if not initialize_mt5():
        print("[!] Cannot run demo bot without active MT5 terminal.")
        return

    symbol = discover_gold_symbol()
    spec = get_symbol_spec(symbol)

    try:
        ai_model = load_ai_model()
        print("[+] Loaded trained AI Model.")
    except FileNotFoundError:
        print("[!] No trained AI model found. Training one now...")
        df_hist = fetch_historical_candles(symbol, config.TIMEFRAME_NAME, config.CANDLE_COUNT_TRAIN)
        ai_model = train_ai_model(df_hist)

    print(f"\n[+] Bot is active on symbol: {symbol} ({config.TIMEFRAME_NAME} candles)")
    print(f"[+] Risk limit: {config.RISK_PER_TRADE_PERCENT}% per trade | Max Open Positions: {config.MAX_OPEN_POSITIONS}")
    print("[+] Press Ctrl+C at any time in VS Code terminal to stop.\n")

    mt5_tf = get_mt5_timeframe(config.TIMEFRAME_NAME)
    last_processed_candle_time = None

    try:
        while True:
            # 1. Lightweight 2-bar check to see if a new completed candle has closed
            quick_rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 2)
            if quick_rates is None or len(quick_rates) < 2:
                time.sleep(5)
                continue

            # In MT5: quick_rates[0] is the newly completed closed bar, quick_rates[1] is the forming bar
            closed_bar_time = pd.to_datetime(quick_rates[0]["time"], unit="s", utc=True)

            # Evaluate only when a NEW completed candle arrives
            if closed_bar_time != last_processed_candle_time:
                last_processed_candle_time = closed_bar_time

                # Fetch full candles for technical indicators & AI model
                df = fetch_historical_candles(symbol, config.TIMEFRAME_NAME, config.CANDLE_COUNT_LIVE)
                closed_candle = df.iloc[-2]
                close_price = closed_candle["close"]

                print("\n" + "=" * 68)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] NEW {config.TIMEFRAME_NAME} CANDLE CLOSED @ ${close_price:.2f}")
                print("-" * 68)

                # 2. Evaluate Strategy Signal with AI
                signal_res = evaluate_market_signal(df, ai_model=ai_model, use_ai=True)
                signal = signal_res["signal"]
                ai_conf = signal_res["ai_confidence"]
                ai_str = f"{ai_conf:.1%}" if ai_conf is not None else "N/A"

                print(f"  Technical Action: {signal}  |  AI Model Probability: {ai_str}")
                print(f"  Decision Context: {signal_res['reason']}")

                # 3. Check Account & Open Positions
                open_bot_positions = get_open_bot_positions(symbol)
                account_info = mt5.account_info()
                print(f"  Account Equity: ${account_info.equity:.2f}  |  Active Trades: {len(open_bot_positions)}/{config.MAX_OPEN_POSITIONS}")

                if signal in ("BUY", "SELL"):
                    # 4. Pre-flight Risk Checks
                    eligible, reason = check_trade_eligibility(
                        account_info=account_info,
                        symbol_spec=spec,
                        open_positions_count=len(open_bot_positions),
                        daily_drawdown_percent=0.0
                    )

                    if not eligible:
                        print(f"  [X] Risk Guard Block: {reason}")
                    else:
                        # 5. Position Sizing
                        lot_size = calculate_lot_size(
                            equity=account_info.equity,
                            entry_price=signal_res["entry_price"],
                            sl_price=signal_res["sl"],
                            symbol_spec=spec
                        )

                        # 6. Execute Order on Demo Account
                        print(f"  [*] Executing {signal} {lot_size} lots @ {signal_res['entry_price']:.2f} (SL: {signal_res['sl']:.2f}, TP: {signal_res['tp']:.2f})...")
                        execute_market_order(
                            symbol=symbol,
                            action=signal,
                            lot_size=lot_size,
                            sl_price=signal_res["sl"],
                            tp_price=signal_res["tp"]
                        )
                print("=" * 68 + "\n")

            else:
                # Lightweight status line without downloading 300 bars or spamming terminal
                tick = mt5.symbol_info_tick(symbol)
                bid_str = f"{tick.bid:.2f}" if tick else "N/A"
                ask_str = f"{tick.ask:.2f}" if tick else "N/A"
                account_info = mt5.account_info()
                eq_str = f"${account_info.equity:.2f}" if account_info else "N/A"
                open_pos = get_open_bot_positions(symbol)
                sys.stdout.write(
                    f"\r[*] [{datetime.now().strftime('%H:%M:%S')}] Monitoring {symbol} ({config.TIMEFRAME_NAME}) | "
                    f"Bid/Ask: {bid_str}/{ask_str} | Equity: {eq_str} | Open Trades: {len(open_pos)} "
                )
                sys.stdout.flush()

            # Sleep 10 seconds before next bar closure check
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[*] Stopping Demo Trading Bot safely...")
    finally:
        shutdown_mt5()


def cmd_change_timeframe():
    """Allows user to change active timeframe between H1, M15, M5."""
    print("\n--- Select Trading Timeframe ---")
    print(f"  Current Timeframe: {config.TIMEFRAME_NAME}")
    print("  [1] H1  (1 Hour  - Best for trend stability, 1 candle/hour)")
    print("  [2] M15 (15 Min  - Balanced for demo testing, 4 candles/hour)")
    print("  [3] M5  (5 Min   - Fast demo testing, 12 candles/hour)")
    choice = input("Select timeframe (1-3): ").strip()
    if choice == "1":
        config.TIMEFRAME_NAME = "H1"
    elif choice == "2":
        config.TIMEFRAME_NAME = "M15"
    elif choice == "3":
        config.TIMEFRAME_NAME = "M5"
    print(f"[+] Active timeframe updated to: {config.TIMEFRAME_NAME}\n")


def interactive_menu():
    """Interactive console menu for beginners."""
    while True:
        print("\n" + "=" * 62)
        print("    MT5 AI GOLD TRADING BOT (XAUUSD) - VS CODE CONTROLLER")
        print("=" * 62)
        print(f"  Active Symbol: {config.DEFAULT_SYMBOL}  |  Timeframe: {config.TIMEFRAME_NAME}")
        print("-" * 62)
        print("  [1] Test MT5 Connection & Account Status")
        print("  [2] Download / Cache Historical Gold Candles")
        print("  [3] Train AI Model (Random Forest with TimeSeries CV)")
        print("  [4] Run Strategy Backtest (Baseline vs AI Comparison)")
        print("  [5] Run Live Demo Trading Bot (Continuous Loop)")
        print("  [6] Switch Timeframe (H1, M15, M5)")
        print("  [0] Exit")
        print("=" * 62)

        choice = input("Select an option (0-6): ").strip()
        if choice == "1":
            cmd_test_connection()
        elif choice == "2":
            cmd_fetch_data()
        elif choice == "3":
            cmd_train_ai()
        elif choice == "4":
            cmd_backtest()
        elif choice == "5":
            cmd_run_live_demo()
        elif choice == "6":
            cmd_change_timeframe()
        elif choice == "0":
            print("Exiting. Happy trading!")
            break
        else:
            print("Invalid choice, please select 0 to 6.")


def main():
    parser = argparse.ArgumentParser(description="MT5 AI Gold Trading Bot")
    parser.add_argument("--test", action="store_true", help="Test MT5 Connection")
    parser.add_argument("--fetch", action="store_true", help="Download historical candle data")
    parser.add_argument("--train", action="store_true", help="Train AI Model")
    parser.add_argument("--backtest", action="store_true", help="Run strategy backtest")
    parser.add_argument("--run", action="store_true", help="Run Demo Trading Bot loop")

    args = parser.parse_args()

    if args.test:
        cmd_test_connection()
    elif args.fetch:
        cmd_fetch_data()
    elif args.train:
        cmd_train_ai()
    elif args.backtest:
        cmd_backtest()
    elif args.run:
        cmd_run_live_demo()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
