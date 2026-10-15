"""
execution.py - MetaTrader 5 Order Execution and Position Management
"""

import logging
from typing import Optional, Dict, Any, List
import MetaTrader5 as mt5
import config

logger = logging.getLogger("Execution")


def get_open_bot_positions(symbol: str) -> List[Any]:
    """Retrieves open positions belonging to this specific bot magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    bot_positions = [p for p in positions if p.magic == config.MAGIC_NUMBER]
    return bot_positions


def execute_market_order(
    symbol: str,
    action: str,            # 'BUY' or 'SELL'
    lot_size: float,
    sl_price: float,
    tp_price: float,
    comment: str = "Gold-AI-Bot"
) -> Optional[Any]:
    """
    Submits a market order to MT5 with attached Stop-Loss and Take-Profit.
    Includes explicit Demo guard and pre-flight validation.
    """
    # 1. Demo Mode Safety Verification
    account = mt5.account_info()
    if account is None:
        logger.error("Could not fetch account info for order execution.")
        return None

    if config.DEMO_ONLY_MODE and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        logger.critical("SAFETY INTERCEPTION: Live money account detected! DEMO_ONLY_MODE is enabled. Aborting order.")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Cannot get symbol info for {symbol}")
        return None

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    # 2. Determine price and order type
    if action == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = symbol_info.ask
    elif action == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = symbol_info.bid
    else:
        logger.error(f"Invalid order action: {action}")
        return None

    # Determine filling mode supported by broker
    filling_type = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & mt5.ORDER_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "price": float(price),
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 20,           # Max allowable slippage in points
        "magic": config.MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    # 3. Pre-flight Check with MT5 Server
    check_result = mt5.order_check(request)
    if check_result is None or check_result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if check_result is None else check_result.comment
        logger.error(f"Order pre-check failed: {err}")
        return None

    # 4. Send Order
    logger.info(f"Submitting {action} {lot_size} lots on {symbol} @ {price} | SL: {sl_price} | TP: {tp_price}")
    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else f"Code {result.retcode}: {result.comment}"
        logger.error(f"Order execution rejected by MT5: {err}")
        return None

    logger.info(f"=== ORDER EXECUTED SUCCESSFULLY ===")
    logger.info(f"Order Ticket: #{result.order} | Volume: {result.volume} | Fill Price: {result.price}")
    return result


def close_all_bot_positions(symbol: str) -> int:
    """Closes all open positions opened by this bot. Useful for emergency stop or manual exit."""
    positions = get_open_bot_positions(symbol)
    closed_count = 0

    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": config.MAGIC_NUMBER,
            "comment": "Bot-Emergency-Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            closed_count += 1
            logger.info(f"Closed position #{pos.ticket} for {symbol}")
        else:
            logger.error(f"Failed to close position #{pos.ticket}: {mt5.last_error()}")

    return closed_count
