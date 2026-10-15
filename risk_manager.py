"""
risk_manager.py - Capital Preservation, Position Sizing, and Drawdown Protection
"""

import logging
import math
from typing import Tuple, Dict, Any
import config

logger = logging.getLogger("RiskManager")


def calculate_lot_size(
    equity: float,
    entry_price: float,
    sl_price: float,
    symbol_spec: Dict[str, Any]
) -> float:
    """
    Calculates exact lot size so that if Stop-Loss is hit,
    the loss strictly equals RISK_PER_TRADE_PERCENT (e.g. 0.5% of account equity).
    Adjusts for broker min_volume, max_volume, and volume_step.
    """
    if sl_price <= 0 or entry_price <= 0:
        return symbol_spec["min_volume"]

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return symbol_spec["min_volume"]

    # Maximum dollar amount willing to lose on this single trade
    dollar_risk = equity * (config.RISK_PER_TRADE_PERCENT / 100.0)

    # Contract specifications from MT5
    contract_size = symbol_spec.get("contract_size", 100.0)  # Standard Gold = 100 oz
    tick_value = symbol_spec.get("tick_value", 1.0)
    tick_size = symbol_spec.get("tick_size", 0.01)

    # Dollar loss per 1.0 standard lot if price moves sl_distance
    loss_per_lot = (sl_distance / tick_size) * tick_value if tick_size > 0 else sl_distance * contract_size
    if loss_per_lot <= 0:
        return symbol_spec["min_volume"]

    raw_lot = dollar_risk / loss_per_lot

    # Broker volume boundary adjustments
    step = symbol_spec.get("volume_step", 0.01)
    min_vol = symbol_spec.get("min_volume", 0.01)
    max_vol = symbol_spec.get("max_volume", 50.0)

    # Round down to nearest valid step to never exceed risk threshold
    steps_count = math.floor(raw_lot / step)
    calculated_lot = round(steps_count * step, 2)

    # Clamp to broker limits
    final_lot = max(min_vol, min(calculated_lot, max_vol))

    # Calculate actual dollar and percentage risk resulting from broker min_vol clamp
    actual_dollar_risk = final_lot * loss_per_lot
    actual_risk_pct = (actual_dollar_risk / equity) * 100.0 if equity > 0 else 0.0

    if actual_risk_pct > config.RISK_PER_TRADE_PERCENT * 2:
        logger.warning(
            f"[!] ACCOUNT NOTICE: Equity (${equity:.2f}) is too small to achieve exact {config.RISK_PER_TRADE_PERCENT}% risk. "
            f"Broker minimum is {min_vol} lot. Actual Stop-Loss risk on {final_lot} lot is ${actual_dollar_risk:.2f} ({actual_risk_pct:.1f}% of equity). "
            f"(Tip: Reset demo balance to $1,000+ in MT5 for professional 0.5% sizing)."
        )
    else:
        logger.info(
            f"Position Sizing: Equity=${equity:.2f} | Risk=${dollar_risk:.2f} ({config.RISK_PER_TRADE_PERCENT}%) | "
            f"SL Distance=${sl_distance:.2f} | Lot Size={final_lot}"
        )
    return final_lot


def check_trade_eligibility(
    account_info: Any,
    symbol_spec: Dict[str, Any],
    open_positions_count: int,
    daily_drawdown_percent: float
) -> Tuple[bool, str]:
    """
    Pre-flight safety gate before any trade can be sent to MT5:
    1. Demo Mode Check
    2. Position limits (No over-trading)
    3. Daily loss ceiling (Protects against emotional revenge trading or bad market days)
    4. Spread limit (Protects against low-liquidity slippage)
    5. Free margin check
    """
    # 1. Demo account check
    if config.DEMO_ONLY_MODE and getattr(account_info, "trade_mode", None) != 0:
        # In MT5: ACCOUNT_TRADE_MODE_DEMO = 0
        return False, "SAFETY REJECTION: Account is not in DEMO mode."

    # 2. Maximum open positions check
    if open_positions_count >= config.MAX_OPEN_POSITIONS:
        return False, f"Max open positions reached ({open_positions_count}/{config.MAX_OPEN_POSITIONS})."

    # 3. Daily loss ceiling check
    if daily_drawdown_percent >= config.MAX_DAILY_LOSS_PERCENT:
        return False, f"Daily drawdown limit triggered ({daily_drawdown_percent:.2f}% >= {config.MAX_DAILY_LOSS_PERCENT}%)."

    # 4. Spread check
    current_spread = symbol_spec.get("spread_points", 0)
    if current_spread > config.MAX_SPREAD_POINTS:
        return False, f"Spread too wide ({current_spread} points > {config.MAX_SPREAD_POINTS} max allowed)."

    # 5. Margin adequacy (allows demo accounts with $50-$100 balance)
    free_margin = getattr(account_info, "margin_free", 0)
    if free_margin < 20.0:
        return False, f"Insufficient free margin (${free_margin:.2f} available, minimum $20.00 required)."

    return True, "Pre-flight risk checks passed successfully."
