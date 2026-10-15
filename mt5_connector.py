"""
mt5_connector.py - Safe MetaTrader 5 Connection & Symbol Discovery
"""

import logging
import MetaTrader5 as mt5
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5Connector")


def initialize_mt5() -> bool:
    """
    Connects to the MT5 Desktop Terminal.
    First tries to attach to already running MT5 instance.
    If login credentials are provided in config, it logs into that account.
    """
    init_kwargs = {}
    if config.MT5_PATH:
        init_kwargs["path"] = config.MT5_PATH

    # Initialize connection (4-second timeout to avoid freezing if terminal not running)
    if not mt5.initialize(timeout=4000, **init_kwargs):
        error = mt5.last_error()
        logger.error(f"MT5 initialization failed: {error}")
        print("\n[!] Could not connect to MetaTrader 5.")
        print("    1. Please ensure MetaTrader 5 desktop application is installed and running.")
        print("    2. Make sure you are logged into your broker Demo account inside MT5.\n")
        return False

    # Optional explicit login if credentials are supplied
    if config.MT5_LOGIN and config.MT5_PASSWORD and config.MT5_SERVER:
        try:
            login_int = int(config.MT5_LOGIN)
            if not mt5.login(login=login_int, password=config.MT5_PASSWORD, server=config.MT5_SERVER):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
        except ValueError:
            logger.error("MT5_LOGIN must be an integer account number.")
            mt5.shutdown()
            return False

    # Verify Account Information
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to retrieve account info: {mt5.last_error()}")
        mt5.shutdown()
        return False

    logger.info("=== MetaTrader 5 Connected Successfully ===")
    logger.info(f"Account: {account_info.login} | Server: {account_info.server} | Currency: {account_info.currency}")
    logger.info(f"Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f} | Leverage: 1:{account_info.leverage}")

    # Check Demo Status
    is_demo = account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    logger.info(f"Account Trade Mode: {'DEMO' if is_demo else 'REAL / CONTEST'}")

    if config.DEMO_ONLY_MODE and not is_demo:
        logger.critical("SAFETY HALT: DEMO_ONLY_MODE is enabled, but current MT5 account is NOT a Demo account!")
        mt5.shutdown()
        return False

    return True


def discover_gold_symbol() -> str:
    """
    Finds the exact broker-specific ticker symbol for Gold (e.g. XAUUSD, GOLD, XAUUSDm).
    Enables it in MT5 Market Watch.
    """
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.warning("No symbols returned by MT5.")
        return config.DEFAULT_SYMBOL

    symbol_names = [s.name for s in all_symbols]

    for candidate in config.CANDIDATE_SYMBOLS:
        if candidate in symbol_names:
            logger.info(f"Discovered Gold symbol: '{candidate}'")
            # Ensure it is visible in Market Watch
            if not mt5.symbol_select(candidate, True):
                logger.warning(f"Could not select {candidate} in Market Watch.")
            return candidate

    # Fallback search by description containing 'gold'
    for s in all_symbols:
        if "gold" in s.description.lower() or "xau" in s.name.lower():
            logger.info(f"Discovered Gold symbol by description: '{s.name}' ({s.description})")
            mt5.symbol_select(s.name, True)
            return s.name

    logger.warning(f"Could not detect specific Gold symbol, defaulting to: {config.DEFAULT_SYMBOL}")
    return config.DEFAULT_SYMBOL


def get_symbol_spec(symbol: str) -> dict:
    """
    Fetches contract specifications for the symbol:
    lot limits, tick value, tick size, point, spread, etc.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol '{symbol}' not found in MT5!")

    spec = {
        "name": info.name,
        "point": info.point,
        "digits": info.digits,
        "spread_points": info.spread,
        "spread_price": info.spread * info.point,
        "contract_size": info.trade_contract_size,
        "min_volume": info.volume_min,
        "max_volume": info.volume_max,
        "volume_step": info.volume_step,
        "tick_value": info.trade_tick_value,
        "tick_size": info.trade_tick_size,
        "trade_mode": info.trade_mode,
        "ask": info.ask,
        "bid": info.bid,
    }
    return spec


def shutdown_mt5():
    """Disconnects and shuts down MT5 communication session."""
    mt5.shutdown()
    logger.info("MT5 connection closed.")
