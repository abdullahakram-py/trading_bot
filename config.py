"""
config.py - Central Configuration for MT5 AI Trading Bot (Gold / XAUUSD)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if present
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for folder in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    folder.mkdir(exist_ok=True)

# -------------------------------------------------------------
# TRADING INSTRUMENT & TIMEFRAME
# -------------------------------------------------------------
# Common broker naming variations for Gold
CANDIDATE_SYMBOLS = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a", "XAUUSD.", "XAUUSD_i"]
DEFAULT_SYMBOL = "XAUUSD"

# Timeframe in MT5 constant format (e.g., H1 = 1 Hour)
# mt5.TIMEFRAME_H1 = 16385, mt5.TIMEFRAME_M15 = 15, mt5.TIMEFRAME_M5 = 5
TIMEFRAME_NAME = "H1"
CANDLE_COUNT_TRAIN = 4000  # Number of hourly bars to fetch for AI training
CANDLE_COUNT_LIVE = 300    # Number of bars required for live indicator calculations

# -------------------------------------------------------------
# RISK MANAGEMENT RULES (CAPITAL PROTECTION FIRST)
# -------------------------------------------------------------
DEMO_ONLY_MODE = os.getenv("DEMO_ONLY_MODE", "True").lower() in ("true", "1", "yes")

# Capital allocation & loss limits
RISK_PER_TRADE_PERCENT = float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", 0.5))  # 0.5% risk per trade
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", 2.5))  # Halt trading if daily loss reaches 2.5%
MAX_OPEN_POSITIONS = 1                                                         # Max 1 open trade per instrument

# Volatility-based Stop-Loss & Take-Profit (based on Average True Range - ATR)
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.5   # Stop loss is 1.5x ATR away from entry
TP_ATR_MULTIPLIER = 2.5   # Take profit is 2.5x ATR away from entry (1 : 1.67 Risk/Reward)

# Spread protection: Maximum allowed spread in price points (XM Gold spread is typically 50-65 points)
MAX_SPREAD_POINTS = 85    # 85 points on Gold = $0.85 max allowable spread

# Unique Bot Identifier for MetaTrader 5 order tracking
MAGIC_NUMBER = 884102

# -------------------------------------------------------------
# MACHINE LEARNING SETTINGS
# -------------------------------------------------------------
ML_MODEL_PATH = MODELS_DIR / "gold_rf_model.joblib"
ML_MIN_CONFIDENCE = 0.58       # Probability threshold (e.g., 58% confidence required from AI)
CV_SPLITS = 5                  # TimeSeriesSplit folds for chronological validation
TEST_SIZE_RATIO = 0.20         # Last 20% reserved for out-of-sample forward testing

# -------------------------------------------------------------
# MT5 ACCOUNT CREDENTIALS (OPTIONAL)
# If left empty, the script attaches to the already open & logged in MT5 desktop application.
# -------------------------------------------------------------
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")
