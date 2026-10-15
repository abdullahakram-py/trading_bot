"""
ml_model.py - AI Machine Learning Signal Filter (Random Forest Classifier)
"""

import logging
from pathlib import Path
from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

import config
from indicators import add_technical_indicators

logger = logging.getLogger("MLModel")

# Features used for model training and live inference
FEATURE_COLUMNS = [
    "ema_diff",      # Trend divergence (EMA20 - EMA50) / Price
    "rsi",           # Momentum indicator (0-100)
    "atr_pct",       # Normalized market volatility
    "bb_pct",        # Location within Bollinger Bands (0 to 1)
    "return_1",      # 1-bar price return
    "return_3",      # 3-bar price return
    "return_5",      # 5-bar price return
    "candle_body",   # Candle body size relative to ATR
    "candle_range",  # Candle high-low range relative to ATR
]


def create_labels(df: pd.DataFrame, horizon: int = 8, profit_multiplier: float = 2.0, loss_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Labels each candle for a supervised classification task:
    Simulates entering at the close of candle `i`:
    Does the price reach (profit_multiplier * ATR) before hitting (loss_multiplier * ATR) within `horizon` bars?
    1 = Successful setup (Reached TP before SL)
    0 = Unsuccessful setup (Hit SL or timed out)
    """
    df = df.copy()
    labels = []

    for i in range(len(df) - horizon):
        entry_price = df["close"].iloc[i]
        atr = df["atr"].iloc[i]

        tp_distance = atr * profit_multiplier
        sl_distance = atr * loss_multiplier

        # Determine if rule baseline signal is Long or Short
        is_uptrend = df["ema20"].iloc[i] > df["ema50"].iloc[i]
        success = 0

        future_highs = df["high"].iloc[i + 1 : i + 1 + horizon].values
        future_lows = df["low"].iloc[i + 1 : i + 1 + horizon].values

        if is_uptrend:
            target_price = entry_price + tp_distance
            stop_price = entry_price - sl_distance

            # Check bar by bar in future horizon
            for h, l in zip(future_highs, future_lows):
                if l <= stop_price:
                    success = 0
                    break
                if h >= target_price:
                    success = 1
                    break
        else:
            target_price = entry_price - tp_distance
            stop_price = entry_price + sl_distance

            for h, l in zip(future_highs, future_lows):
                if h >= stop_price:
                    success = 0
                    break
                if l <= target_price:
                    success = 1
                    break

        labels.append(success)

    # Remaining bars at the end don't have enough forward horizon
    labels.extend([np.nan] * horizon)
    df["target"] = labels
    return df


def prepare_ml_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Ensures indicators and labels exist, then returns clean X, y."""
    df_feat = add_technical_indicators(df)
    df_labeled = create_labels(df_feat)

    # Drop rows without targets or valid features
    clean_df = df_labeled.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["target"].astype(int)
    return X, y


def train_ai_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Trains a Random Forest Classifier using chronological TimeSeriesSplit cross-validation.
    Ensures zero forward-looking data leakage.
    """
    X, y = prepare_ml_dataset(df)

    if len(X) < 200:
        raise ValueError(f"Insufficient historical data ({len(X)} bars) to train AI model. Need at least 500.")

    # Chronological Train-Test Split (Last 20% reserved for out-of-sample forward testing)
    split_idx = int(len(X) * (1 - config.TEST_SIZE_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training AI Model on {len(X_train)} samples, testing on {len(X_test)} out-of-sample samples...")

    # TimeSeries Cross-Validation on training set
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    cv_scores = []

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,            # Shallow depth prevents overfitting noise
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    for fold, (train_cv, val_cv) in enumerate(tscv.split(X_train)):
        rf.fit(X_train.iloc[train_cv], y_train.iloc[train_cv])
        val_pred = rf.predict(X_train.iloc[val_cv])
        score = accuracy_score(y_train.iloc[val_cv], val_pred)
        cv_scores.append(score)

    logger.info(f"TimeSeries CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Fit on all training data
    rf.fit(X_train, y_train)

    # Evaluate on untouched Out-Of-Sample forward test data
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    logger.info("=== Out-Of-Sample Performance Evaluation ===")
    logger.info(f"Test Accuracy: {acc:.4f} | Precision: {prec:.4f}")

    # Feature Importance analysis
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    logger.info("\nTop Predictive Features:\n" + importances.to_string())

    # Save model
    joblib.dump(rf, config.ML_MODEL_PATH)
    logger.info(f"Saved trained AI model to: {config.ML_MODEL_PATH}")

    return rf


def load_ai_model() -> RandomForestClassifier:
    """Loads the saved Random Forest model."""
    if not config.ML_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {config.ML_MODEL_PATH}. Run training first.")
    return joblib.load(config.ML_MODEL_PATH)


def predict_setup_probability(model: RandomForestClassifier, candle_row: pd.Series) -> float:
    """
    Computes model probability that the current trade setup succeeds.
    Returns float probability between 0.0 and 1.0.
    """
    features = pd.DataFrame([candle_row[FEATURE_COLUMNS]])
    probabilities = model.predict_proba(features)
    # Probability of class 1 (Successful trade)
    return float(probabilities[0][1])
