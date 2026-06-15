"""Deep autoencoder for reconstruction-based anomaly detection.

A symmetric dense autoencoder is trained per pair on an assumed-normal warm-up
period (the first two years). Bars that the network reconstructs poorly are
flagged: a regime the model never saw produces a large reconstruction error.
Features are standardised with the training set statistics only, so no test
information leaks into the scaler.

TensorFlow is imported lazily. If it is unavailable the detector degrades to a
no-op (zero scores, no flags) and logs a warning, so the rest of the system
still runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import settings
from config.logging_config import get_logger
from src.detectors import normalize_unit_interval, pair_feature_matrix, to_long_format

logger = get_logger()

_ENCODER_UNITS = [64, 32, 16, 8]
_DEFAULT_EPOCHS = 100
_BATCH_SIZE = 32
_PATIENCE = 10


def _import_tf():
    """Import TensorFlow if present.

    Returns:
        The tensorflow module, or None if it is not installed.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415  (intentional lazy import)

        return tf
    except ImportError:
        logger.warning("TensorFlow not installed; autoencoder detector is a no-op.")
        return None


def build_autoencoder(input_dim: int, tf):
    """Construct the symmetric dense autoencoder.

    Args:
        input_dim: Number of input features.
        tf: The tensorflow module.

    Returns:
        A compiled Keras model mapping input to its reconstruction.
    """
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in _ENCODER_UNITS:
        x = layers.Dense(units, activation="relu")(x)
    for units in reversed(_ENCODER_UNITS[:-1]):
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def _train_and_score_pair(
    block: pd.DataFrame,
    train_window: int,
    epochs: int,
    tf,
) -> tuple[pd.Series, pd.Series]:
    """Train the autoencoder on the warm-up window and score the full series.

    Args:
        block: Single-pair feature matrix.
        train_window: Number of leading bars treated as normal for training.
        epochs: Maximum training epochs.
        tf: The tensorflow module.

    Returns:
        A tuple of (reconstruction error series, boolean flag series).
    """
    medians = block.median(axis=0)
    filled = block.fillna(medians).fillna(0.0)
    values = filled.to_numpy()
    n, dim = values.shape
    if n <= train_window or dim == 0:
        empty = pd.Series(np.nan, index=block.index)
        return empty, pd.Series(False, index=block.index)

    train = values[:train_window]
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    all_scaled = scaler.transform(values)

    tf.random.set_seed(0)
    model = build_autoencoder(dim, tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=_PATIENCE, restore_best_weights=True
        )
    ]
    model.fit(
        train_scaled,
        train_scaled,
        epochs=epochs,
        batch_size=_BATCH_SIZE,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )

    recon = model.predict(all_scaled, verbose=0)
    errors = np.mean((all_scaled - recon) ** 2, axis=1)
    train_errors = errors[:train_window]
    threshold = np.percentile(train_errors, settings.DETECTOR.autoencoder_reconstruction_percentile)

    error_series = pd.Series(errors, index=block.index)
    flag_series = error_series > threshold
    return error_series, flag_series


def autoencoder_detector(
    features: pd.DataFrame,
    train_window: int | None = None,
    epochs: int = _DEFAULT_EPOCHS,
) -> pd.DataFrame:
    """Autoencoder reconstruction-error detector.

    Args:
        features: Multi-level (pair, feature) frame.
        train_window: Leading bars used as assumed-normal training data.
            Defaults to the backtest training window (about two years).
        epochs: Maximum training epochs per pair.

    Returns:
        Long-format detector output named "autoencoder". If TensorFlow is
        unavailable, scores are zero and no bars are flagged.
    """
    train_window = train_window or settings.BACKTEST.walk_forward_train_days
    pairs = list(features.columns.get_level_values(0).unique())
    score_wide = pd.DataFrame(0.0, index=features.index, columns=pairs)
    flag_wide = pd.DataFrame(False, index=features.index, columns=pairs)

    tf = _import_tf()
    if tf is None:
        return to_long_format(score_wide, flag_wide, "autoencoder")

    for pair in pairs:
        block = pair_feature_matrix(features, pair)
        errors, flags = _train_and_score_pair(block, train_window, epochs, tf)
        score_wide[pair] = normalize_unit_interval(errors).fillna(0.0)
        flag_wide[pair] = flags.fillna(False)

    return to_long_format(score_wide, flag_wide, "autoencoder")
