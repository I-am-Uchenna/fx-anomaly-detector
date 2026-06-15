"""Live FX data ingestion from Yahoo Finance.

Primary (and only) source is yfinance. The fetcher downloads daily OHLCV for
every configured pair plus the auxiliary macro tickers, aligns them to a single
business-day index, forward fills short gaps (weekends, holidays) and flags
longer gaps as data-quality problems. Raw downloads are cached to data/raw as
date-stamped parquet files.

A failed download for a single ticker is logged as a warning and skipped rather
than aborting the whole run, so a transient Yahoo outage on one symbol does not
take the system down.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import settings
from config.logging_config import get_logger

logger = get_logger()

# Canonical OHLCV column order used everywhere downstream.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class DataQuality:
    """Data quality summary for one ticker.

    Args:
        symbol: The ticker the metrics describe.
        n_rows: Number of rows after alignment.
        completeness_ratio: Fraction of expected business days with a close.
        gap_count: Number of gaps longer than the forward-fill limit.
        start: First date present.
        end: Last date present.
    """

    symbol: str
    n_rows: int
    completeness_ratio: float
    gap_count: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and subset a yfinance frame to canonical OHLCV columns.

    Args:
        frame: Raw single-ticker frame from yfinance with title-case columns.

    Returns:
        A frame containing exactly OHLCV_COLUMNS. Missing columns (for example
        volume, which Yahoo often omits for FX) are filled with NaN.
    """
    renamed = frame.rename(columns={c: str(c).lower().replace(" ", "_") for c in frame.columns})
    out = pd.DataFrame(index=renamed.index)
    for col in OHLCV_COLUMNS:
        out[col] = renamed[col] if col in renamed.columns else pd.NA
    return out.astype("float64")


def fetch_ticker(
    symbol: str,
    start: str,
    end: str | None,
    interval: str,
) -> pd.DataFrame | None:
    """Download OHLCV for one ticker from Yahoo Finance.

    Args:
        symbol: Yahoo Finance ticker, e.g. "EURUSD=X".
        start: Inclusive start date as "YYYY-MM-DD".
        end: Exclusive end date as "YYYY-MM-DD", or None for up to today.
        interval: Sampling interval understood by yfinance, e.g. "1d".

    Returns:
        A tz-naive DataFrame indexed by date with canonical OHLCV columns, or
        None if the download returned no data.

    Raises:
        ValueError: If symbol is empty.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string.")

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        logger.warning("No data returned for {}", symbol)
        return None

    # yfinance returns MultiIndex columns when a list of tickers is passed; for
    # a single ticker the columns are usually flat, but guard for both shapes.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = _normalise_columns(raw)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _align_and_fill(
    frames: dict[str, pd.DataFrame],
    max_fill_days: int,
) -> tuple[dict[str, pd.DataFrame], list[DataQuality]]:
    """Align all frames to a shared business-day index and forward fill gaps.

    Args:
        frames: Mapping of symbol to its OHLCV frame.
        max_fill_days: Maximum consecutive missing days to forward fill.

    Returns:
        A tuple of (aligned_frames, quality_reports).
    """
    if not frames:
        return {}, []

    full_start = min(f.index.min() for f in frames.values())
    full_end = max(f.index.max() for f in frames.values())
    business_index = pd.bdate_range(start=full_start, end=full_end)

    aligned: dict[str, pd.DataFrame] = {}
    reports: list[DataQuality] = []
    for symbol, frame in frames.items():
        reindexed = frame.reindex(business_index)
        present_before_fill = reindexed["close"].notna()
        # Forward fill only short gaps; limit caps the run length that is filled.
        filled = reindexed.ffill(limit=max_fill_days)

        # A gap longer than the fill limit leaves NaNs; count contiguous runs.
        still_missing = filled["close"].isna()
        gap_count = int((still_missing & ~still_missing.shift(1, fill_value=False)).sum())
        completeness = float(present_before_fill.mean())

        aligned[symbol] = filled
        reports.append(
            DataQuality(
                symbol=symbol,
                n_rows=len(filled),
                completeness_ratio=completeness,
                gap_count=gap_count,
                start=business_index.min(),
                end=business_index.max(),
            )
        )
    return aligned, reports


def _cache_raw(frames: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    """Persist each raw frame to a date-stamped parquet file.

    Args:
        frames: Mapping of symbol to OHLCV frame.
        raw_dir: Directory to write into; created if absent.

    Returns:
        None.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    for symbol, frame in frames.items():
        safe_symbol = symbol.replace("=", "").replace("^", "").replace(".", "")
        path = raw_dir / f"{safe_symbol}_{stamp}.parquet"
        frame.to_parquet(path)
        logger.debug("Cached {} rows for {} to {}", len(frame), symbol, path)


def fetch_fx_data(
    symbols: list[str] | None = None,
    start: str = settings.DATA_START_DATE,
    end: str | None = None,
    interval: str = settings.DATA_FREQUENCY,
    include_macro: bool = True,
    cache: bool = True,
    raw_dir: Path = _RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Fetch, align and cache OHLCV for the requested FX pairs.

    Args:
        symbols: Pair tickers to fetch. Defaults to every pair in FX_PAIRS.
        start: Inclusive start date "YYYY-MM-DD".
        end: Exclusive end date "YYYY-MM-DD", or None for today.
        interval: yfinance interval string, e.g. "1d".
        include_macro: If True, also fetch VIX, DXY and gold tickers.
        cache: If True, write raw downloads to raw_dir.
        raw_dir: Cache directory for raw parquet files.

    Returns:
        Mapping of symbol to an aligned, gap-filled OHLCV DataFrame. Symbols
        that failed to download are omitted.

    Raises:
        RuntimeError: If no requested symbol could be downloaded.
    """
    if symbols is None:
        symbols = [pair.symbol for pair in settings.FX_PAIRS]
    targets = list(symbols)
    if include_macro:
        targets += [t for t in settings.MACRO_TICKERS if t not in targets]

    downloaded: dict[str, pd.DataFrame] = {}
    for symbol in targets:
        try:
            frame = fetch_ticker(symbol, start=start, end=end, interval=interval)
        except Exception as exc:  # network or parsing failure for one ticker
            logger.warning("Download failed for {}: {}", symbol, exc)
            continue
        if frame is not None:
            downloaded[symbol] = frame

    if not downloaded:
        raise RuntimeError("No symbols could be downloaded from yfinance.")

    aligned, reports = _align_and_fill(downloaded, settings.MAX_FORWARD_FILL_DAYS)
    for report in reports:
        logger.info(
            "{}: rows={} completeness={:.3f} gaps={} range={}..{}",
            report.symbol,
            report.n_rows,
            report.completeness_ratio,
            report.gap_count,
            report.start.date() if report.start is not None else None,
            report.end.date() if report.end is not None else None,
        )

    if cache:
        _cache_raw(aligned, raw_dir)

    return aligned
