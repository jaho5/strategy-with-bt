"""
Historical stock data downloader using yfinance.

Downloads OHLCV data for equities and ETFs, with local file-based caching
and automatic retries on failure.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"

SECTOR_ETFS = [
    "SPY", "QQQ", "XLF", "XLE", "XLK", "XLV", "XLI", "XLB",
    "XLU", "XLP", "XLY", "IWM", "EFA", "EEM", "TLT", "GLD",
    "SLV", "USO",
]

# Top 100 S&P 500 constituents by market cap (as of early 2026).
_SP500_TOP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B",
    "LLY", "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA",
    "PG", "COST", "JNJ", "HD", "ABBV", "WMT", "NFLX", "BAC",
    "KO", "MRK", "CVX", "CRM", "ORCL", "AMD", "PEP", "ACN",
    "TMO", "LIN", "MCD", "CSCO", "ABT", "ADBE", "WFC", "DHR",
    "TXN", "PM", "GE", "QCOM", "ISRG", "INTU", "CMCSA", "NEE",
    "AMGN", "NOW", "IBM", "CAT", "GS", "AMAT", "VZ", "PFE",
    "BKNG", "T", "RTX", "SPGI", "LOW", "BLK", "UNP", "HON",
    "MS", "DE", "ELV", "PLD", "SBUX", "ADP", "MDLZ", "BA",
    "SCHW", "MMC", "CB", "SYK", "GILD", "LMT", "ADI", "VRTX",
    "BMY", "AMT", "TJX", "CI", "MO", "SO", "LRCX", "TMUS",
    "FI", "CME", "DUK", "CL", "ZTS", "SLB", "BDX", "REGN",
    "PGR", "SNPS", "EOG", "AON",
]


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    """Produce a deterministic filename-safe cache key."""
    raw = f"{'_'.join(sorted(tickers))}_{start}_{end}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return digest


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _read_cache(key: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{key}.parquet"
    if path.exists():
        logger.debug("Cache hit: %s", path)
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("Corrupt cache file %s – will re-download", path)
            path.unlink(missing_ok=True)
    return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    _ensure_cache_dir()
    path = CACHE_DIR / f"{key}.parquet"
    df.to_parquet(path)
    logger.debug("Cached data to %s", path)


def _download_with_retry(
    tickers: list[str],
    start: str,
    end: str,
    *,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance with exponential-backoff retries.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.
    start, end : str
        Date strings accepted by yfinance (e.g. ``"2020-01-01"``).
    max_retries : int
        Number of attempts before giving up.
    backoff : float
        Base seconds for exponential back-off between retries.

    Returns
    -------
    pd.DataFrame
        Multi-level-columned DataFrame when multiple tickers are requested,
        or a simple OHLCV DataFrame for a single ticker.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                threads=True,
            )
            if df.empty:
                raise ValueError(
                    f"yfinance returned empty DataFrame for {tickers}"
                )
            return df
        except Exception as exc:
            last_exc = exc
            wait = backoff ** attempt
            logger.warning(
                "Download attempt %d/%d failed (%s). Retrying in %.1fs …",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {tickers} after {max_retries} attempts"
    ) from last_exc


# ── Public API ────────────────────────────────────────────────────────────────


def download_universe(
    tickers: list[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for *tickers* between *start* and *end*.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    start, end : str
        Date range in ``"YYYY-MM-DD"`` format.
    use_cache : bool
        If True (default), look up / store results in the local Parquet cache.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date.  For multiple tickers the columns are
        a ``MultiIndex`` of ``(Price, Ticker)``.
    """
    if not tickers:
        raise ValueError("tickers list must not be empty")

    key = _cache_key(tickers, start, end)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            logger.info(
                "Loaded %d rows from cache for %d tickers",
                len(cached),
                len(tickers),
            )
            return cached

    logger.info(
        "Downloading %d tickers (%s … %s) from yfinance",
        len(tickers),
        start,
        end,
    )
    df = _download_with_retry(tickers, start, end)

    # For a single ticker yfinance returns flat columns; normalise to
    # MultiIndex so the caller always gets the same shape.
    if len(tickers) == 1 and not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product(
            [df.columns, tickers], names=["Price", "Ticker"]
        )

    if use_cache:
        _write_cache(key, df)

    return df


def get_sp500_tickers() -> list[str]:
    """Return the top-100 S&P 500 constituents by market capitalisation."""
    return list(_SP500_TOP100)


def download_etf_data(
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a set of major sector / asset-class ETFs.

    The ETF universe is: SPY, QQQ, XLF, XLE, XLK, XLV, XLI, XLB, XLU,
    XLP, XLY, IWM, EFA, EEM, TLT, GLD, SLV, USO.

    Parameters
    ----------
    start, end : str
        Date range in ``"YYYY-MM-DD"`` format.
    use_cache : bool
        If True (default), look up / store results in the local Parquet cache.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date with a ``MultiIndex`` of
        ``(Price, Ticker)`` columns.
    """
    return download_universe(
        SECTOR_ETFS,
        start,
        end,
        use_cache=use_cache,
    )


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Quick smoke test: download a small window for a handful of tickers.
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start, end = "2024-01-01", "2024-03-01"

    df = download_universe(tickers, start, end)
    print(f"\ndownload_universe  →  {df.shape}")
    print(df.head())

    etf_df = download_etf_data(start, end)
    print(f"\ndownload_etf_data  →  {etf_df.shape}")
    print(etf_df.head())

    print(f"\nget_sp500_tickers  →  {len(get_sp500_tickers())} tickers")
