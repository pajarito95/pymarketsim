import pandas as pd
import numpy as np


def load_stockmarl_close_series(csv_path: str, ticker: str) -> np.ndarray:
    raw = pd.read_csv(csv_path)

    ticker_row = raw.iloc[0]
    data = raw.iloc[2:].copy().reset_index(drop=True)

    close_cols = [col for col in raw.columns if str(col).startswith("Close")]
    selected_col = None

    for col in close_cols:
        if ticker_row[col] == ticker:
            selected_col = col
            break

    if selected_col is None:
        raise ValueError(f"Could not find Close column for ticker {ticker}")

    prices = pd.to_numeric(data[selected_col], errors="coerce").dropna().to_numpy(dtype=float)
    return prices


def load_two_asset_series(
    csv_path: str,
    ticker_a: str = "AAPL",
    ticker_b: str = "XOM",
    sim_time: int = 500,
) -> dict[str, np.ndarray]:
    prices_a = load_stockmarl_close_series(csv_path, ticker=ticker_a)
    prices_b = load_stockmarl_close_series(csv_path, ticker=ticker_b)

    n = min(len(prices_a), len(prices_b), sim_time + 1)
    return {
        ticker_a: prices_a[:n],
        ticker_b: prices_b[:n],
    }

def load_multi_asset_series(
    csv_path: str,
    tickers: list[str],
    sim_time: int = 500,
) -> dict[str, np.ndarray]:
    if len(tickers) == 0:
        raise ValueError("tickers must contain at least one ticker")
    
    series = {}
    lengths = []

    for ticker in tickers:
        prices = load_stockmarl_close_series(csv_path, ticker=ticker)
        series[ticker] = prices
        lengths.append(len(prices))

    n = min(min(lengths), sim_time + 1)

    return {ticker: prices[:n] for ticker, prices in series.items()}