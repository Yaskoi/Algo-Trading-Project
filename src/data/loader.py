from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Optional, Tuple

import pandas as pd


DF_RE = re.compile(r"^df_(.+?)_")  # df_<ticker>_...pkl


def sanitize(s: str) -> str:
    # normalise pour comparer tickers "bizarres" vs filenames
    return (
        s.replace("^", "")
         .replace("=F", "_F")
         .replace("=X", "_X")
         .replace("=", "_")
         .replace("/", "_")
         .replace(".", "_")
    )


def extract_ticker_from_filename(filename: str) -> Optional[str]:
    m = DF_RE.match(filename)
    if not m:
        return None
    return m.group(1)


def load_pickle_df(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} does not contain a DataFrame")
    df = obj.copy()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to coerce
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def ticker_matches(desired: str, from_file: str) -> bool:
    # match either exact or via sanitize variants
    return desired == from_file or sanitize(desired) == sanitize(from_file)
