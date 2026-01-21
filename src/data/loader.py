import pickle
import pandas as pd
from pathlib import Path

def load_pkl(file_path: Path) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        df = pickle.loads(f.read())
    # Standardisation minimale
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close in {file_path}")
    # assure index datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def list_pkl_files(day_dir: Path) -> list[Path]:
    return sorted([p for p in day_dir.iterdir() if p.name.startswith("df_") and p.suffix == ".pkl"])

def ticker_from_filename(p: Path) -> str:
    # df_AMD_20250131_120000.pkl -> AMD
    return p.name.split("_")[1]
