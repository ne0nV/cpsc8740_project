from pathlib import Path
import pandas as pd

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_parquet(df: "pd.DataFrame", path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
