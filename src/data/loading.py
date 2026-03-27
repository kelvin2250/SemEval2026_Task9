import pandas as pd

from .validation import validate_dataframe_schema


def load_csv_for_task(csv_path: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_dataframe_schema(df, task=task)
    return df
