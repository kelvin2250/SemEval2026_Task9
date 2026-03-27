from typing import Iterable

import pandas as pd

ST1_REQUIRED_COLUMNS = ("id", "text", "polarization")
ST2_REQUIRED_COLUMNS = (
    "id",
    "text",
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
)


class SchemaValidationError(ValueError):
    pass


def _missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def validate_dataframe_schema(df: pd.DataFrame, task: str) -> None:
    task_normalized = task.lower().strip()
    if task_normalized not in {"st1", "st2"}:
        raise SchemaValidationError(f"Unsupported task '{task}'. Expected 'st1' or 'st2'.")

    required_columns = ST1_REQUIRED_COLUMNS if task_normalized == "st1" else ST2_REQUIRED_COLUMNS
    missing = _missing_columns(df, required_columns)
    if missing:
        raise SchemaValidationError(
            "Invalid dataset schema for "
            f"{task_normalized.upper()}. Missing columns: {missing}. "
            f"Required columns: {list(required_columns)}"
        )

    if df["text"].isna().any():
        raise SchemaValidationError("Invalid dataset schema: column 'text' contains null values.")

    if task_normalized == "st1":
        invalid_labels = set(df["polarization"].dropna().unique()) - {0, 1}
        if invalid_labels:
            raise SchemaValidationError(
                f"Invalid labels in 'polarization': {sorted(invalid_labels)}. Allowed values: [0, 1]."
            )
