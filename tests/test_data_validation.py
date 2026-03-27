import unittest

import pandas as pd

from src.data.validation import SchemaValidationError, validate_dataframe_schema


class TestDataValidation(unittest.TestCase):
    def test_st1_schema_valid(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "text": ["x", "y"],
                "polarization": [0, 1],
            }
        )
        validate_dataframe_schema(df, task="st1")

    def test_st1_schema_missing_column_raises(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "text": ["x", "y"],
            }
        )
        with self.assertRaises(SchemaValidationError):
            validate_dataframe_schema(df, task="st1")

    def test_st2_schema_valid(self):
        df = pd.DataFrame(
            {
                "id": ["x"],
                "text": ["sample"],
                "political": [1],
                "racial/ethnic": [0],
                "religious": [0],
                "gender/sexual": [0],
                "other": [0],
            }
        )
        validate_dataframe_schema(df, task="st2")


if __name__ == "__main__":
    unittest.main()
