import pandas as pd
import pandera.pandas as pa
from typing import List, Dict, Any, Optional

class ValidationResult:
    """
    Handles the output of a Pandera validation.
    """
    def __init__(self, df: pd.DataFrame, errors: Optional[pd.DataFrame] = None):
        self.df = df
        self.errors = errors # Pandera failure_cases DataFrame

    @property
    def is_valid(self) -> bool:
        return self.errors is None or self.errors.empty

    def get_clean(self) -> pd.DataFrame:
        """Returns the DataFrame without any rows that failed validation."""
        if self.is_valid:
            return self.df.copy()
        
        # Pandera failure_cases usually has an 'index' column referring to the original df index
        failed_indices = self.errors["index"].unique()
        # Ensure we only drop if indices exist (handle None/NaN if necessary)
        failed_indices = [i for i in failed_indices if i is not None]
        return self.df.drop(index=failed_indices).copy()

    def get_errors(self) -> pd.DataFrame:
        """Returns only the rows from the original DataFrame that failed any check."""
        if self.is_valid:
            return pd.DataFrame(columns=self.df.columns)
        
        failed_indices = self.errors["index"].unique()
        failed_indices = [i for i in failed_indices if i is not None]
        return self.df.loc[failed_indices].copy()

    def report(self) -> pd.DataFrame:
        """Returns a summary of failed checks (count of unique affected rows)."""
        if self.is_valid:
            return pd.DataFrame(columns=["check", "unique_failed_rows"])
        
        # Group by check and count unique indices
        summary = self.errors.groupby("check")["index"].nunique().reset_index(name="unique_failed_rows")
        return summary

class DataValidator:
    """
    A modular wrapper around Pandera DataFrameSchema.
    """
    def __init__(self, columns: Optional[Dict[str, pa.Column]] = None):
        self.columns = columns or {}
        self.global_checks = []

    def add_column(self, name: str, dtype: Any = None, checks: Optional[List[pa.Check]] = None, **kwargs):
        """Add a column-level validation."""
        self.columns[name] = pa.Column(dtype=dtype, checks=checks, **kwargs)
        return self

    def add_check(self, check: pa.Check):
        """Add a global/DataFrame-level check (e.g. comparing two columns)."""
        self.global_checks.append(check)
        return self

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validates the DataFrame using the defined schema.
        Uses lazy=True to catch all errors instead of stopping at the first one.
        """
        schema = pa.DataFrameSchema(
            columns=self.columns,
            checks=self.global_checks
        )
        
        try:
            schema.validate(df, lazy=True)
            return ValidationResult(df)
        except pa.errors.SchemaErrors as err:
            return ValidationResult(df, err.failure_cases)

# Helper function for quick creation
def create_validator() -> DataValidator:
    return DataValidator()
