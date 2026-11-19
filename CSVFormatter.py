from decimal import Decimal, InvalidOperation
from pathlib import Path
import pandas as pd


class CSVFormatter:
    """Handles plain-number formatting and scientific-notation-free CSV exports."""

    @staticmethod
    def format_plain_number(val) -> str:
        """Format a float/Decimal cleanly in fixed-point notation without scientific notation."""
        if pd.isna(val):
            return ""
        try:
            d = Decimal(str(float(val))).normalize()
            text = format(d, "f")
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return "0" if text in {"", "-0"} else text
        except (InvalidOperation, ValueError, TypeError):
            return str(val)

    @classmethod
    def export_csv_no_scientific(cls, df: pd.DataFrame, path: Path) -> None:
        """Export DataFrame to CSV ensuring floating-point numbers never use scientific notation."""
        export_df = df.copy()
        for col in export_df.columns:
            if pd.api.types.is_float_dtype(export_df[col]):
                export_df[col] = export_df[col].map(cls.format_plain_number)
        export_df.to_csv(path, index=False)
