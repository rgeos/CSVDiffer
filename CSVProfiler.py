import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CSVFormatter import CSVFormatter

try:
    from data_profiling import ProfileReport
except ImportError:
    from ydata_profiling import ProfileReport


@dataclass
class ResidualAnalysisResult:
    """Container for residual outlier analysis outputs."""

    html_section: str
    outlier_series: pd.Series
    total_rows: int
    outlier_count: int


class ColumnSchemaParser:
    """Parses COLUMN=TYPE[:FORMAT] CLI strings into pandas dtypes and date formats."""

    SUPPORTED_TYPES = {"string", "int", "float", "bool", "category", "datetime"}
    PANDAS_DTYPES = {
        "string": "string",
        "int": "Int64",
        "float": "float64",
        "bool": "boolean",
        "category": "category",
        "datetime": "string",
    }

    @classmethod
    def parse(cls, specs: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
        """Parse COLUMN=TYPE entries into column types and date formats."""
        types: dict[str, str] = {}
        date_formats: dict[str, str] = {}

        for spec in specs:
            if "=" not in spec:
                raise ValueError(f"Invalid column type {spec!r}; use COLUMN=TYPE")

            column, type_spec = spec.split("=", 1)
            column = column.strip()
            type_spec = type_spec.strip()

            if not column:
                raise ValueError(f"Missing column name in {spec!r}")

            kind, separator, date_format = type_spec.partition(":")
            kind = kind.lower()

            if kind not in cls.SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported type {kind!r} for column {column!r}. "
                    "Use string, int, float, bool, category, or datetime."
                )

            types[column] = kind
            if kind == "datetime" and separator:
                date_formats[column] = date_format

        return types, date_formats

    @classmethod
    def to_pandas_dtypes(cls, column_types: dict[str, str]) -> dict[str, str]:
        """Map parsed schema kinds to pandas read_csv dtype strings."""
        return {col: cls.PANDAS_DTYPES[kind] for col, kind in column_types.items()}


class ResidualAnalyzer:
    """Fits OLS regressions, flags residual outliers, and renders the comparison report."""

    DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("report_template.html")

    def __init__(
        self,
        target: str,
        z_threshold: float = 3.5,
        raw_threshold: float | None = None,
        log_target: bool = False,
        template_path: Path | None = None,
    ) -> None:
        self.target = target
        self.z_threshold = z_threshold
        self.raw_threshold = raw_threshold
        self.log_target = log_target
        self.template_path = template_path or self.DEFAULT_TEMPLATE_PATH

    @staticmethod
    def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
        """Fit ordinary least squares regression and compute predictions, residuals, R2, and RMSE."""
        X_design = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        fitted = X_design @ beta
        residuals = y - fitted

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return {
            "beta": beta,
            "fitted": fitted,
            "residuals": residuals,
            "r2": r2,
            "rmse": rmse,
            "n": len(y),
        }

    def _detect_outliers(
        self, residuals: np.ndarray, scale_label: str
    ) -> tuple[np.ndarray, str]:
        """Identify residual outliers via absolute cutoff, Modified Z-score (MAD), or standard Z-score."""
        if self.raw_threshold is not None:
            cutoff = float(self.raw_threshold)
            is_outlier = np.abs(residuals) > cutoff
            rule_desc = f"|Residual| > {cutoff:g} ({scale_label})"
            return is_outlier, rule_desc

        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad > 0:
            mod_z = 0.67449 * np.abs(residuals - med) / mad
            is_outlier = mod_z > self.z_threshold
            cutoff = self.z_threshold * mad / 0.67449
            rule_desc = (
                f"Modified Z-score > {self.z_threshold:g} "
                f"(|Residual - median| > {cutoff:.4f} on {scale_label})"
            )
            return is_outlier, rule_desc

        std = np.std(residuals)
        is_outlier = np.abs(residuals - np.mean(residuals)) > (self.z_threshold * std)
        cutoff = self.z_threshold * std
        rule_desc = f"Z-score > {self.z_threshold:g} (|Residual| > {cutoff:.4f} on {scale_label})"
        return is_outlier, rule_desc

    def _render_plot_base64(
        self,
        full_fit: dict,
        clean_fit: dict,
        is_outlier: np.ndarray,
        has_refit: bool,
        scale_label: str,
    ) -> str:
        """Render side-by-side residual comparison chart as a base64 PNG string."""
        inlier_mask = ~is_outlier
        n_outliers = int(is_outlier.sum())
        n_inliers = int(inlier_mask.sum())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)

        # Left panel: Original model with outliers highlighted.
        ax1 = axes[0]
        ax1.scatter(
            full_fit["fitted"][inlier_mask],
            full_fit["residuals"][inlier_mask],
            alpha=0.55,
            edgecolors="none",
            color="#337ab7",
            label=f"Inliers ({n_inliers})",
        )
        if n_outliers > 0:
            ax1.scatter(
                full_fit["fitted"][is_outlier],
                full_fit["residuals"][is_outlier],
                alpha=0.8,
                edgecolors="none",
                color="#d9534f",
                label=f"Outliers ({n_outliers})",
            )
        ax1.axhline(0, color="#333333", linestyle="--", linewidth=1.1)
        ax1.set_title(
            f"1. All Data (R² = {full_fit['r2']:.2%} | RMSE = {full_fit['rmse']:.3f})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax1.set_xlabel(f"Fitted Values ({scale_label})")
        ax1.set_ylabel("Residuals (Actual - Predicted)")
        ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Right panel: Refitted model without outliers.
        ax2 = axes[1]
        if has_refit:
            ax2.scatter(
                clean_fit["fitted"],
                clean_fit["residuals"],
                alpha=0.55,
                edgecolors="none",
                color="#2e8b57",
                label=f"Inliers Refit ({n_inliers})",
            )
            ax2.axhline(0, color="#333333", linestyle="--", linewidth=1.1)
            ax2.set_title(
                f"2. Outliers Removed (R² = {clean_fit['r2']:.2%} | RMSE = {clean_fit['rmse']:.3f})",
                fontsize=10.5,
                fontweight="bold",
            )
            ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
        else:
            ax2.text(
                0.5,
                0.5,
                "No outliers flagged (or < 3 inliers left)\nRefit identical to baseline model",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=10,
                color="#666666",
            )
            ax2.set_title(
                "2. Outliers Removed (No Change)", fontsize=10.5, fontweight="bold"
            )

        ax2.set_xlabel(f"Fitted Values ({scale_label})")
        ax2.set_ylabel("Residuals (Actual - Predicted)")
        ax2.grid(True, linestyle=":", alpha=0.5)

        transform_tag = " [log1p(target)]" if self.log_target else ""
        fig.suptitle(
            f"Regression Residual Comparison — Target: {self.target}{transform_tag}",
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def analyze(self, df: pd.DataFrame) -> ResidualAnalysisResult:
        """Run full OLS residual analysis and populate the external HTML template."""
        if self.target not in df.columns:
            raise ValueError(f"Target column {self.target!r} not found in CSV.")

        if not pd.api.types.is_numeric_dtype(
            df[self.target]
        ) or pd.api.types.is_bool_dtype(df[self.target]):
            raise ValueError(
                f"Target column {self.target!r} must be numeric (int or float)."
            )

        feature_cols = [
            col
            for col in df.columns
            if col != self.target
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        ]
        if not feature_cols:
            raise ValueError("No numeric predictor columns available for regression.")

        clean_df = df[[self.target] + feature_cols].dropna()
        if len(clean_df) < 3:
            raise ValueError("Not enough complete rows to fit a regression model.")

        y_raw = clean_df[self.target].to_numpy(dtype=float)
        if self.log_target:
            if np.any(y_raw <= -1.0):
                raise ValueError(
                    f"Cannot apply log1p transformation to {self.target!r}: values must be > -1."
                )
            y = np.log1p(y_raw)
            scale_label = f"log1p({self.target})"
        else:
            y = y_raw
            scale_label = "original scale"

        X = clean_df[feature_cols].to_numpy(dtype=float)

        full_fit = self.fit_ols(X, y)
        is_outlier, rule_desc = self._detect_outliers(
            full_fit["residuals"], scale_label
        )

        outlier_series = pd.Series(pd.NA, index=df.index, dtype="boolean")
        outlier_series.loc[clean_df.index] = is_outlier

        inlier_mask = ~is_outlier
        n_outliers = int(is_outlier.sum())
        n_inliers = int(inlier_mask.sum())
        pct_outliers = (n_outliers / len(clean_df)) * 100.0

        has_refit = n_inliers >= 3 and n_outliers > 0
        clean_fit = (
            self.fit_ols(X[inlier_mask], y[inlier_mask]) if has_refit else full_fit
        )

        orig_rmse_row = ""
        if self.log_target:
            full_orig_rmse = float(
                np.sqrt(np.mean((y_raw - np.expm1(full_fit["fitted"])) ** 2))
            )
            clean_orig_rmse = float(
                np.sqrt(
                    np.mean((y_raw[inlier_mask] - np.expm1(clean_fit["fitted"])) ** 2)
                )
            )
            orig_rmse_delta = clean_orig_rmse - full_orig_rmse
            orig_rmse_row = (
                f"<tr>\n"
                f"  <td><strong>Back-Transformed RMSE ({self.target})</strong></td>\n"
                f"  <td>{full_orig_rmse:.4f}</td>\n"
                f"  <td>{clean_orig_rmse:.4f}</td>\n"
                f"  <td>{orig_rmse_delta:+.4f}</td>\n"
                f"</tr>"
            )

        img_b64 = self._render_plot_base64(
            full_fit=full_fit,
            clean_fit=clean_fit,
            is_outlier=is_outlier,
            has_refit=has_refit,
            scale_label=scale_label,
        )

        if not self.template_path.is_file():
            raise FileNotFoundError(f"HTML template not found: {self.template_path}")

        template_html = self.template_path.read_text(encoding="utf-8")
        rendered_html = template_html.format(
            target=self.target,
            transform_tag=" [log1p(target)]" if self.log_target else "",
            total_rows=len(clean_df),
            n_outliers=n_outliers,
            pct_outliers=pct_outliers,
            rule_desc=rule_desc,
            n_predictors=len(feature_cols),
            predictors_list=", ".join(feature_cols),
            full_n=full_fit["n"],
            clean_n=clean_fit["n"],
            delta_n=clean_fit["n"] - full_fit["n"],
            full_r2=full_fit["r2"],
            clean_r2=clean_fit["r2"],
            r2_delta=(clean_fit["r2"] - full_fit["r2"]) * 100.0,
            scale_label=scale_label,
            full_rmse=full_fit["rmse"],
            clean_rmse=clean_fit["rmse"],
            rmse_delta=clean_fit["rmse"] - full_fit["rmse"],
            orig_rmse_row=orig_rmse_row,
            img_b64=img_b64,
        )

        return ResidualAnalysisResult(
            html_section=rendered_html,
            outlier_series=outlier_series,
            total_rows=len(clean_df),
            outlier_count=n_outliers,
        )


class CSVProfiler:
    """Generates HTML profiling reports with optional OLS residual outlier diagnostics."""

    SCHEMA_MAP = {
        "string": "text",
        "int": "numeric",
        "float": "numeric",
        "bool": "boolean",
        "category": "categorical",
        "datetime": "datetime",
    }

    def __init__(
        self,
        csv_path: Path | str,
        output_path: Path | str | None = None,
        column_type_specs: Sequence[str] = (),
        target: str | None = None,
        log_target: bool = False,
        outlier_z: float = 3.5,
        outlier_threshold: float | None = None,
        outlier_csv_path: Path | str | None = None,
        template_path: Path | str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.output_path = (
            Path(output_path)
            if output_path
            else self.csv_path.with_name(f"{self.csv_path.stem}_report.html")
        )
        self.outlier_csv_path = (
            Path(outlier_csv_path)
            if outlier_csv_path
            else self.csv_path.with_name(f"{self.csv_path.stem}_outliers.csv")
        )

        self.column_types, self.date_formats = ColumnSchemaParser.parse(
            column_type_specs
        )
        self.target = target
        self.log_target = log_target
        self.outlier_z = outlier_z
        self.outlier_threshold = outlier_threshold
        self.template_path = Path(template_path) if template_path else None

    def load_dataframe(self) -> pd.DataFrame:
        """Load the CSV file with explicit column dtypes and datetime parsing."""
        dtype_map = ColumnSchemaParser.to_pandas_dtypes(self.column_types)

        try:
            df = pd.read_csv(self.csv_path, dtype=dtype_map, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding="latin1")

        missing_columns = set(self.column_types) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Column(s) not found in CSV: {', '.join(sorted(missing_columns))}"
            )

        for column, kind in self.column_types.items():
            if kind == "datetime":
                df[column] = pd.to_datetime(
                    df[column],
                    format=self.date_formats.get(column),
                    errors="raise",
                )

        return df

    def build_profile_config(self, has_outliers: bool = False) -> dict:
        """Build ydata-profiling configuration for data quality diagnostics."""
        type_schema = {
            col: self.SCHEMA_MAP[kind] for col, kind in self.column_types.items()
        }
        if has_outliers:
            type_schema["residual_outlier"] = "boolean"

        interaction_targets = []
        if self.target:
            interaction_targets.append(self.target)
        if has_outliers:
            interaction_targets.append("residual_outlier")

        return {
            "title": f"Profile Report: {self.csv_path.name}",
            "explorative": True,
            "type_schema": type_schema,
            "vars": {
                "num": {
                    "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                    "skewness_threshold": 5,
                    "low_categorical_threshold": 5,
                },
                "cat": {
                    "length": True,
                    "characters": False,
                    "words": True,
                    "cardinality_threshold": 50,
                    "n_obs": 10,
                },
            },
            "correlations": {
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "cramers": {"calculate": True},
                "phi_k": {"calculate": True},
            },
            "missing_diagrams": {
                "bar": True,
                "matrix": True,
                "heatmap": True,
            },
            "duplicates": {
                "head": 10,
            },
            "interactions": {
                "continuous": True,
                "targets": interaction_targets,
            },
            "samples": {
                "head": 10,
                "tail": 10,
                "random": 10,
            },
        }

    def run_profile(self) -> dict:
        """Execute profiling, optional residual outlier analysis, and save artifacts."""
        df = self.load_dataframe()

        residual_html = ""
        has_outliers = False
        outlier_count = 0
        saved_outlier_csv: Path | None = None

        if self.target:
            analyzer = ResidualAnalyzer(
                target=self.target,
                z_threshold=self.outlier_z,
                raw_threshold=self.outlier_threshold,
                log_target=self.log_target,
                template_path=self.template_path,
            )
            analysis = analyzer.analyze(df)
            df["residual_outlier"] = analysis.outlier_series
            residual_html = analysis.html_section
            has_outliers = True
            outlier_count = analysis.outlier_count

            CSVFormatter.export_csv_no_scientific(df, self.outlier_csv_path)
            saved_outlier_csv = self.outlier_csv_path

        profile_kwargs = self.build_profile_config(has_outliers=has_outliers)
        report = ProfileReport(df, **profile_kwargs)
        html_content = report.to_html()

        if residual_html:
            if "</body>" in html_content:
                html_content = html_content.replace(
                    "</body>", f"{residual_html} </body>", 1
                )
            else:
                html_content += residual_html

        self.output_path.write_text(html_content, encoding="utf-8")

        return {
            "report_path": self.output_path,
            "outlier_csv_path": saved_outlier_csv,
            "outlier_count": outlier_count,
            "has_outliers": has_outliers,
        }
