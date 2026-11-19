#!/usr/bin/env python

import sys
import click
from CSVDiffer import CSVDiffer
from CSVProfiler import CSVProfiler


@click.group()
def cli():
    """CLI toolkit for diffing and profiling CSV datasets."""
    pass


@cli.command("CSVDiff")
@click.option(
    "-c",
    "--current",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to 'current.csv'.",
)
@click.option(
    "-n",
    "--new",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to 'new.csv'.",
)
@click.option(
    "-a",
    "--added",
    default="added.csv",
    show_default=True,
    help="Output path for added rows.",
)
@click.option(
    "-u",
    "--updated",
    default="updated.csv",
    show_default=True,
    help="Output path for changed rows.",
)
@click.option(
    "-m",
    "--merged",
    default="merged.csv",
    show_default=True,
    help="Output path for merged rows.",
)
def diff_command(current, new, added, updated, merged):
    """App tracking cell deltas and structural column differences between CSV files."""
    click.echo("Initializing diff engine...")

    try:
        differ = CSVDiffer(
            current_path=current,
            new_path=new,
            added_path=added,
            updated_path=updated,
            merged_path=merged,
        )

        added_count, updated_count, header_diffs = differ.run_diff()

        click.secho("\n✨ Execution Successful!", fg="green", bold=True)
        click.echo(f" 📂 Added rows:   {added_count} -> Saved to {added}")
        click.echo(f" 📂 Updated rows: {updated_count} -> Saved to {updated}")
        click.echo(f" 📂 Fully merged rows were saved to: {merged}")

        has_header_issues = (
            header_diffs["missing_in_new"]
            or header_diffs["missing_in_current"]
            or header_diffs["order_mismatch"]
        )

        if has_header_issues:
            click.secho(
                "\n⚠️  Header Mismatch Warnings Detected:", fg="yellow", bold=True
            )

            if header_diffs["missing_in_new"]:
                click.echo(
                    f"   - Removed Columns (In current but missing from new): {header_diffs['missing_in_new']}"
                )

            if header_diffs["missing_in_current"]:
                click.echo(
                    f"   - New Columns (In new but missing from current): {header_diffs['missing_in_current']}"
                )

            if header_diffs["order_mismatch"]:
                click.echo(
                    "   - Order Mismatch: Both files share identical column names, but their index positioning differs."
                )
        else:
            click.secho("\n✅ Headers match perfectly across both files.", fg="green")

    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg="red", bold=True)
        sys.exit(1)


@cli.command("CSVProfile")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="HTML report path (default: <input-name>_report.html).",
)
@click.option(
    "--column-type",
    multiple=True,
    metavar="COLUMN=TYPE",
    help=(
        "Set a column type. Repeat as needed. Types: string, int, float, "
        "bool, category, datetime. Datetime may include a format, e.g. "
        "order_date=datetime:%d/%m/%Y."
    ),
)
@click.option(
    "--target",
    type=str,
    default=None,
    help="Numeric target column to run regression, flag residual outliers, and export labeled CSV.",
)
@click.option(
    "--log-target",
    is_flag=True,
    default=False,
    help="Apply log1p(target) transformation before fitting regression and flagging outliers.",
)
@click.option(
    "--outlier-z",
    type=float,
    default=3.5,
    show_default=True,
    help="Modified Z-score threshold for flagging residual outliers.",
)
@click.option(
    "--outlier-threshold",
    type=float,
    default=None,
    help="Optional absolute residual cutoff (overrides --outlier-z, e.g. 500).",
)
@click.option(
    "--outlier-csv",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output path for labeled CSV (default: <input-name>_outliers.csv).",
)
@click.option(
    "--template",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Custom HTML template path for residual analysis section.",
)
def profile_command(
    csv_file,
    output,
    column_type,
    target,
    log_target,
    outlier_z,
    outlier_threshold,
    outlier_csv,
    template,
):
    """Create an HTML profiling report from a CSV file with optional residual outlier detection."""
    click.echo("Initializing CSV profiler...")

    try:
        profiler = CSVProfiler(
            csv_path=csv_file,
            output_path=output,
            column_type_specs=column_type,
            target=target,
            log_target=log_target,
            outlier_z=outlier_z,
            outlier_threshold=outlier_threshold,
            outlier_csv_path=outlier_csv,
            template_path=template,
        )

        result = profiler.run_profile()

        click.secho("\n✨ Profiling Complete!", fg="green", bold=True)
        click.echo(f" 📊 HTML report saved to: {result['report_path']}")

        if result["has_outliers"]:
            click.echo(
                f" 🚨 Flagged outliers:    {result['outlier_count']} -> Labeled CSV saved to: {result['outlier_csv_path']}"
            )

    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg="red", bold=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
