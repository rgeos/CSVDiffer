#!/usr/bin/env python

import click
import sys
from CSVDiffer import CSVDiffer


@click.command()
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
def main(current, new, added, updated, merged):
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


if __name__ == "__main__":
    main()
