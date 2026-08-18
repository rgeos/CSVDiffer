import csv
import os


class CSVDiffer:
    """A class to compare two CSV files, track header mismatches, and extract row diffs."""

    def __init__(
        self, current_path, new_path, added_path="added.csv", updated_path="updated.csv"
    ):
        self.current_path = current_path
        self.new_path = new_path
        self.added_path = added_path
        self.updated_path = updated_path
        self.headers = []
        self.added_rows = []
        self.updated_rows = []
        self.header_diffs = {
            "missing_in_new": [],
            "missing_in_current": [],
            "order_mismatch": False,
        }

    def validate_files(self):
        """Validates that input files exist on disk."""
        if not os.path.exists(self.current_path):
            raise FileNotFoundError(f"Current file not found: {self.current_path}")
        if not os.path.exists(self.new_path):
            raise FileNotFoundError(f"New file not found: {self.new_path}")

    def run_diff(self):
        """Executes the pipeline and analyzes structural header integrity."""
        self.validate_files()
        current_data = self._load_current_data()
        self._process_new_data(current_data)
        self._write_results()

        return len(self.added_rows), len(self.updated_rows), self.header_diffs

    @staticmethod
    def _verify_duplicate_headers(headers, file_label="CSV"):
        """
        Raise ValueError if duplicate headers exist (case-insensitive).
        Treats headers as duplicates if they match after stripping whitespace
        and lowercasing.
        """
        if not headers:
            return

        normalized = [(h.strip().lower() if h is not None else "") for h in headers]

        # Count duplicates
        duplicates = sorted(
            {h for h in normalized if normalized.count(h) > 1 if h != ""}
        )

        if duplicates:
            raise ValueError(
                f"Duplicate headers found in {file_label} file: {duplicates}"
            )

    def _load_current_data(self):
        """Reads current.csv headers and records row configurations."""
        current_data = {}
        with open(self.current_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                self.headers = next(reader)
                self._verify_duplicate_headers(self.headers, file_label="current")
            except StopIteration:
                raise ValueError(
                    f"The file {self.current_path} is empty or missing headers."
                )

            for row in reader:
                if row:
                    current_data[row[0]] = row
        return current_data

    def _process_new_data(self, current_data):
        """Compares new data against old data and validates column naming uniformity."""
        with open(self.new_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                new_headers = next(reader)
                self._verify_duplicate_headers(new_headers, file_label="new")
            except StopIteration:
                raise ValueError(
                    f"The file {self.new_path} is empty or missing headers."
                )

            # Check if first header in both files is the same
            if not self.headers or not new_headers:
                raise ValueError("One or both CSV files are missing header data.")
            current_first_row = (self.headers[0] or "").strip()
            new_first_row = (new_headers[0] or "").strip()

            if current_first_row != new_first_row:
                raise ValueError(
                    f"Header mismatch: first header differs "
                    f"(current: {current_first_row!r}, new: {new_first_row!r})."
                )

            # Validate column discrepancies
            self._check_header_differences(self.headers, new_headers)

            for row in reader:
                if not row:
                    continue

                key = row[0]

                if key not in current_data:
                    self.added_rows.append(row)
                else:
                    current_row = current_data[key]
                    has_diff = False
                    updated_row = [key]

                    # Loop through remaining columns up to the maximum common safe length
                    max_cols = max(len(row), len(current_row))
                    for i in range(1, max_cols):
                        curr_val = current_row[i] if i < len(current_row) else ""
                        new_val = row[i] if i < len(row) else ""

                        if curr_val != new_val:
                            has_diff = True
                            updated_row.append(f"{curr_val} -> {new_val}")
                        else:
                            updated_row.append(new_val)

                    if has_diff:
                        self.updated_rows.append(updated_row)

    def _check_header_differences(self, current_hdrs, new_hdrs):
        """Analyzes column name drops, structural additions, and positioning mismatches."""
        clean_current = [h.strip() for h in current_hdrs if h]
        clean_new = [h.strip() for h in new_hdrs if h]

        curr_set = set(clean_current)
        new_set = set(clean_new)

        self.header_diffs["missing_in_new"] = list(curr_set - new_set)
        self.header_diffs["missing_in_current"] = list(new_set - curr_set)

        # Check order mismatch only if the column item lists are identical
        if (
            not self.header_diffs["missing_in_new"]
            and not self.header_diffs["missing_in_current"]
        ):
            if current_hdrs != new_hdrs:
                self.header_diffs["order_mismatch"] = True

    def _write_results(self):
        """Persists structural rows output cleanly."""
        with open(self.added_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.added_rows)

        with open(self.updated_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.updated_rows)
