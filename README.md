# CSVDiffer

A command-line utility to compare two CSV files and report differences.

## Features

- Compares rows between an “current/new” CSV and outputs the diff in two files
  - `updated.csv` lists all the rows that exist in both files but have been updated in the `new.csv`
  - `added.csv` list all the rows that exist in `new.csv` but not in `current.csv`
- Validates CSV header compatibility before performing the diff
  - Stops if the first header name in both files does not match
  - Stops if either CSV contains duplicate header names (case-insensitive)
- When differences are found, the tool prints a summary of the changes and exits normally.
- When header validation fails, the tool prints an error message and exits with a non-zero status.

## Requirements

- Python 3.8+ (recommended)
- Click

## Install

Clone the repository & install requirements

```bash
git clone https://github.com/rgeos/CSVDiffer.git
cd CSVDiffer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --current path/to/current.csv --new path/to/new.csv
```


