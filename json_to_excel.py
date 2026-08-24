"""Consolidate extracted PTR JSON files into a single Excel workbook.

Reads only the JSON files themselves — nothing is cross-referenced against
the FD index or the source PDFs. Nothing runs on import:

    from pathlib import Path
    import json_to_excel as jx

    base = Path(r"C:\\Users\\User\\Desktop\\congress trades\\congress_disclosure_downloads\\fillings\\json")

    tables = jx.build_excel(
        json_root=base,
        years=[2023, 2024, 2025],
        out_path=base.parent / "ptr_transactions.xlsx",
    )

Produces one sheet per level of the schema:

    Filings              one row per JSON file
    Transactions         one row per transaction
    Investment Vehicles  one row per vehicle/asset-class entry
    Exercise Details     one row per exercised-option detail
    Problems             files that failed to load, and count mismatches
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y")

# Excel refuses control characters and caps a cell at 32,767 characters.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MAX_CELL_CHARS = 32000

FILING_FIELDS = [
    "Filing ID",
    "Name",
    "Status",
    "State/District",
    "IPO",
    "Digitally Signed Date",
    "Number of Transactions",
    "Other Information",
]

TRANSACTION_FIELDS = [
    "ID",
    "Owner",
    "Asset Name",
    "Asset Symbol",
    "Asset Type",
    "Transaction Type",
    "Partial Transaction",
    "Number of Assets Transacted",
    "Date",
    "Notification Date",
    "Amount",
    "Cap. Gains > $200?",
    "Filing Status",
    "Location",
    "Subholding of",
    "Description",
    "Options Type",
    "Options Strike Price",
    "Options Expiry Date",
    "Exercise Option",
    "Number of Options Exercised",
]

VEHICLE_FIELDS = ["Item", "Owner", "Location", "Description"]

EXERCISE_FIELDS = ["Date Purchased", "Option Strike Price", "Option Expiration Date"]


# --------------------------------------------------------------------------
# Value helpers
# --------------------------------------------------------------------------

def clean_cell(value):
    """Make a value safe for Excel: no control chars, no oversized strings."""
    if isinstance(value, str):
        value = _CONTROL_CHARS.sub("", value).strip()
        if len(value) > MAX_CELL_CHARS:
            value = value[:MAX_CELL_CHARS] + "…[truncated]"
        return value
    if isinstance(value, (list, dict)):
        # An unexpected nested blob: keep it readable rather than losing it
        return clean_cell(json.dumps(value, ensure_ascii=False))
    return value


def parse_date(raw):
    """Return a datetime for Excel date sorting, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_amount_range(raw):
    """Split '$1,001 - $15,000' into (1001.0, 15000.0).

    'Over $50,000,000' gives (50000000.0, None); a single figure gives the
    same value for both ends. Returns (None, None) when nothing parses.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, None

    numbers = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+(?:\.\d+)?", raw)]
    numbers = [n for n in numbers if n or n == 0]
    if not numbers:
        return None, None

    lowered = raw.lower()
    if "over" in lowered or "more than" in lowered:
        return numbers[0], None
    if "under" in lowered or "less than" in lowered:
        return None, numbers[0]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers), max(numbers)


def _get(record, key):
    """Fetch a key from a dict-ish record without exploding on bad input."""
    if not isinstance(record, dict):
        return None
    return clean_cell(record.get(key))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def iter_json_files(json_root, years):
    """Yield (year, path) for every .json under json_root/<year>/."""
    json_root = Path(json_root)
    for year in years:
        year_dir = json_root / str(year)
        if not year_dir.is_dir():
            LOGGER.warning("No folder for %s at %s", year, year_dir)
            continue
        for path in sorted(year_dir.glob("*.json")):
            yield str(year), path


def load_records(json_root, years):
    """Load every JSON. Returns (records, problems).

    records is a list of (year, source_stem, dict); problems collects files
    that could not be read at all.
    """
    records, problems = [], []

    for year, path in iter_json_files(json_root, years):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(
                {
                    "Year": year,
                    "Source File": path.stem,
                    "Issue": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if not isinstance(data, dict):
            problems.append(
                {
                    "Year": year,
                    "Source File": path.stem,
                    "Issue": f"Top level is {type(data).__name__}, expected object",
                }
            )
            continue

        records.append((year, path.stem, data))

    return records, problems


# --------------------------------------------------------------------------
# Table building
# --------------------------------------------------------------------------

def build_tables(records, problems=None):
    """Turn loaded records into the four sheets plus a problems sheet."""
    problems = list(problems or [])
    filings, transactions, vehicles, exercises = [], [], [], []

    for year, source, record in records:
        filing = {"Year": year, "Source File": source}
        filing.update({field: _get(record, field) for field in FILING_FIELDS})

        raw_transactions = record.get("Transactions")
        raw_transactions = raw_transactions if isinstance(raw_transactions, list) else []
        filing["Transactions Listed"] = len(raw_transactions)

        stated = record.get("Number of Transactions")
        if isinstance(stated, int) and stated != len(raw_transactions):
            problems.append(
                {
                    "Year": year,
                    "Source File": source,
                    "Issue": (
                        f"'Number of Transactions'={stated} but "
                        f"{len(raw_transactions)} listed"
                    ),
                }
            )

        filings.append(filing)

        # ---- investment vehicles ----------------------------------------
        raw_vehicles = record.get("Investment Vehicle Details")
        if isinstance(raw_vehicles, list):
            for position, vehicle in enumerate(raw_vehicles, start=1):
                row = {
                    "Year": year,
                    "Source File": source,
                    "Filing ID": filing.get("Filing ID"),
                    "Vehicle #": position,
                }
                row.update({field: _get(vehicle, field) for field in VEHICLE_FIELDS})
                vehicles.append(row)

        # ---- transactions ------------------------------------------------
        for position, transaction in enumerate(raw_transactions, start=1):
            row = {
                "Year": year,
                "Source File": source,
                "Filing ID": filing.get("Filing ID"),
                "Name": filing.get("Name"),
                "State/District": filing.get("State/District"),
                "Transaction #": position,
            }
            row.update(
                {field: _get(transaction, field) for field in TRANSACTION_FIELDS}
            )

            # Derived purely from the fields above, for sorting and sizing
            row["Date (parsed)"] = parse_date(row.get("Date"))
            row["Notification Date (parsed)"] = parse_date(row.get("Notification Date"))
            low, high = parse_amount_range(row.get("Amount"))
            row["Amount Min"] = low
            row["Amount Max"] = high

            transactions.append(row)

            # ---- exercised option details --------------------------------
            raw_details = (
                transaction.get("Exercise Option Details")
                if isinstance(transaction, dict)
                else None
            )
            if isinstance(raw_details, list):
                for detail_no, detail in enumerate(raw_details, start=1):
                    detail_row = {
                        "Year": year,
                        "Source File": source,
                        "Filing ID": filing.get("Filing ID"),
                        "Transaction #": position,
                        "Detail #": detail_no,
                        "Asset Name": row.get("Asset Name"),
                    }
                    detail_row.update(
                        {field: _get(detail, field) for field in EXERCISE_FIELDS}
                    )
                    exercises.append(detail_row)

    return {
        "Filings": pd.DataFrame(filings),
        "Transactions": pd.DataFrame(transactions),
        "Investment Vehicles": pd.DataFrame(vehicles),
        "Exercise Details": pd.DataFrame(exercises),
        "Problems": pd.DataFrame(problems),
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_workbook(tables, out_path, freeze_header: bool = True) -> Path:
    """Write each table to its own sheet, with sensible column widths."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            if frame.empty:
                # Keep the sheet so the workbook shape is predictable
                frame = pd.DataFrame({"(no rows)": []})
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            worksheet = writer.sheets[sheet_name[:31]]
            if freeze_header:
                worksheet.freeze_panes = "A2"
            for column_no, column in enumerate(frame.columns, start=1):
                sample = frame[column].astype(str).head(200)
                width = max(len(str(column)), *(sample.str.len().tolist() or [0]))
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=column_no).column_letter
                ].width = min(max(width + 2, 10), 60)

    return out_path


def build_excel(json_root, years, out_path):
    """Load, flatten and write in one call. Returns the tables dict."""
    records, problems = load_records(json_root, years)
    LOGGER.info("Loaded %d JSON files (%d unreadable)", len(records), len(problems))

    tables = build_tables(records, problems)
    write_workbook(tables, out_path)

    LOGGER.info(
        "Wrote %s: %d filings, %d transactions, %d problems",
        out_path,
        len(tables["Filings"]),
        len(tables["Transactions"]),
        len(tables["Problems"]),
    )
    return tables
