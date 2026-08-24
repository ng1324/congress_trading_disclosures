"""Find extracted PTR JSON files whose transaction count doesn't match the
number of transactions actually listed, and optionally clear them out so the
extraction stage will redo them.

Nothing runs on import, and nothing is ever deleted unless you ask twice.

    from pathlib import Path
    import audit_transactions as at

    base = Path(r"C:\\Users\\User\\Desktop\\congress trades\\congress_disclosure_downloads\\fillings\\json")

    # 1. look first
    report = at.find_mismatches(base, [2023, 2024, 2025])
    print(report[report["Status"] != "OK"])

    # 2. dry run: shows what would happen, touches nothing
    at.resolve(report, action="move", quarantine_dir=base.parent / "quarantine")

    # 3. do it for real
    at.resolve(report, action="move", quarantine_dir=base.parent / "quarantine",
               dry_run=False)

Deleting or moving a JSON makes gemini_extract.pending_pdfs() see that PDF as
unprocessed, so the next extract_many() run re-does exactly those filings.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

STATUS_OK = "OK"
STATUS_MISMATCH = "Mismatch"
STATUS_NO_COUNT = "No count stated"
STATUS_UNREADABLE = "Unreadable"

# What resolve() acts on unless told otherwise. Deliberately excludes OK.
DEFAULT_TARGET_STATUSES = (STATUS_MISMATCH,)


def check_file(path) -> dict:
    """Compare 'Number of Transactions' against the Transactions array."""
    path = Path(path)
    row = {
        "Source File": path.stem,
        "Path": str(path),
        "Stated": None,
        "Listed": None,
        "Difference": None,
        "Status": STATUS_OK,
        "Detail": "",
    }

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        row["Status"] = STATUS_UNREADABLE
        row["Detail"] = f"{type(exc).__name__}: {exc}"
        return row

    if not isinstance(data, dict):
        row["Status"] = STATUS_UNREADABLE
        row["Detail"] = f"Top level is {type(data).__name__}, expected object"
        return row

    transactions = data.get("Transactions")
    listed = len(transactions) if isinstance(transactions, list) else 0
    row["Listed"] = listed

    stated = data.get("Number of Transactions")
    if not isinstance(stated, int) or isinstance(stated, bool):
        row["Status"] = STATUS_NO_COUNT
        row["Detail"] = f"'Number of Transactions' is {stated!r}"
        return row

    row["Stated"] = stated
    row["Difference"] = listed - stated
    if stated != listed:
        row["Status"] = STATUS_MISMATCH
        row["Detail"] = f"stated {stated}, listed {listed}"

    return row


def find_mismatches(json_root, years) -> pd.DataFrame:
    """Check every JSON under json_root/<year>/. Returns one row per file."""
    json_root = Path(json_root)
    rows = []

    for year in years:
        year_dir = json_root / str(year)
        if not year_dir.is_dir():
            LOGGER.warning("No folder for %s at %s", year, year_dir)
            continue
        for path in sorted(year_dir.glob("*.json")):
            row = check_file(path)
            row["Year"] = str(year)
            rows.append(row)

    if not rows:
        LOGGER.warning("No JSON files found")
        return pd.DataFrame(
            columns=["Year", "Source File", "Path", "Stated", "Listed",
                     "Difference", "Status", "Detail"]
        )

    report = pd.DataFrame(rows)[
        ["Year", "Source File", "Path", "Stated", "Listed",
         "Difference", "Status", "Detail"]
    ]

    counts = report["Status"].value_counts().to_dict()
    LOGGER.info("Checked %d files: %s", len(report), counts)
    return report


def save_report(report: pd.DataFrame, out_path) -> Path:
    """Write the audit to Excel so you have a record of what was flagged."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_excel(out_path, index=False)
    return out_path


def resolve(
    report: pd.DataFrame,
    action: str = "report",
    statuses=DEFAULT_TARGET_STATUSES,
    quarantine_dir=None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Act on flagged files.

    action="report"  list them, change nothing (default)
    action="move"    relocate to quarantine_dir, recoverable  <- recommended
    action="delete"  unlink permanently, no recycle bin

    Both destructive actions additionally require dry_run=False. Returns the
    affected rows with an "Action" column describing what was done.
    """
    if action not in ("report", "move", "delete"):
        raise ValueError(f"Unknown action {action!r}")

    if report.empty:
        LOGGER.info("Nothing to resolve")
        return report

    targets = report[report["Status"].isin(statuses)].copy()
    if targets.empty:
        LOGGER.info("No files matched statuses %s", tuple(statuses))
        targets["Action"] = []
        return targets

    if action == "report" or dry_run:
        verb = "would be moved" if action == "move" else (
            "would be deleted" if action == "delete" else "flagged"
        )
        targets["Action"] = verb
        LOGGER.info(
            "%d file(s) %s. Pass dry_run=False to apply.", len(targets), verb
        )
        for path in targets["Path"]:
            LOGGER.info("  %s", path)
        return targets

    if action == "move":
        if quarantine_dir is None:
            raise ValueError("action='move' requires quarantine_dir")
        quarantine_dir = Path(quarantine_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_root = quarantine_dir / stamp
        destination_root.mkdir(parents=True, exist_ok=True)

    performed = []
    for _, row in targets.iterrows():
        path = Path(row["Path"])

        # Guard against acting on anything that isn't one of our JSON files
        if path.suffix.lower() != ".json" or not path.is_file():
            performed.append(f"skipped (not a .json file): {path}")
            continue

        try:
            if action == "move":
                destination = destination_root / str(row["Year"]) / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
                performed.append(f"moved -> {destination}")
            else:
                path.unlink()
                performed.append("deleted")
        except OSError as exc:
            performed.append(f"failed: {exc}")

    targets["Action"] = performed
    done = sum(1 for a in performed if a.startswith(("moved", "deleted")))
    LOGGER.info("%s %d of %d file(s)", action.title() + "d", done, len(targets))
    return targets


def audit(
    json_root,
    years,
    action: str = "report",
    statuses=DEFAULT_TARGET_STATUSES,
    quarantine_dir=None,
    dry_run: bool = True,
    report_path=None,
):
    """Scan, optionally save the report, then resolve. Returns (report, acted)."""
    report = find_mismatches(json_root, years)
    if report_path:
        save_report(report, report_path)
    acted = resolve(
        report,
        action=action,
        statuses=statuses,
        quarantine_dir=quarantine_dir,
        dry_run=dry_run,
    )
    return report, acted
