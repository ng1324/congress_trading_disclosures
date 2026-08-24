"""Congress trades pipeline: one entry point for all four stages.

Nothing runs on import. In Jupyter:

    import main
    P = main.paths(2026)          # see the folder layout for a year
    main.stage1_download(2026)    # index + fetch PTR PDFs
    main.stage2_extract(2026)     # PDFs -> JSON via Gemini
    main.stage3_audit(2026)       # find bad extractions
    main.stage4_excel([2026])     # JSON -> workbook

Or the whole thing:

    main.run_all([2025, 2026])

Every stage is resumable. Re-running skips work already done, so an
interrupted run is picked up by calling the same function again.

Everything is written next to this file — no paths to configure. The layout,
relative to the folder holding main.py and your notebook:

    ./
      index/     2026_list.xlsx   filing index + download status
      temp/      2026/2026FD.xml  extracted from the Clerk's zip
      downloads/ 2026/*.pdf       PTR PDFs
      json/      2026/*.json      one .json per extracted PDF
      quarantine/                 bad extractions moved aside, timestamped
      reports/                    audit reports + the consolidated workbook
"""

from __future__ import annotations

import logging
from pathlib import Path

# The four stage modules. Rename here if your filenames differ.
import congress_disclosures_main as hd
import process_docs as gx
import audit_transactions as at
import json_to_excel as jx

# --------------------------------------------------------------------------
# Settings — edit these
# --------------------------------------------------------------------------

# Everything lives beside this file, so the project folder can be moved or
# copied to another machine without editing anything. Resolved from __file__
# rather than the working directory, which in Jupyter is wherever the kernel
# happened to start and can be changed by an errant os.chdir().
BASE_DIR = Path(__file__).resolve().parent

# Filing types to download. "P" is the Periodic Transaction Report, the only
# type that lists trades. "C" candidate reports and "X" extensions have none.
FILING_TYPES = ("P",)

MODEL = "gemini-2.5-flash"

# Where API keys come from. Either set the GEMINI_API_KEYS environment
# variable (comma-separated) or paste them into API_KEYS below.
API_KEYS: list[str] = []

LOG_LEVEL = logging.INFO


def setup_logging(level=None) -> None:
    """Send progress to the notebook output. Safe to call repeatedly."""
    logging.basicConfig(
        level=level if level is not None else LOG_LEVEL,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def get_api_keys() -> list[str]:
    """API_KEYS if filled in, otherwise the GEMINI_API_KEYS env var."""
    if API_KEYS:
        return list(API_KEYS)
    return gx.load_api_keys()  # raises with a clear message if unset


# --------------------------------------------------------------------------
# Folder layout
# --------------------------------------------------------------------------

class Paths:
    """Every path the pipeline uses for one year, relative to main.py."""

    def __init__(self, year: int, base=None):
        base = Path(base) if base is not None else BASE_DIR
        self.year = int(year)
        self.base = base
        self.index = base / "index" / f"{year}_list.xlsx"
        self.temp = base / "temp" / str(year)
        self.downloads = base / "downloads" / str(year)
        self.json = base / "json" / str(year)
        # Shared across years. The audit and Excel stages walk json_root/<year>.
        self.json_root = base / "json"
        self.quarantine = base / "quarantine"
        self.reports = base / "reports"

    def __repr__(self) -> str:
        return (
            f"Paths(year={self.year})\n"
            f"  index     {self.index}\n"
            f"  downloads {self.downloads}\n"
            f"  json      {self.json}\n"
            f"  reports   {self.reports}"
        )


def paths(year: int, base=None) -> Paths:
    """Paths for one year. Call this to see where things will land.

    base defaults to the folder containing main.py. Pass one only if you
    keep the data somewhere other than beside the code.
    """
    return Paths(year, base)


def _base_for(base=None) -> Path:
    return Path(base) if base is not None else BASE_DIR


# --------------------------------------------------------------------------
# Stage 1: index the year and download PTR PDFs
# --------------------------------------------------------------------------

def stage1_download(
    year: int,
    base=None,
    filing_types=FILING_TYPES,
    refresh_index: bool = False,
    limit: int | None = None,
):
    """Fetch the Clerk's archive, build the index, download the PDFs.

    limit caps how many PDFs to fetch — use a small number for a first run.
    refresh_index=True re-downloads the archive and rebuilds the index,
    which you want when new filings have been posted.

    Returns the index DataFrame.
    """
    setup_logging()
    P = paths(year, base)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Stage 1: %s filings for %s", filing_types, year)

    if P.index.exists() and not refresh_index:
        df = hd.load_index(P.index)
        LOGGER.info("Using existing index (%d rows) at %s", len(df), P.index)
    else:
        xml_path = hd.download_disclosure_list(year, P.temp.parent)
        df = hd.build_disclosure_index(xml_path, year)
        hd.save_index(df, P.index)
        LOGGER.info("Indexed %d filings -> %s", len(df), P.index)

    try:
        df = hd.download_filings(
            df,
            P.downloads,
            filing_types=filing_types,
            checkpoint_path=P.index,
            limit=limit,
        )
    finally:
        hd.save_index(df, P.index)  # keep progress even on Ctrl-C

    counts = df.loc[df["FilingType"].isin(filing_types), "Download Status"].value_counts()
    LOGGER.info("Download status: %s", counts.to_dict())
    return df


# --------------------------------------------------------------------------
# Stage 2: extract PDFs to JSON with Gemini
# --------------------------------------------------------------------------

def stage2_extract(
    year: int,
    base=None,
    api_keys=None,
    model: str = MODEL,
    limit: int | None = None,
    start: int | None = None,
    end: int | None = None,
    file_types=(".pdf",),
    skip_existing: bool = True,
    on_exhausted: str = "raise",
):
    """Send each downloaded PDF to Gemini and save the structured JSON.

    limit caps the run; start/end take a slice of the file listing. Files
    that already have JSON are skipped unless skip_existing=False.

    Raises gx.QuotaExhausted if every API key runs out mid-run — the JSON
    written so far is intact, so re-run once quota resets.
    """
    setup_logging()
    P = paths(year, base)
    LOGGER = logging.getLogger(__name__)

    if not P.downloads.is_dir():
        raise FileNotFoundError(
            f"No downloads folder at {P.downloads}. Run stage1_download({year}) first."
        )

    keys = list(api_keys) if api_keys else get_api_keys()
    outstanding = gx.pending_files(
        P.downloads, P.json, skip_existing=skip_existing, file_types=file_types
    )
    LOGGER.info(
        "Stage 2: %d file(s) outstanding, %d key(s), model %s",
        len(outstanding), len(keys), model,
    )

    return gx.extract_many(
        source_dir=P.downloads,
        json_dir=P.json,
        api_keys=keys,
        model=model,
        start=start,
        end=end,
        skip_existing=skip_existing,
        file_types=file_types,
        limit=limit,
        on_exhausted=on_exhausted,
    )


# --------------------------------------------------------------------------
# Stage 3: audit the extractions
# --------------------------------------------------------------------------

def stage3_audit(
    years,
    base=None,
    action: str = "report",
    dry_run: bool = True,
    statuses=None,
):
    """Flag JSON whose stated transaction count doesn't match the list.

    action="report" (default) changes nothing. action="move" quarantines the
    bad files so stage 2 will redo them; it also needs dry_run=False.

    Returns (report, acted) DataFrames.
    """
    setup_logging()
    years = [years] if isinstance(years, int) else list(years)
    root = _base_for(base)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if statuses is not None:
        kwargs["statuses"] = statuses

    return at.audit(
        root / "json",
        years,
        action=action,
        quarantine_dir=root / "quarantine",
        dry_run=dry_run,
        report_path=reports_dir / "audit_report.xlsx",
        **kwargs,
    )


# --------------------------------------------------------------------------
# Stage 4: consolidate JSON into one workbook
# --------------------------------------------------------------------------

def stage4_excel(years, base=None, out_path=None):
    """Flatten every JSON for these years into a multi-sheet workbook.

    Returns the dict of DataFrames that was written.
    """
    setup_logging()
    years = [years] if isinstance(years, int) else list(years)
    root = _base_for(base)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        span = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        out_path = reports_dir / f"ptr_transactions_{span}.xlsx"

    return jx.build_excel(
        json_root=root / "json",
        years=years,
        out_path=out_path,
    )


# --------------------------------------------------------------------------
# Everything, in order
# --------------------------------------------------------------------------

def run_all(
    years,
    base=None,
    api_keys=None,
    model: str = MODEL,
    download_limit: int | None = None,
    extract_limit: int | None = None,
    refresh_index: bool = False,
):
    """Stages 1-2 for each year, then 3-4 across all of them.

    Stops on the first error rather than pressing on with bad data. If a
    quota runs out mid-extraction, the audit and Excel stages still run over
    whatever was extracted.
    """
    setup_logging()
    years = [years] if isinstance(years, int) else list(years)
    LOGGER = logging.getLogger(__name__)
    summary = {}

    for year in years:
        LOGGER.info("=" * 60)
        LOGGER.info("Year %s", year)
        LOGGER.info("=" * 60)

        stage1_download(
            year, base, refresh_index=refresh_index, limit=download_limit
        )
        try:
            summary[year] = stage2_extract(
                year, base, api_keys=api_keys, model=model, limit=extract_limit
            )
        except gx.QuotaExhausted as exc:
            LOGGER.error("%s", exc)
            LOGGER.error("Stopping extraction; continuing to audit and Excel.")
            summary[year] = exc.results
            break

    LOGGER.info("=" * 60)
    report, _ = stage3_audit(years, base)
    flagged = report[report["Status"] != at.STATUS_OK] if not report.empty else report
    if len(flagged):
        LOGGER.warning(
            "%d file(s) flagged by the audit; see the Problems sheet", len(flagged)
        )

    tables = stage4_excel(years, base)
    return {"extraction": summary, "audit": report, "tables": tables}


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def status(years, base=None):
    """How far along each year is. Reads the filesystem, changes nothing."""
    import pandas as pd

    years = [years] if isinstance(years, int) else list(years)
    rows = []
    for year in years:
        P = paths(year, base)
        pdfs = len(list(P.downloads.glob("*.pdf"))) if P.downloads.is_dir() else 0
        jsons = len(list(P.json.glob("*.json"))) if P.json.is_dir() else 0
        rows.append(
            {
                "Year": year,
                "Index built": P.index.exists(),
                "PDFs downloaded": pdfs,
                "JSON extracted": jsons,
                "Outstanding": max(pdfs - jsons, 0),
            }
        )
    return pd.DataFrame(rows)