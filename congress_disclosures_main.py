"""U.S. House financial disclosures: index a year's filings and fetch the PDFs.

Nothing runs on import. Call the functions yourself, e.g.

    from pathlib import Path
    import house_disclosures as hd

    year = 2026
    work_dir = Path(r"C:\\Users\\User\\Desktop\\congress trades\\congress_disclosure_downloads")

    # --- stage 1: get the XML -------------------------------------------
    xml_path = hd.download_disclosure_list(year, work_dir / "temp")

    # --- stage 2: build the index ---------------------------------------
    df = hd.build_disclosure_index(xml_path, year)
    hd.save_index(df, work_dir / f"{year}_list.xlsx")

    # --- stage 3: download the PTR PDFs ---------------------------------
    df = hd.load_index(work_dir / f"{year}_list.xlsx")          # resume later
    df = hd.download_filings(df, work_dir / "downloads" / str(year))
    hd.save_index(df, work_dir / f"{year}_list.xlsx")

Or do all three with hd.run_pipeline(year, work_dir).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ARCHIVE_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
OTHER_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}/{doc_id}.pdf"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

MAX_REQUESTS_PER_SECOND = 2.0
NUM_WORKERS = 4
CHUNK_SIZE = 8192
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = (10, 60)  # (connect, read)

STATUS_PENDING = "Incomplete"
STATUS_COMPLETE = "Complete"
STATUS_MISSING = "Missing (404)"
ERROR_PREFIX = "_###ERROR"

DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d")
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

# House PDFs are small, but anything under this is almost certainly a
# truncated download or an HTML error page saved with a .pdf extension.
MIN_PDF_BYTES = 1000


# --------------------------------------------------------------------------
# Stage 1: fetch and unpack the year's archive
# --------------------------------------------------------------------------

def download_disclosure_list(year: int, dest_dir, timeout: int = 60) -> Path:
    """Download and extract {year}FD.zip. Returns the extracted XML path.

    Members are extracted under the names the archive gives them (currently
    {year}FD.xml and {year}FD.txt); nothing is renamed.
    """
    dest_dir = Path(dest_dir) / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        ARCHIVE_URL.format(year=year), headers=HEADERS, timeout=timeout
    )
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = archive.namelist()
        for name in names:
            target = (dest_dir / name).resolve()
            if not target.is_relative_to(dest_dir.resolve()):
                raise ValueError(f"Refusing to extract outside dest_dir: {name}")
        archive.extractall(dest_dir)
    LOGGER.info("Extracted %s to %s", ", ".join(names), dest_dir)

    # Prefer the conventional name, but fall back to whatever XML is inside
    # so a change on the Clerk's end doesn't break this.
    xml_names = [n for n in names if n.lower().endswith(".xml")]
    preferred = f"{year}FD.xml"
    chosen = preferred if preferred in xml_names else (xml_names[0] if xml_names else None)
    if chosen is None:
        raise FileNotFoundError(f"No .xml in the archive, found: {names}")

    return dest_dir / chosen


# --------------------------------------------------------------------------
# Stage 2: turn the XML into an index
# --------------------------------------------------------------------------

def _safe_component(value: str) -> str:
    """Make one filename component safe on Windows and POSIX."""
    cleaned = _ILLEGAL.sub("", (value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "unknown"


def parse_filing_date(raw: str):
    """Return YYYYMMDD, or None if the date is missing or unparseable."""
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    LOGGER.warning("Unrecognised FilingDate %r", raw)
    return None


def pdf_url(filing_type: str, year, doc_id: str) -> str:
    """PTRs live under a different path from every other filing type."""
    template = PTR_URL if filing_type == "P" else OTHER_URL
    return template.format(year=year, doc_id=doc_id)


def build_disclosure_index(xml_path, fallback_year: int) -> pd.DataFrame:
    """Parse the disclosure XML into a DataFrame with FileName and URL columns."""
    root = ET.parse(Path(xml_path)).getroot()

    records = []
    for member in root.findall("Member"):
        record = {child.tag: (child.text or "").strip() for child in member}

        year = record.get("Year") or str(fallback_year)
        date_str = parse_filing_date(record.get("FilingDate", "")) or year

        record["FileName"] = "_".join(
            _safe_component(part)
            for part in (
                date_str,
                record.get("FilingType", ""),
                record.get("First", ""),
                record.get("Last", ""),
                record.get("DocID", ""),
            )
        )
        record["URL"] = pdf_url(
            record.get("FilingType", ""), year, record.get("DocID", "")
        )
        records.append(record)

    if not records:
        raise ValueError(f"No <Member> elements found in {xml_path}")

    df = pd.DataFrame(records)
    df["Download Status"] = STATUS_PENDING

    duplicates = int(df["FileName"].duplicated().sum())
    if duplicates:
        LOGGER.warning("%d duplicate FileName values", duplicates)

    return df


def save_index(df: pd.DataFrame, index_path) -> None:
    """Write the index to Excel, creating parent folders as needed."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(index_path, index=False)


def load_index(index_path) -> pd.DataFrame:
    """Read the index back as strings, so DocIDs never become integers."""
    return pd.read_excel(Path(index_path), keep_default_na=False, dtype=str)


# --------------------------------------------------------------------------
# Stage 3: download the PDFs
# --------------------------------------------------------------------------

class RateLimiter:
    """Caps requests per second across every worker thread, not per thread."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            time.sleep(wait)


_thread_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread: connection reuse without sharing state."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return session


def fetch_one(url: str, dest, limiter: "RateLimiter | None" = None) -> str:
    """Download one PDF. Returns the status string to record. Usable alone."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= MIN_PDF_BYTES:
        return STATUS_COMPLETE

    partial = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            response = _session().get(url, stream=True, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                response.close()
                return STATUS_MISSING  # no PDF for this DocID; don't retry
            if response.status_code in (403, 429) or response.status_code >= 500:
                code = response.status_code
                response.close()
                if attempt == MAX_ATTEMPTS:
                    return f"{ERROR_PREFIX} HTTP {code}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()

            with response, open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)

            if partial.stat().st_size < MIN_PDF_BYTES:
                partial.unlink(missing_ok=True)
                raise OSError("suspiciously small response")

            os.replace(partial, dest)  # atomic: dest exists only when complete
            return STATUS_COMPLETE

        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS:
                return f"{ERROR_PREFIX} {type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)

    return f"{ERROR_PREFIX} exhausted retries"


def pending_rows(df: pd.DataFrame, filing_types=("P",)):
    """Index labels still needing a download: never fetched, or errored."""
    status = df["Download Status"].fillna(STATUS_PENDING).astype(str)
    return df.index[
        df["FilingType"].isin(filing_types)
        & ((status == STATUS_PENDING) | status.str.contains(ERROR_PREFIX, regex=False))
    ]


def download_filings(
    df: pd.DataFrame,
    output_dir,
    filing_types=("P",),
    checkpoint_path=None,
    checkpoint_every: int = 100,
    limit: "int | None" = None,
) -> pd.DataFrame:
    """Download PDFs for the given filing types, skipping rows already done.

    Mutates and returns df. Pass checkpoint_path to save progress periodically,
    and limit to try a small batch first.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = pending_rows(df, filing_types)
    if limit is not None:
        todo = todo[:limit]

    if len(todo) == 0:
        LOGGER.info("Nothing to download")
        return df

    limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(
                fetch_one,
                df.at[i, "URL"],
                output_dir / f"{df.at[i, 'FileName']}.pdf",
                limiter,
            ): i
            for i in todo
        }

        with tqdm(total=len(futures), unit="file", desc="Downloading") as bar:
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()
                df.at[i, "Download Status"] = result  # main thread only
                if result.startswith(ERROR_PREFIX):
                    tqdm.write(f"{df.at[i, 'FileName']}: {result}")

                completed += 1
                bar.update(1)
                if checkpoint_path and completed % checkpoint_every == 0:
                    save_index(df, checkpoint_path)

    if checkpoint_path:
        save_index(df, checkpoint_path)
    return df


# --------------------------------------------------------------------------
# Optional convenience wrapper
# --------------------------------------------------------------------------

def run_pipeline(
    year: int,
    work_dir,
    filing_types=("P",),
    refresh_index: bool = False,
    limit: "int | None" = None,
) -> pd.DataFrame:
    """All three stages. Reuses an existing index unless refresh_index=True."""
    work_dir = Path(work_dir) / str(year)
    index_path = work_dir / f"{year}_list.xlsx"

    if index_path.exists() and not refresh_index:
        df = load_index(index_path)
        LOGGER.info("Resuming from %s (%d rows)", index_path, len(df))
    else:
        xml_path = download_disclosure_list(year, work_dir)
        df = build_disclosure_index(xml_path, year)
        save_index(df, index_path)
        LOGGER.info("Indexed %d filings", len(df))

    try:
        df = download_filings(
            df,
            work_dir / "downloads",
            filing_types=filing_types,
            checkpoint_path=index_path,
            limit=limit,
        )
    finally:
        save_index(df, index_path)  # persist progress even on Ctrl-C

    return df