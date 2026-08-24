"""Extract structured data from House PTR PDFs with Gemini.

Nothing runs on import. Typical use:

    from pathlib import Path
    import gemini_extract as gx

    year = 2026
    base = Path(r"C:\\Users\\User\\Desktop\\congress trades\\congress_disclosure_downloads")

    keys = gx.load_api_keys()                    # from GEMINI_API_KEYS env var
    results = gx.extract_many(
        source_dir=base / "downloads" / str(year),
        json_dir=base / "json" / str(year),
        api_keys=keys,
        model="gemini-2.5-flash",
        limit=10,                                # try a small batch first
    )

Re-running skips any PDF that already has a JSON file, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import sys
import os
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
PROMPT_TEXT = "From the following document, extract the specified fields."

# Free-tier flash sits around 10 requests/minute per key; raise if you're on
# a paid tier. This paces each key independently.
REQUESTS_PER_MINUTE_PER_KEY = 10
MAX_ATTEMPTS = 4
BACKOFF_BASE = 5  # seconds; 5, 10, 20 between retries

# Inline PDF bytes are capped at ~20MB per request. House PTRs are far
# smaller, so anything near this is a sign something else went wrong.
MAX_INLINE_BYTES = 18 * 1024 * 1024
MIN_PDF_BYTES = 1000

# Extensions the extractor understands, and how each is sent to Gemini.
# PDFs go as binary parts; text formats are sent as text, which costs fewer
# tokens and sidesteps mime-type quibbles.
SUPPORTED_TYPES = {
    ".pdf": {"mime": "application/pdf", "binary": True, "min_bytes": MIN_PDF_BYTES},
    ".md": {"mime": "text/markdown", "binary": False, "min_bytes": 20},
    ".markdown": {"mime": "text/markdown", "binary": False, "min_bytes": 20},
    ".txt": {"mime": "text/plain", "binary": False, "min_bytes": 20},
}
DEFAULT_FILE_TYPES = (".pdf",)


OUTPUT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Filing ID": {"type": "string"},
        "Name": {"type": "string"},
        "Status": {"type": "string"},
        "State/District": {"type": "string"},
        "IPO": {"type": "string"},
        "Digitally Signed Date": {"type": "string"},
        "Investment Vehicle Details": {
            "type": "array",
            "description": "This can be 'Asset Class Details' in older documents",
            "items": {
                "type": "object",
                "properties": {
                    "Item": {"type": "string"},
                    "Owner": {"type": "string"},
                    "Location": {"type": "string"},
                    "Description": {"type": "string"},
                },
            },
        },
        "Number of Transactions": {"type": "integer"},
        "Transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ID": {"type": "string"},
                    "Owner": {"type": "string"},
                    "Asset Name": {"type": "string"},
                    "Asset Symbol": {"type": "string"},
                    "Asset Type": {
                        "type": "string",
                        "description": "This information can be found in the [] square bracket in Asset",
                    },
                    "Transaction Type": {"type": "string"},
                    "Partial Transaction": {"type": "boolean"},
                    "Number of Assets Transacted": {
                        "type": "integer",
                        "description": "The number of asset type that are bought or sold (i.e. shares, options, or etc.).",
                    },
                    "Date": {"type": "string"},
                    "Notification Date": {"type": "string"},
                    "Amount": {"type": "string"},
                    "Cap. Gains > $200?": {"type": "string"},
                    "Filing Status": {"type": "string"},
                    "Location": {"type": "string"},
                    "Subholding of": {"type": "string"},
                    "Description": {"type": "string"},
                    "Options Type": {
                        "type": "string",
                        "description": "Call or Put Option",
                    },
                    "Options Strike Price": {"type": "string"},
                    "Options Expiry Date": {"type": "string"},
                    "Exercise Option": {"type": "boolean"},
                    "Number of Options Exercised": {"type": "integer"},
                    "Exercise Option Details": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Date Purchased": {"type": "string"},
                                "Option Strike Price": {"type": "string"},
                                "Option Expiration Date": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "Other Information": {
            "type": "string",
            "description": "Other relavent information or note that were not covered in the above template",
        },
    },
}


class TruncatedResponse(RuntimeError):
    """The model stopped before finishing the JSON (usually MAX_TOKENS)."""


class QuotaExhausted(RuntimeError):
    """Every API key is spent; the run stopped early.

    The partial {stem: status} mapping is attached as .results, and the
    filenames still outstanding as .remaining.
    """

    def __init__(self, message, results=None, remaining=None):
        super().__init__(message)
        self.results = results or {}
        self.remaining = remaining or []


def load_api_keys(env_var: str = "GEMINI_API_KEYS") -> list[str]:
    """Read comma-separated keys from an env var, so they stay out of source."""
    raw = os.environ.get(env_var, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError(
            f"No API keys found in ${env_var}. Set it to a comma-separated list."
        )
    return keys


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        getattr(exc, "code", None) == 429
        or "429" in text
        or "RESOURCE_EXHAUSTED" in text
    )


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code >= 500:
        return True
    text = str(exc)
    return any(s in text for s in ("500", "502", "503", "504", "UNAVAILABLE", "DEADLINE"))


class RateLimiter:
    """Paces one key's requests. Each key gets its own instance."""

    def __init__(self, per_minute: float):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            time.sleep(wait)


class KeyHandle:
    """One API key with its own client and rate limiter."""

    __slots__ = ("key", "client", "limiter")

    def __init__(self, key, per_minute):
        self.key = key
        self.client = genai.Client(api_key=key)
        self.limiter = RateLimiter(per_minute)

    @property
    def masked(self) -> str:
        return f"...{self.key[-4:]}" if len(self.key) > 4 else "key"


class KeyPool:
    """Keys as a shared resource, so a dead key doesn't strand its backlog.

    A task checks out a key, uses it, and checks it back in. A key that is
    out of quota gets retired instead of returned, and the remaining keys
    absorb the outstanding work. When the last key retires, acquire() returns
    None and the run can wind down immediately.
    """

    def __init__(self, api_keys, per_minute: float = REQUESTS_PER_MINUTE_PER_KEY):
        keys = list(dict.fromkeys(api_keys))  # de-duplicate, keep order
        if not keys:
            raise ValueError("api_keys is empty")
        self._per_minute = per_minute
        self._pending = list(keys)
        self._idle: list[KeyHandle] = []
        self._live = len(keys)
        self._retired: list[tuple[str, str]] = []
        self._condition = threading.Condition()

    @property
    def live_keys(self) -> int:
        with self._condition:
            return self._live

    @property
    def retired_keys(self):
        with self._condition:
            return list(self._retired)

    def acquire(self, timeout: float | None = None):
        """Check out a key, waiting for one to free up. None if all retired."""
        with self._condition:
            while True:
                if self._idle:
                    return self._idle.pop(0)
                if self._pending:
                    key = self._pending.pop(0)
                    break
                if self._live <= 0:
                    return None
                if not self._condition.wait(timeout):
                    return None
        # Client construction happens outside the lock
        return KeyHandle(key, self._per_minute)

    def release(self, handle: KeyHandle) -> None:
        with self._condition:
            self._idle.append(handle)
            self._condition.notify()

    def retire(self, handle: KeyHandle, reason: str) -> int:
        """Take a key out of service permanently. Returns keys still live."""
        with self._condition:
            self._live -= 1
            self._retired.append((handle.masked, reason))
            remaining = self._live
            self._condition.notify_all()  # wake anyone waiting on a free key
        LOGGER.warning(
            "Retiring key %s (%s); %d key(s) still live", handle.masked, reason, remaining
        )
        return remaining


def file_spec(path) -> dict:
    """How to send this file, by extension. Raises for unsupported types."""
    suffix = Path(path).suffix.lower()
    spec = SUPPORTED_TYPES.get(suffix)
    if spec is None:
        raise ValueError(
            f"Unsupported file type {suffix or '(none)'}; "
            f"expected one of {', '.join(sorted(SUPPORTED_TYPES))}"
        )
    return spec


def build_content_part(path):
    """Turn a file into the content part Gemini expects for its type."""
    path = Path(path)
    spec = file_spec(path)
    data = path.read_bytes()

    if len(data) < spec["min_bytes"]:
        raise ValueError(
            f"{path.name} is only {len(data)} bytes; too small to be a real "
            f"{path.suffix} document"
        )
    if len(data) > MAX_INLINE_BYTES:
        raise ValueError(
            f"{path.name} is {len(data) / 1e6:.1f}MB; too large to send inline"
        )

    if spec["binary"]:
        return types.Part.from_bytes(data=data, mime_type=spec["mime"])

    # Text formats: decode here so an encoding problem surfaces as a clear
    # error rather than mojibake inside the model's output.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        LOGGER.warning("%s had invalid UTF-8; undecodable bytes replaced", path.name)

    if not text.strip():
        raise ValueError(f"{path.name} contains no text")
    return text


def extract_file(file_path, client, model: str = DEFAULT_MODEL) -> dict:
    """Send one PDF or text document to Gemini and return the parsed JSON.

    No retries here; process_one handles those.
    """
    file_path = Path(file_path)
    content_part = build_content_part(file_path)

    response = client.models.generate_content(
        model=model,
        contents=[content_part, PROMPT_TEXT],
        config={
            "response_mime_type": "application/json",
            "response_schema": OUTPUT_SCHEMA,
        },
    )

    finish = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish = str(getattr(candidates[0], "finish_reason", "") or "")
    if finish and "STOP" not in finish.upper():
        # MAX_TOKENS, SAFETY, RECITATION: the JSON will be unusable
        raise TruncatedResponse(f"finish_reason={finish}")

    text = getattr(response, "text", None)
    if not text:
        raise TruncatedResponse("empty response body")

    return json.loads(text)


# Backwards-compatible alias for the PDF-only name
extract_pdf = extract_file


def check_record(record: dict, name: str) -> None:
    """Warn when the model's own transaction count disagrees with the list."""
    stated = record.get("Number of Transactions")
    listed = len(record.get("Transactions") or [])
    if isinstance(stated, int) and stated != listed:
        LOGGER.warning(
            "%s: 'Number of Transactions'=%s but %d transactions listed",
            name, stated, listed,
        )


def write_json(record: dict, out_path) -> None:
    """Atomic write, so an interrupted run never leaves a half-valid file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=4, ensure_ascii=False)
        os.replace(tmp, out_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_source_files(source_dir, file_types=DEFAULT_FILE_TYPES) -> list[Path]:
    """Every matching file in the folder, in the sort order indices refer to.

    file_types is a sequence of extensions, with or without the leading dot.
    Pass ("pdf", "md") to take both. Sorted by name, then by the order the
    types were given, so a .pdf and .md of the same stem stay adjacent and
    the listing is stable between runs.
    """
    source_dir = Path(source_dir)
    wanted = []
    for raw in file_types:
        suffix = raw if str(raw).startswith(".") else f".{raw}"
        suffix = suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type {suffix}; "
                f"expected one of {', '.join(sorted(SUPPORTED_TYPES))}"
            )
        if suffix not in wanted:
            wanted.append(suffix)

    order = {suffix: position for position, suffix in enumerate(wanted)}
    found = [p for p in source_dir.glob("*") if p.suffix.lower() in order]
    return sorted(found, key=lambda p: (p.stem, order[p.suffix.lower()]))


# Backwards-compatible alias for the PDF-only name
def list_pdfs(pdf_dir) -> list[Path]:
    """Every PDF in the folder. Kept for callers written before other types."""
    return list_source_files(pdf_dir, (".pdf",))


def pending_files(
    source_dir,
    json_dir,
    start: int | None = None,
    end: int | None = None,
    skip_existing: bool = True,
    index_over: str = "all",
    file_types=DEFAULT_FILE_TYPES,
) -> list[Path]:
    """Source files to process. Creates json_dir if missing.

    file_types picks which extensions to look at, e.g. (".pdf",) or
    ("pdf", "md"). Output JSON is named after the file stem, so a .pdf and
    .md with the same stem map to the same JSON: whichever is processed
    first wins, and with skip_existing=True the other is then skipped.

    The slice is Python-style [start, end), so start=0, end=100 is the first
    hundred and end is exclusive. Either may be None.

    index_over decides what those positions count over:
      "all" (default) positions in the full sorted listing on disk, whether
            or not each file has been extracted. The listing does not change
            as work completes, so a given range always covers the same files
            — use this to split work into fixed blocks.
      "pending" positions among only the files still needing extraction, so
            start=0, end=100 means "the next hundred outstanding files".

    skip_existing=True (default) drops any file that already has a JSON. Set
    it False to re-extract and overwrite them; the two index_over modes then
    behave identically, since nothing is filtered out.
    """
    if index_over not in ("all", "pending"):
        raise ValueError(f"index_over must be 'all' or 'pending', got {index_over!r}")

    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    every = list_source_files(source_dir, file_types)
    done = {p.stem for p in json_dir.glob("*.json")} if skip_existing else set()

    if index_over == "all":
        # Slice first, then drop finished files: positions stay put
        return [p for p in every[start:end] if p.stem not in done]

    # Filter first, then slice: positions track the shrinking backlog
    return [p for p in every if p.stem not in done][start:end]


# Backwards-compatible alias for the PDF-only name
def pending_pdfs(pdf_dir, json_dir, **kwargs) -> list[Path]:
    """PDFs still needing extraction. Kept for pre-file_types callers."""
    kwargs.setdefault("file_types", (".pdf",))
    return pending_files(pdf_dir, json_dir, **kwargs)


def _try_with_key(source_path, out_path, handle, model, stop):
    """Attempt one PDF with one key. Returns (status, retire_this_key)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop is not None and stop.is_set():
            return "Cancelled", False

        handle.limiter.acquire()
        try:
            record = extract_file(source_path, handle.client, model)
            check_record(record, source_path.stem)
            write_json(record, out_path)
            return "Complete", False

        except (ValueError, TruncatedResponse) as exc:
            # Bad input or unusable output: retrying won't change anything
            return f"ERROR {type(exc).__name__}: {exc}", False

        except json.JSONDecodeError as exc:
            if attempt == MAX_ATTEMPTS:
                return f"ERROR invalid JSON: {exc}", False

        except Exception as exc:
            if _is_quota_error(exc):
                # Kill the key on the first 429, no backoff: the PDF goes
                # straight to whichever key is still alive.
                return "quota exhausted", True
            if not _is_retryable(exc):
                return f"ERROR {type(exc).__name__}: {exc}", False
            if attempt == MAX_ATTEMPTS:
                return f"ERROR {type(exc).__name__}: {exc}", False

        delay = BACKOFF_BASE * (2 ** (attempt - 1))
        if stop is not None:
            if stop.wait(delay):  # wakes immediately on interrupt
                return "Cancelled", False
        else:
            time.sleep(delay)

    return "ERROR exhausted retries", False


def process_one(
    source_path,
    json_dir,
    pool: KeyPool,
    model: str = DEFAULT_MODEL,
    stop: "threading.Event | None" = None,
) -> str:
    """Extract one file, moving to another key if this one runs out of quota.

    If stop is set the task returns immediately without spending a request,
    which is how an interrupted run drains its queue cheaply.
    """
    source_path = Path(source_path)
    out_path = Path(json_dir) / f"{source_path.stem}.json"

    while True:
        if stop is not None and stop.is_set():
            return "Cancelled"

        handle = pool.acquire()
        if handle is None:
            # Every key is spent: stop the whole run rather than churn
            if stop is not None:
                stop.set()
            return "ERROR no API keys left"

        status, retire_key = _try_with_key(source_path, out_path, handle, model, stop)

        if retire_key:
            remaining = pool.retire(handle, status)
            if remaining <= 0:
                if stop is not None:
                    stop.set()
                return "ERROR quota exhausted (no keys left)"
            continue  # same PDF, next key: the work is not lost

        pool.release(handle)
        return status


def extract_many(
    source_dir,
    json_dir,
    api_keys,
    model: str = DEFAULT_MODEL,
    start: int | None = None,
    end: int | None = None,
    skip_existing: bool = True,
    index_over: str = "all",
    file_types=DEFAULT_FILE_TYPES,
    limit: int | None = None,
    per_minute: float = REQUESTS_PER_MINUTE_PER_KEY,
    on_exhausted: str = "raise",
) -> dict[str, str]:
    """Extract source documents to JSON. Returns {stem: status}.

    file_types selects which extensions to process: (".pdf",) by default,
    or e.g. ("pdf", "md") to take both. See SUPPORTED_TYPES.

    One worker thread per API key, each paced independently.

    start/end index the full sorted listing by default (index 0 = first file
    in the folder); pass index_over="pending" to index only the files still
    outstanding. skip_existing=False re-extracts files that already
    have JSON, overwriting them. limit, if given, caps the result further.

    on_exhausted decides what happens when the last key dies mid-run:
      "raise"  (default) stop and raise QuotaExhausted, halting the caller
      "exit"   log and sys.exit(2), for unattended scripts
      "return" log and return the partial results
    In every case the queue is dropped first, so no further calls are made.
    """
    if on_exhausted not in ("raise", "exit", "return"):
        raise ValueError(f"Unknown on_exhausted {on_exhausted!r}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = pending_files(
        source_dir,
        json_dir,
        start=start,
        end=end,
        skip_existing=skip_existing,
        index_over=index_over,
        file_types=file_types,
    )
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        LOGGER.info("Nothing to extract")
        return {}

    if not skip_existing:
        already = sum(1 for p in todo if (Path(json_dir) / f"{p.stem}.json").exists())
        if already:
            LOGGER.warning(
                "skip_existing=False: %d of %d file(s) will be overwritten",
                already, len(todo),
            )

    LOGGER.info(
        "Extracting %d file(s): %s ... %s", len(todo), todo[0].stem, todo[-1].stem
    )

    pool = KeyPool(api_keys, per_minute)
    results: dict[str, str] = {}
    stop = threading.Event()

    executor = ThreadPoolExecutor(max_workers=len(api_keys))
    exhausted = False
    try:
        futures = {
            executor.submit(process_one, source, json_dir, pool, model, stop): source
            for source in todo
        }
        with tqdm(total=len(futures), unit="file", desc="Extracting") as bar:
            for future in as_completed(futures):
                source = futures[future]
                status = future.result()
                results[source.stem] = status
                if status != "Complete":
                    tqdm.write(f"{source.stem}: {status}")
                bar.update(1)

                if pool.live_keys <= 0:
                    # Last key just died: drop the queue instead of walking
                    # it to mark every remaining file Cancelled.
                    exhausted = True
                    stop.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    except KeyboardInterrupt:
        # Without this the executor would drain the whole queue on the way
        # out, spending quota on work you just asked it to abandon.
        stop.set()
        executor.shutdown(wait=False, cancel_futures=True)
        LOGGER.warning(
            "Interrupted: %d done, stopping. Already-written JSON is intact; "
            "re-run to continue.", len(results),
        )
        raise
    finally:
        executor.shutdown(wait=False)

    failed = sum(1 for v in results.values() if v != "Complete")
    LOGGER.info("Extracted %d, failed %d", len(results) - failed, failed)
    if pool.retired_keys:
        LOGGER.warning(
            "%d key(s) retired this run: %s",
            len(pool.retired_keys),
            ", ".join(f"{k} ({why})" for k, why in pool.retired_keys),
        )

    if exhausted or pool.live_keys <= 0:
        remaining = [p.stem for p in todo if results.get(p.stem) != "Complete"]
        message = (
            f"All {len(api_keys)} API key(s) exhausted after "
            f"{sum(1 for v in results.values() if v == 'Complete')} file(s); "
            f"{len(remaining)} still outstanding. Re-run once quota resets."
        )
        if on_exhausted == "raise":
            raise QuotaExhausted(message, results=results, remaining=remaining)
        LOGGER.error(message)
        if on_exhausted == "exit":
            sys.exit(2)

    return results