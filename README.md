# Congress Trades Pipeline

Downloads U.S. House financial disclosures, extracts the trades from each
Periodic Transaction Report (PTR) with Gemini, and consolidates everything
into an Excel workbook.

Dataset from 2015-Early August 2026: https://www.kaggle.com/datasets/ng3412/congress-trading-disclosures

Disclaimer: Built with Claude. An original less clunky but undocumented version is available on request.

Four stages, each resumable. If a run is interrupted, call the same function
again and it picks up where it stopped.

```
Clerk's website  ──▶  PDFs  ──▶  JSON  ──▶  audit  ──▶  Excel
   stage 1            stage 2          stage 3        stage 4
```

---

## 1. Setup

### Files

Put these five files in the same folder:

| File | Role |
|---|---|
| `main.py` | the entry point — the only file you call |
| `congress_disclosures_main.py` | stage 1: index and download |
| `process_docs.py` | stage 2: Gemini extraction |
| `audit_transactions.py` | stage 3: quality check |
| `json_to_excel.py` | stage 4: workbook |

### Packages

Run once in a terminal, or in a notebook cell with a leading `!`:

```
pip install pandas openpyxl requests tqdm google-genai
```

### API keys

Stage 2 needs at least one Gemini API key from
<https://aistudio.google.com/apikey>. More keys means more throughput, since
each gets its own worker thread.

Preferred — set an environment variable so keys stay out of your code:

```
setx GEMINI_API_KEYS "key1,key2,key3"
```

`setx` only affects **new** processes, so restart Jupyter afterwards.

Alternative — open `main.py` and fill in:

```python
API_KEYS = ["key1", "key2", "key3"]
```

### Where files go

Nothing to configure. Everything is written beside `main.py`, so put your
notebook in the same folder and the project stays self-contained — you can
zip it, move it, or copy it to another machine without editing a path.

```
your-project-folder/
  main.py                       ← and the four module files
  congress_trades.ipynb         ← your notebook, same folder
  index/      2026_list.xlsx    filing index + per-file download status
  temp/       2026/2026FD.xml   extracted from the Clerk's zip
  downloads/  2026/*.pdf        the PTR PDFs
  json/       2026/*.json       one JSON per extracted PDF
  quarantine/                   bad extractions moved aside, timestamped
  reports/                      audit report + the final workbook
```

The first six folders are created on demand. Paths are resolved from
`main.py`'s own location, not the working directory, so output lands in the
same place regardless of where the kernel was started.

To keep data somewhere else — a different drive, say — pass `base=` to any
stage:

```python
main.stage1_download(2026, base=r"D:\congress-data")
```

---

## 2. Running it in Jupyter

### First cell — always

```python
import main
main.setup_logging()          # progress messages appear in the notebook
```

If you edit any of the module files, restart the kernel or:

```python
import importlib
importlib.reload(main)
```

### Check the layout before downloading anything

```python
main.paths(2026)
```

Confirm those paths point where you expect before running anything else.

### Stage 1 — download the PDFs

```python
df = main.stage1_download(2026)
```

Start small the first time to confirm everything works:

```python
df = main.stage1_download(2026, limit=20)
```

Then re-run without `limit` to get the rest. Already-downloaded files are
skipped, so nothing is fetched twice.

When new filings are posted and you want a fresh index:

```python
df = main.stage1_download(2026, refresh_index=True)
```

Only `FilingType == "P"` is downloaded, since those are the only filings that
list trades. To fetch others:

```python
df = main.stage1_download(2026, filing_types=("P", "A"))
```

### Stage 2 — extract with Gemini

**Try a handful first.** This stage costs API quota, and a bad prompt or
model choice is much cheaper to discover on 5 files than 500.

```python
results = main.stage2_extract(2026, limit=5)
```

Inspect one before continuing:

```python
import json
from pathlib import Path
p = next(Path(main.paths(2026).json).glob("*.json"))
print(json.dumps(json.loads(p.read_text()), indent=2)[:1500])
```

Happy? Run the rest:

```python
results = main.stage2_extract(2026)
```

Useful arguments:

| Argument | Effect |
|---|---|
| `limit=50` | stop after 50 files |
| `start=0, end=200` | process a fixed slice of the folder listing |
| `model="gemini-2.5-pro"` | stronger model — better on long, messy filings |
| `skip_existing=False` | re-extract files that already have JSON |
| `file_types=("pdf","md")` | also process markdown/text sources |

### Stage 3 — audit

Finds JSON where the filing's stated transaction count doesn't match the
number actually extracted. That mismatch is the main symptom of a truncated
extraction, and it's silent otherwise.

```python
report, _ = main.stage3_audit([2026])
report[report["Status"] != "OK"]
```

To clear the bad ones so stage 2 redoes them — dry run first:

```python
main.stage3_audit([2026], action="move")                  # shows what it would do
main.stage3_audit([2026], action="move", dry_run=False)   # actually moves them
```

Files go to `quarantine/<timestamp>/`, not the bin, so you can compare the
old extraction against the new one. Then re-run stage 2 and consider a
stronger model:

```python
main.stage2_extract(2026, model="gemini-2.5-pro")
```

### Stage 4 — build the workbook

```python
tables = main.stage4_excel([2024, 2025, 2026])
```

Writes `reports/ptr_transactions_2024-2026.xlsx` with five sheets:

| Sheet | Contents |
|---|---|
| **Filings** | one row per filing |
| **Transactions** | one row per trade — the main sheet |
| **Investment Vehicles** | trusts and funds named in the filing |
| **Exercise Details** | option-exercise specifics |
| **Problems** | anything that didn't load cleanly |

Join sheets on `Source File`. Transactions include `Amount Min`/`Amount Max`
(parsed from ranges like `$1,001 - $15,000`, so you can sum and sort) and
`Date (parsed)` as a real date, since the raw text sorts wrongly in Excel.

**Check the Problems sheet first.** Empty means a clean run.

### Everything at once

Once you trust each stage:

```python
summary = main.run_all([2024, 2025, 2026])
```

### Where am I?

```python
main.status([2024, 2025, 2026])
```

```
 Year  Index built  PDFs downloaded  JSON extracted  Outstanding
 2024         True              487             487            0
 2025         True              502             340          162
```

---

## 3. Interrupting a run

Press **Ctrl-C** once (in Jupyter: the ■ stop button) and wait a few seconds.
Don't hit it repeatedly — that can interrupt the cleanup itself.

In-flight work finishes, the queue is dropped, and nothing further is
requested. Every file is written atomically, so you never end up with a
half-written PDF or JSON — a file either exists complete or doesn't exist.
Re-run the same stage to continue.

---

## 4. Troubleshooting

**`ModuleNotFoundError: No module named 'main'`**
The notebook isn't in the folder with the `.py` files. Check with
`import os; os.getcwd()`, then either move the notebook next to them or add
a cell before the import:
`import sys; sys.path.insert(0, r"C:\path\to\folder")`
Data still lands beside `main.py`, not beside the notebook.

**`ValueError: No API keys found in $GEMINI_API_KEYS`**
The variable isn't set, or Jupyter started before you set it. Restart
Jupyter, or paste the keys into `API_KEYS` in `main.py`.

**`QuotaExhausted: All 3 API key(s) exhausted`**
Every key hit its limit. Already-extracted JSON is safe. Wait for the quota
to reset — free-tier daily quotas roll over at midnight Pacific — and re-run
the same command. The exception carries `.results` and `.remaining` if you
want the detail.

**Keys retiring early in a run**
A key is retired on its first 429. If that happens within the first few
files, you're tripping the per-minute rate limit rather than the daily cap.
Lower `REQUESTS_PER_MINUTE_PER_KEY` near the top of `process_docs.py`.

**`Missing (404)` in the download status column**
No PDF exists at that DocID. Usually genuine, occasionally a sign the filing
type routes to a different URL path. Not an error worth chasing unless it
affects many rows.

**Lots of `Mismatch` in the audit**
The model is truncating long filings. Re-extract those with
`model="gemini-2.5-pro"`.

**`FileNotFoundError: No downloads folder`**
Stage 2 ran before stage 1 for that year. Run `main.stage1_download(year)`.

**`PermissionError` writing an .xlsx**
The workbook is open in Excel. Close it and re-run.

---

## 5. Costs and etiquette

Stage 1 is rate-limited to 2 requests/second against a government website.
Please don't raise it.

Stage 2 is the only stage that costs money. One API call per PDF; a year of
PTRs is typically several hundred filings. `limit` is there so you can
measure real cost on a small sample before committing to a full year.
