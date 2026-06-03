"""
download_scores.py
------------------
Downloads public-domain piano scores from IMSLP into ./data/.
All files are public domain. Run before index_scores.py.

Modes
-----
1. Legacy hardcoded list (backward-compatible):
       python download_scores.py

2. CSV-driven with IMSLP MediaWiki lookup and PDF download:
       python download_scores.py --csv data/frsm_scores.csv [--download]

   Without --download the script only populates/updates the imslp_url
   column via the MediaWiki API and writes a review CSV.  Add --download
   to also fetch the first available PDF for each matched work.

CSV schema (data/frsm_scores.csv)
----------------------------------
composer, work, catalog, opus, nickname, imslp_url, review_flag, notes

IMSLP matching strategy
------------------------
1. Normalize query: strip accents, fold case, expand ♭→b / ♯→sharp,
   collapse spaces, drop stray punctuation.
2. Try MediaWiki search: action=query&list=search against the IMSLP wiki
   using "<composer> <work> [opus]" as the search query.
3. Score each candidate page title against our normalized query using
   three sub-scores:
     a. Exact opus/catalog number hit   (+3 bonus)
     b. Composer token present          (+2 bonus)
     c. Longest-common-subsequence ratio of title tokens
4. Accept the top candidate when its combined score exceeds ACCEPT_THRESH.
5. Rows that fall below threshold get review_flag=REVIEW in the output CSV
   so you can inspect and manually fill imslp_url.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "ScoreChat/1.0 (educational use; public domain scores only; "
                  "contact: github.com/jt0321/scorechat)"
}

IMSLP_API = "https://imslp.org/api.php"
IMSLP_BASE = "https://imslp.org/wiki/"

# Minimum combined match score to auto-accept a candidate
ACCEPT_THRESH = 2.0

# Seconds to sleep between IMSLP API calls (be polite)
API_SLEEP = 1.2

# Legacy hardcoded scores (mode 1 – backward compatible)
SCORES = [
    (
        "https://vmirror.imslp.org/files/imglnks/usimg/8/83/IMSLP699774-PMLP1458-Bethoven_Moonlight_Sonata_No.14.pdf",
        "Beethoven_Moonlight_Sonata_Op27_No2.pdf",
    ),
    (
        "https://s9.imslp.org/files/imglnks/usimg/5/5e/IMSLP1519-PMLP02467-Chopin-Op10No3.pdf",
        "Chopin_Etude_Op10_No3.pdf",
    ),
    (
        "https://s9.imslp.org/files/imglnks/usimg/4/43/IMSLP00749-PMLP01458-Beethoven-Op02No01.pdf",
        "Beethoven_Sonata_Op2_No1.pdf",
    ),
    (
        "https://s9.imslp.org/files/imglnks/usimg/3/3b/IMSLP00751-PMLP02467-Chopin-Op09No02.pdf",
        "Chopin_Nocturne_Op9_No2.pdf",
    ),
]


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------
_FLAT_RE = re.compile(r"[♭\-]?\s*flat\b|♭", re.IGNORECASE)
_SHARP_RE = re.compile(r"[♯#]\s*sharp\b|♯|#", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """
    Lowercase, strip accents, expand ♭→flat and ♯→sharp,
    remove punctuation, collapse whitespace.
    """
    if not text:
        return ""
    # Decompose unicode and drop combining marks (accents)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    # Normalise flat/sharp symbols to words
    text = re.sub(r"[♭b]-flat|♭", "flat", text)
    text = re.sub(r"[♯#]-sharp|♯", "sharp", text)
    # Drop remaining punctuation (keep digits and letters)
    text = _PUNCT_RE.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_opus_tokens(text: str) -> set[str]:
    """Return a set of lower-case opus/catalog tokens found in text."""
    tokens: set[str] = set()
    # Op. 53, Op 53, op53
    for m in re.finditer(r"op\.?\s*(\d+(?:\s*no\.?\s*\d+)?)", text, re.I):
        tokens.add("op" + re.sub(r"\s+", "", m.group(1)))
    # BWV 830, D. 960, S. 178, M. 55
    for m in re.finditer(r"\b([A-Z]{1,4})\.?\s*(\d+(?:/\d+)?)\b", text, re.I):
        tokens.add(m.group(1).lower() + m.group(2).replace("/", ""))
    return tokens


def lcs_ratio(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Longest-common-subsequence token ratio (0–1)."""
    if not a_tokens or not b_tokens:
        return 0.0
    m, n = len(a_tokens), len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / max(m, n)


# ---------------------------------------------------------------------------
# IMSLP MediaWiki API helpers
# ---------------------------------------------------------------------------

def mw_search(query: str, limit: int = 8) -> list[dict]:
    """
    Search IMSLP's MediaWiki for page titles matching *query*.
    Uses the OpenSearch (opensearch) API which is reliably supported by IMSLP.
    Returns a list of {title} dicts (compatible with the scorer below).

    Note: IMSLP's ``action=query&list=search`` endpoint returns empty results
    for most classical-music title queries due to server-side restrictions.
    ``action=opensearch`` (prefix/title search) works correctly.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": str(limit),
        "format": "json",
    }
    try:
        resp = requests.get(IMSLP_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # opensearch returns [query, [titles], [descs], [urls]]
        titles = data[1] if len(data) > 1 else []
        return [{"title": t} for t in titles]
    except Exception as exc:
        print(f"  [WARN] MediaWiki opensearch error: {exc}")
        return []


def score_candidate(
    candidate_title: str,
    composer: str,
    work: str,
    opus: str,
    catalog: str,
) -> float:
    """
    Score a IMSLP page title against our query fields.
    Higher is better.
    """
    norm_title = normalize(candidate_title)
    norm_work = normalize(work)
    norm_composer = normalize(composer)

    title_tokens = norm_title.split()
    work_tokens = norm_work.split()
    comp_tokens = norm_composer.split()

    score = lcs_ratio(work_tokens, title_tokens)

    # Bonus: composer last-name present in title
    if comp_tokens:
        last = comp_tokens[-1]
        if last in title_tokens:
            score += 2.0

    # Bonus: exact opus/catalog number match
    query_opus_tokens = extract_opus_tokens(f"op {opus} {catalog}")
    title_opus_tokens = extract_opus_tokens(norm_title)
    if query_opus_tokens and (query_opus_tokens & title_opus_tokens):
        score += 3.0

    return score


def find_imslp_url(
    composer: str,
    work: str,
    opus: str,
    catalog: str,
    nickname: str,
) -> tuple[Optional[str], bool]:
    """
    Return (imslp_url, needs_review).
    imslp_url is None if nothing was found above threshold.
    needs_review is True when the best match is below ACCEPT_THRESH.

    Strategy: IMSLP opensearch works best with short, title-fragment queries.
    We try two forms:
      1. "<Work short title> <opus/catalog>" – prefix of the IMSLP page title
      2. "<Work short title> <composer last name>" – fallback
    """
    comp_last = composer.strip().split()[-1]  # e.g. "Beethoven"

    # Strip key-signature suffix for a shorter prefix query
    work_short = re.sub(r"\s+(in\s+[A-G][\s-].*|No\.\s*\d+.*)$", "", work, flags=re.I).strip()
    if not work_short:
        work_short = work

    # Normalize opus for the query
    opus_clean = ""
    if opus:
        opus_clean = "Op." + re.sub(r"[Oo]p\.?\s*", "", opus).strip()
    cat_clean = catalog.strip() if catalog else ""

    # Build two candidate queries
    q1_parts = [work_short]
    if opus_clean:
        q1_parts.append(opus_clean)
    elif cat_clean:
        q1_parts.append(cat_clean)
    query1 = " ".join(q1_parts)

    q2_parts = [work_short, comp_last]
    query2 = " ".join(q2_parts)

    candidates = mw_search(query1, limit=10)
    if not candidates:
        time.sleep(API_SLEEP)
        candidates = mw_search(query2, limit=10)
    time.sleep(API_SLEEP)

    if not candidates:
        return None, True

    best_title: Optional[str] = None
    best_score = -1.0
    for c in candidates:
        s = score_candidate(c["title"], composer, work, opus, catalog)
        if s > best_score:
            best_score = s
            best_title = c["title"]

    if best_score >= ACCEPT_THRESH and best_title:
        url = IMSLP_BASE + best_title.replace(" ", "_")
        return url, False

    # Return best guess but flag for review
    if best_title:
        url = IMSLP_BASE + best_title.replace(" ", "_")
        return url, True

    return None, True


# ---------------------------------------------------------------------------
# CSV-driven mode
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, do_download: bool = False) -> Path:
    """
    Read *csv_path*, look up IMSLP URLs for rows missing them,
    write back the updated CSV, and optionally download PDFs.

    Returns the path to a separate review CSV for rows needing manual check.
    """
    rows: list[dict] = []
    fieldnames = [
        "composer", "work", "catalog", "opus", "nickname",
        "imslp_url", "review_flag", "notes",
    ]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure all expected columns exist
            for col in fieldnames:
                row.setdefault(col, "")
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {csv_path}")
    updated = 0

    for row in rows:
        if row.get("imslp_url"):
            print(f"  ✓ {row['composer']} – {row['work']} (URL already present)")
            continue

        print(f"  → Searching IMSLP: {row['composer']} – {row['work']} …")
        url, needs_review = find_imslp_url(
            composer=row["composer"],
            work=row["work"],
            opus=row.get("opus", ""),
            catalog=row.get("catalog", ""),
            nickname=row.get("nickname", ""),
        )

        if url:
            row["imslp_url"] = url
            row["review_flag"] = "REVIEW" if needs_review else ""
            status = "REVIEW" if needs_review else "OK"
            print(f"    {status}: {url}")
        else:
            row["review_flag"] = "REVIEW"
            print(f"    NOT FOUND – flagged for review")
        updated += 1

    # Write updated main CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nUpdated {updated} rows → {csv_path}")

    # Write review CSV
    review_path = csv_path.with_name(csv_path.stem + "_review.csv")
    review_rows = [r for r in rows if r.get("review_flag") == "REVIEW"]
    if review_rows:
        with open(review_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(review_rows)
        print(f"Review CSV ({len(review_rows)} rows) → {review_path}")
    else:
        print("All rows matched – no review CSV needed.")
        review_path = None

    # Optional: download PDFs for matched rows
    if do_download:
        _download_pdfs(rows)

    return review_path


def _download_pdfs(rows: list[dict]) -> None:
    """
    For each row with a populated imslp_url, attempt to fetch the IMSLP
    page and download the first listed PDF file link.

    NOTE: IMSLP PDFs are rate-limited and may require a cookie/session.
    This function makes a best-effort download; failures are logged, not fatal.
    """
    for row in rows:
        url = row.get("imslp_url", "")
        if not url or row.get("review_flag") == "REVIEW":
            continue
        composer = normalize(row["composer"]).replace(" ", "_")
        work_slug = re.sub(r"\s+", "_", normalize(row["work"]))[:40]
        filename = f"{composer}_{work_slug}.pdf"
        out_path = DATA_DIR / filename
        if out_path.exists():
            print(f"  ✓ {filename} already exists, skipping.")
            continue

        # Use IMSLP's Special:IMSLPDisclaimerAccept redirect to get the file list
        # The IMSLP API exposes file links via action=query on the wiki page
        page_title = url.replace(IMSLP_BASE, "").replace("_", " ")
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extlinks",
            "format": "json",
            "ellimit": "5",
        }
        try:
            resp = requests.get(IMSLP_API, params=params, headers=HEADERS, timeout=15)
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            pdf_links = []
            for page_data in pages.values():
                for el in page_data.get("extlinks", []):
                    link = el.get("*", "")
                    if link.lower().endswith(".pdf"):
                        pdf_links.append(link)
            if not pdf_links:
                print(f"  ✗ No direct PDF links found for {row['work']}")
                continue

            pdf_url = pdf_links[0]
            print(f"  Downloading {filename} from {pdf_url[:60]}…")
            r = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=30)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = out_path.stat().st_size // 1024
            print(f"  ✓ Saved {filename} ({size_kb} KB)")
        except Exception as exc:
            print(f"  ✗ Failed to download {row['work']}: {exc}")
        time.sleep(API_SLEEP)


# ---------------------------------------------------------------------------
# Legacy mode (backward-compatible)
# ---------------------------------------------------------------------------

def _legacy_download() -> None:
    for url, filename in SCORES:
        path = DATA_DIR / filename
        if path.exists():
            print(f"✓ {filename} already exists, skipping.")
            continue
        print(f"Downloading {filename} …")
        try:
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = path.stat().st_size // 1024
            print(f"✓ Saved {filename} ({size_kb} KB)")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

    print("\nDone. Run `python index_scores.py` to build the FAISS index.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / match IMSLP scores for ScoreChat."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to score list CSV (e.g. data/frsm_scores.csv). "
             "Enables CSV-driven IMSLP lookup mode.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Also download PDFs for matched rows (requires --csv).",
    )
    args = parser.parse_args()

    if args.csv:
        if not args.csv.exists():
            print(f"ERROR: CSV not found at {args.csv}")
            return
        process_csv(args.csv, do_download=args.download)
    else:
        _legacy_download()


if __name__ == "__main__":
    main()
