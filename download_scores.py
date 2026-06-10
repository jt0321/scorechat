"""
download_scores.py
------------------
Downloads public-domain piano scores from IMSLP into ./data/.
These PDFs are the raw input to the ingest pipeline (step 1 of OMR → analysis → pgvector).
All files are public domain.

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

IMSLP_API   = "https://imslp.org/api.php"
IMSLP_BASE  = "https://imslp.org/wiki/"

# Minimum combined match score to auto-accept a candidate
ACCEPT_THRESH = 2.0

# Seconds to sleep between IMSLP API calls (be polite)
API_SLEEP = 1.2

# Hardcoded seed scores (mode 1).
# Full metadata is carried here so ingest_scores.py doesn't need to re-derive it.
SCORES: list[dict] = [
    {
        "url":      "https://vmirror.imslp.org/files/imglnks/usimg/8/83/IMSLP699774-PMLP1458-Bethoven_Moonlight_Sonata_No.14.pdf",
        "filename": "Beethoven_Moonlight_Sonata_Op27_No2.pdf",
        "composer": "Ludwig van Beethoven",
        "title":    "Piano Sonata No. 14 in C-sharp minor",
        "opus":     "Op. 27, No. 2",
        "key":      "C-sharp minor",
        "year":     1801,
        "imslp":    "https://imslp.org/wiki/Piano_Sonata_No.14,_Op.27_No.2_(Beethoven,_Ludwig_van)",
    },
    {
        "url":      "https://s9.imslp.org/files/imglnks/usimg/5/5e/IMSLP1519-PMLP02467-Chopin-Op10No3.pdf",
        "filename": "Chopin_Etude_Op10_No3.pdf",
        "composer": "Frédéric Chopin",
        "title":    "Étude in E major",
        "opus":     "Op. 10, No. 3",
        "key":      "E major",
        "year":     1833,
        "imslp":    "https://imslp.org/wiki/Études,_Op.10_(Chopin,_Frédéric)",
    },
    {
        "url":      "https://s9.imslp.org/files/imglnks/usimg/4/43/IMSLP00749-PMLP01458-Beethoven-Op02No01.pdf",
        "filename": "Beethoven_Sonata_Op2_No1.pdf",
        "composer": "Ludwig van Beethoven",
        "title":    "Piano Sonata No. 1 in F minor",
        "opus":     "Op. 2, No. 1",
        "key":      "F minor",
        "year":     1796,
        "imslp":    "https://imslp.org/wiki/Piano_Sonata_No.1,_Op.2_No.1_(Beethoven,_Ludwig_van)",
    },
    {
        "url":      "https://s9.imslp.org/files/imglnks/usimg/3/3b/IMSLP00751-PMLP02467-Chopin-Op09No02.pdf",
        "filename": "Chopin_Nocturne_Op9_No2.pdf",
        "composer": "Frédéric Chopin",
        "title":    "Nocturne in E-flat major",
        "opus":     "Op. 9, No. 2",
        "key":      "E-flat major",
        "year":     1832,
        "imslp":    "https://imslp.org/wiki/Nocturnes,_Op.9_(Chopin,_Frédéric)",
    },
]


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip accents, expand ♭/♯ to words, remove punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[♭b]-flat|♭", "flat", text)
    text = re.sub(r"[♯#]-sharp|♯", "sharp", text)
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_opus_tokens(text: str) -> set[str]:
    """Return lower-case opus/catalog tokens found in text."""
    tokens: set[str] = set()
    for m in re.finditer(r"op\.?\s*(\d+(?:\s*no\.?\s*\d+)?)", text, re.I):
        tokens.add("op" + re.sub(r"\s+", "", m.group(1)))
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
    Search IMSLP via opensearch API.
    Returns list of {title} dicts.

    Note: action=query&list=search returns empty results on IMSLP due to
    server-side restrictions; action=opensearch works correctly.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit":  str(limit),
        "format": "json",
    }
    try:
        resp = requests.get(IMSLP_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
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
    """Score an IMSLP page title against our query fields. Higher is better."""
    norm_title    = normalize(candidate_title)
    norm_work     = normalize(work)
    comp_tokens   = normalize(composer).split()
    title_tokens  = norm_title.split()
    work_tokens   = norm_work.split()

    score = lcs_ratio(work_tokens, title_tokens)

    if comp_tokens and comp_tokens[-1] in title_tokens:
        score += 2.0

    query_opus = extract_opus_tokens(f"op {opus} {catalog}")
    title_opus = extract_opus_tokens(norm_title)
    if query_opus and (query_opus & title_opus):
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
    """
    comp_last  = composer.strip().split()[-1]
    work_short = re.sub(r"\s+(in\s+[A-G][\s-].*|No\.\s*\d+.*)$", "", work, flags=re.I).strip() or work
    opus_clean = ("Op." + re.sub(r"[Oo]p\.?\s*", "", opus).strip()) if opus else ""
    cat_clean  = catalog.strip() if catalog else ""

    q1 = " ".join(filter(None, [work_short, opus_clean or cat_clean]))
    q2 = f"{work_short} {comp_last}"

    candidates = mw_search(q1, limit=10)
    if not candidates:
        time.sleep(API_SLEEP)
        candidates = mw_search(q2, limit=10)
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

    url = (IMSLP_BASE + best_title.replace(" ", "_")) if best_title else None
    return url, (best_score < ACCEPT_THRESH)


# ---------------------------------------------------------------------------
# CSV-driven mode
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, do_download: bool = False) -> Optional[Path]:
    """
    Read csv_path, look up IMSLP URLs for rows missing them,
    write back the updated CSV, and optionally download PDFs.
    Returns the path to a review CSV, or None if all rows matched.
    """
    fieldnames = ["composer", "work", "catalog", "opus", "nickname",
                  "imslp_url", "review_flag", "notes"]
    rows: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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
            row["imslp_url"]   = url
            row["review_flag"] = "REVIEW" if needs_review else ""
            print(f"    {'REVIEW' if needs_review else 'OK'}: {url}")
        else:
            row["review_flag"] = "REVIEW"
            print("    NOT FOUND – flagged for review")
        updated += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nUpdated {updated} rows → {csv_path}")

    review_rows  = [r for r in rows if r.get("review_flag") == "REVIEW"]
    review_path  = csv_path.with_name(csv_path.stem + "_review.csv")
    if review_rows:
        with open(review_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(review_rows)
        print(f"Review CSV ({len(review_rows)} rows) → {review_path}")
    else:
        print("All rows matched – no review CSV needed.")
        review_path = None

    if do_download:
        _download_pdfs(rows)

    return review_path


def _download_pdfs(rows: list[dict]) -> None:
    """
    For each row with a populated imslp_url, fetch the IMSLP page and
    download the first listed PDF.
    NOTE: IMSLP PDFs are rate-limited; failures are logged, not fatal.
    """
    for row in rows:
        url = row.get("imslp_url", "")
        if not url or row.get("review_flag") == "REVIEW":
            continue
        composer  = normalize(row["composer"]).replace(" ", "_")
        work_slug = re.sub(r"\s+", "_", normalize(row["work"]))[:40]
        filename  = f"{composer}_{work_slug}.pdf"
        out_path  = DATA_DIR / filename
        if out_path.exists():
            print(f"  ✓ {filename} already exists, skipping.")
            continue

        page_title = url.replace(IMSLP_BASE, "").replace("_", " ")
        params = {
            "action": "query", "titles": page_title,
            "prop": "extlinks", "format": "json", "ellimit": "5",
        }
        try:
            resp = requests.get(IMSLP_API, params=params, headers=HEADERS, timeout=15)
            data = resp.json()
            pdf_links = [
                el.get("*", "")
                for page_data in data.get("query", {}).get("pages", {}).values()
                for el in page_data.get("extlinks", [])
                if el.get("*", "").lower().endswith(".pdf")
            ]
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
            print(f"  ✓ Saved {filename} ({out_path.stat().st_size // 1024} KB)")
        except Exception as exc:
            print(f"  ✗ Failed to download {row['work']}: {exc}")
        time.sleep(API_SLEEP)


# ---------------------------------------------------------------------------
# Legacy / hardcoded mode
# ---------------------------------------------------------------------------

def _legacy_download() -> None:
    for score in SCORES:
        path = DATA_DIR / score["filename"]
        if path.exists():
            print(f"✓ {score['filename']} already exists, skipping.")
            continue
        print(f"Downloading {score['filename']} …")
        try:
            resp = requests.get(score["url"], headers=HEADERS, stream=True, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Saved {score['filename']} ({path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"✗ Failed to download {score['filename']}: {e}")

    print("\nDone. Run `python ingest_scores.py` to push scores through the pipeline.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / match IMSLP scores for ScoreChat."
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to score list CSV (e.g. data/frsm_scores.csv). "
             "Enables CSV-driven IMSLP lookup mode.",
    )
    parser.add_argument(
        "--download", action="store_true",
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
