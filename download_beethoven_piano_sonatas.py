"""
download_beethoven_piano_sonatas.py
-----------------------------------
Downloads Humdrum (.krn) scores for Beethoven piano sonatas directly from
Craig Sapp's GitHub repository (https://github.com/craigsapp/beethoven-piano-sonatas)
into the local ./data/ directory.

Usage:
    python download_beethoven_piano_sonatas.py [--sonata 32] [--all]
"""

import os
import sys
import argparse
import requests
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GITHUB_API_URL = "https://api.github.com/repos/craigsapp/beethoven-piano-sonatas/contents/kern"
RAW_BASE_URL = "https://raw.githubusercontent.com/craigsapp/beethoven-piano-sonatas/master/kern"

# Standard headers for GitHub API to prevent issues
HEADERS = {
    "User-Agent": "ScoreChat/1.0 (educational use; github.com/jt0321/scorechat)"
}

def get_krn_file_list() -> list[str]:
    """Fetch the list of all .krn files in the kern/ directory of the repository."""
    print("Fetching file list from GitHub repository...")
    try:
        resp = requests.get(GITHUB_API_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        contents = resp.json()
        
        # Filter files ending with .krn
        krn_files = [item["name"] for item in contents if item["name"].endswith(".krn")]
        return sorted(krn_files)
    except Exception as e:
        print(f"Error fetching directory contents from GitHub API: {e}", file=sys.stderr)
        print("Falling back to standard list structure.", file=sys.stderr)
        # Fallback list of common files in case of API rate-limiting/issues
        fallback_files = []
        for s in range(1, 33):
            # Most sonatas have at least 2 movements
            fallback_files.append(f"sonata{s:02d}-1.krn")
            fallback_files.append(f"sonata{s:02d}-2.krn")
        return fallback_files

def download_file(filename: str) -> bool:
    """Download a single .krn file from the raw GitHub URL."""
    out_path = DATA_DIR / filename
    if out_path.exists():
        print(f"  ✓ {filename} already exists, skipping.")
        return True

    url = f"{RAW_BASE_URL}/{filename}"
    print(f"  Downloading {filename} …")
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        if resp.status_code == 404:
            # Silent fallback for non-existent movement numbers in fallback mode
            return False
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    ✓ Saved to {out_path} ({out_path.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"    ✗ Failed to download {filename}: {e}", file=sys.stderr)
        return False

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Beethoven Piano Sonata Humdrum (.krn) scores."
    )
    parser.add_argument(
        "--sonata", type=int, default=32,
        help="Specific sonata number to download (1-32). Default is 32 (Op. 111).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all 32 sonatas (all movements). Warning: this is around 100+ movements.",
    )
    args = parser.parse_args()

    files_to_download = []
    
    if args.all:
        print("Preparing to download all sonatas...")
        files_to_download = get_krn_file_list()
    else:
        # Download specific sonata
        sonata_num = args.sonata
        if sonata_num < 1 or sonata_num > 32:
            print(f"Error: Sonata number must be between 1 and 32.", file=sys.stderr)
            return
            
        print(f"Preparing to download Sonata No. {sonata_num}...")
        # Get list to see exactly which movements exist
        all_files = get_krn_file_list()
        prefix = f"sonata{sonata_num:02d}-"
        files_to_download = [f for f in all_files if f.startswith(prefix)]
        
        # If API failed and fallback list didn't yield matches
        if not files_to_download:
            files_to_download = [f"sonata{sonata_num:02d}-1.krn", f"sonata{sonata_num:02d}-2.krn", f"sonata{sonata_num:02d}-3.krn", f"sonata{sonata_num:02d}-4.krn"]

    downloaded_count = 0
    for filename in files_to_download:
        if download_file(filename):
            downloaded_count += 1
            
    print(f"\nFinished download process. Downloaded/verified {downloaded_count} score file(s).")
    print("Run `python ingest_scores.py` to index the Humdrum scores into the database.")

if __name__ == "__main__":
    main()
