"""
download_scores.py
------------------
Downloads public-domain piano scores from IMSLP into ./data/.
All files are public domain. Run before index_scores.py.

Usage:
    python download_scores.py
"""

from pathlib import Path
import requests

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

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "ScoreChat/1.0 (educational use; public domain scores only)"
}


def main():
    for url, filename in SCORES:
        path = DATA_DIR / filename
        if path.exists():
            print(f"\u2713 {filename} already exists, skipping.")
            continue
        print(f"Downloading {filename} ...")
        try:
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = path.stat().st_size // 1024
            print(f"\u2713 Saved {filename} ({size_kb} KB)")
        except Exception as e:
            print(f"\u2717 Failed to download {filename}: {e}")

    print("\nDone. Run `python index_scores.py` to build the FAISS index.")


if __name__ == "__main__":
    main()
