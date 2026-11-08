"""Download and extract the MovieLens 'latest-small' dataset into data/raw/.

Usage:
    python -m src.download_data
"""
import io
import urllib.request
import zipfile

from src.config import MOVIELENS_URL, RAW_DIR


def download_movielens() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MOVIELENS_URL} ...")
    with urllib.request.urlopen(MOVIELENS_URL, timeout=60) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            name = member.split("/")[-1]
            if name.endswith(".csv"):
                (RAW_DIR / name).write_bytes(zf.read(member))
                print(f"  extracted {name}")
    print(f"Done. Files are in {RAW_DIR}")


if __name__ == "__main__":
    download_movielens()
