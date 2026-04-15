#!/usr/bin/env python3
"""Download and extract the SpamAssassin public corpus into server/data/raw/."""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SpamAssassin public corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Directory to download and extract archives into",
    )
    args = parser.parse_args()
    dest: Path = args.data_dir
    dest.mkdir(parents=True, exist_ok=True)

    for name in ARCHIVES:
        url = BASE + name
        archive_path = dest / name
        if not archive_path.is_file():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, archive_path)
        else:
            print(f"Already have {archive_path.name}, skipping download")
        marker = dest / f".extracted_{name}"
        if marker.is_file():
            print(f"Already extracted {name}, skipping")
            continue
        print(f"Extracting {name} ...")
        with tarfile.open(archive_path, "r:bz2") as tf:
            kwargs: dict = {"path": dest}
            if sys.version_info >= (3, 12):
                kwargs["filter"] = "data"
            tf.extractall(**kwargs)
        marker.write_text("ok\n")
        print(f"Done {name}")

    print(f"Corpus ready under {dest}")


if __name__ == "__main__":
    main()
