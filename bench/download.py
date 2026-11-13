"""Ingest the external IDSSE / DFL data from the public figshare API.

    python -m bench.download            # metadata + tracking of the 7 matches (~2.6 GB)

Dataset: Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data 12, 195 (2025), https://doi.org/10.6084/m9.figshare.28196177, licence CC BY 4.0.

Every file is checked against the md5 published by figshare; files already downloaded correctly
are skipped. Destination: external/idsse/ (gitignored: ~2.6 GB, not part of the code).

If an antivirus intercepts HTTPS, point SSL_CERT_FILE at its root certificate.
"""

import hashlib
import json
import os
import sys
import urllib.request

from . import idsse

ARTICLE = 28196177
API = "https://api.figshare.com/v2/articles/{}/files?page_size=100".format(ARTICLE)
KINDS = ("matchinformation", "positions_raw_observed")   # the bench does not use the events


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_list():
    with urllib.request.urlopen(API, timeout=60) as r:
        files = json.load(r)
    return [f for f in files
            if any(k in f["name"] for k in KINDS) and any(m in f["name"] for m in idsse.MATCHES)]


def fetch(f, folder):
    path = os.path.join(folder, f["name"])
    if os.path.isfile(path) and os.path.getsize(path) == f["size"] and md5(path) == f["computed_md5"]:
        print("  already here", f["name"])
        return
    tmp = path + ".part"
    print("  downloading {} ({:.0f} MB)".format(f["name"], f["size"] / 1e6), flush=True)
    with urllib.request.urlopen(f["download_url"], timeout=120) as r, open(tmp, "wb") as out:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            out.write(chunk)
    if md5(tmp) != f["computed_md5"]:
        os.remove(tmp)
        raise IOError("md5 mismatch for {}: corrupted download, run again".format(f["name"]))
    os.replace(tmp, path)


def main():
    folder = idsse.IDSSE_DIR
    os.makedirs(folder, exist_ok=True)
    files = file_list()
    print("{} files -> {}".format(len(files), folder))
    for f in sorted(files, key=lambda f: f["size"]):
        fetch(f, folder)
    missing = [m for m in idsse.MATCHES if not idsse.find_files(m, folder)]
    if missing:
        sys.exit("missing matches: {}".format(missing))
    print("done: {} matches".format(len(idsse.MATCHES)))


if __name__ == "__main__":
    main()
