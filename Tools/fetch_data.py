#!/usr/bin/env python3
"""Downloads every raw input of the lexicon and the training data listed in Data/manifest.tsv, and
checks each against its sha256 (a git source against its commit).

    python3 Tools/fetch_data.py                 # download what is missing, check everything
    python3 Tools/fetch_data.py --check         # check only, download nothing
    python3 Tools/fetch_data.py --only corpus2  # rows whose path starts with this
    python3 Tools/fetch_data.py --data DIR      # another data directory (default Data/)
    python3 Tools/fetch_data.py --update        # take what the sources serve now and record it

Methods of the manifest:
  url             the file at the address (a fixed version)
  latest          the file at the address, which serves only its current version (Sogou and Baidu
                  cell lexicons, Tatoeba): a newer one is reported as changed, not as a failure
  range:A-B       bytes A to B of the file at the address (a slice of a larger file)
  git:COMMIT:P|Q  a shallow clone at COMMIT with only the paths P, Q checked out
  gdrive:FOLDER   a file of a Google Drive folder, fetched whole with gdown --folder
  derive          made from another row: "zip!member: query" runs the SQL query on the SQLite
                  database inside the zip and writes its rows tab-separated
A file already there with the right sha256 is not downloaded again. The manifest records exactly the
data the shipped lexicon and model were built from: a fixed source that no longer matches fails the
check. --update is for taking newer data on purpose: it records the sha256 of what was downloaded.
"""
import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "Data", "manifest.tsv")
AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def rows():
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                path, method, source, digest, fetched = line.rstrip("\n").split("\t")
                yield path, method, source, digest


def add(path, method, source, data=None):
    """Adds a row for a newly downloaded file (path under Data/), with its sha256; an existing row
    is left as it is. Tools/collect_sogou.py and collect_cells.py record their downloads so."""
    import datetime
    if any(row[0] == path for row in rows()):
        return
    digest = sha256(os.path.join(data or os.path.join(ROOT, "Data"), path))
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write("\t".join([path, method, source, digest, datetime.date.today().isoformat()]) + "\n")


def record(updates):
    """Writes the new sha256 and today's date into the manifest rows of `updates` (path -> sha256)."""
    import datetime
    today = datetime.date.today().isoformat()
    with open(MANIFEST, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        fields = line.split("\t")
        if not line.startswith("#") and len(fields) == 5 and fields[0] in updates:
            fields[3], fields[4] = updates[fields[0]], today
            lines[i] = "\t".join(fields)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, byte_range=None):
    headers = {"User-Agent": AGENT}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    if "pinyin.sogou.com" in url:
        headers["Referer"] = "https://pinyin.sogou.com/dict/"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 22)
    os.replace(part, target)


def git_source(url, method, target):
    _, commit, paths = method.split(":", 2)
    if not os.path.isdir(os.path.join(target, ".git")):
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", "-q", url, target], check=True)
        subprocess.run(["git", "-C", target, "sparse-checkout", "set", *paths.split("|")], check=True)
        subprocess.run(["git", "-C", target, "fetch", "-q", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", target, "checkout", "-q", commit], check=True)


def git_ok(method, target):
    commit = method.split(":", 2)[1]
    head = subprocess.run(["git", "-C", target, "rev-parse", "HEAD"], capture_output=True, text=True)
    return head.returncode == 0 and head.stdout.strip() == commit


def derive(source, data, target):
    archive, rest = source.split("!", 1)
    member, query = rest.split(": ", 1)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(os.path.join(data, archive)) as z:
            database = z.extract(member, tmp)
        with sqlite3.connect(database) as db, open(target, "w", encoding="utf-8") as out:
            for row in db.execute(query):
                out.write("\t".join(str(v) for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check, download nothing")
    parser.add_argument("--only", default="", help="rows whose path starts with this")
    parser.add_argument("--data", default=os.path.join(ROOT, "Data"))
    parser.add_argument("--update", action="store_true", help="download again what differs and record its sha256")
    args = parser.parse_args()

    selected = [r for r in rows() if r[0].startswith(args.only)]
    drive_folders = set()
    bad, changed, updates = [], [], {}
    for path, method, source, digest in selected:
        target = os.path.join(args.data, path)
        if method.startswith("git:"):
            if not args.check:
                git_source(source, method, target)
            if not git_ok(method, target):
                bad.append(f"{path}: not at commit {method.split(':')[1]}")
            continue
        if os.path.exists(target) and sha256(target) == digest:
            continue
        if args.check:
            if method == "latest" and os.path.exists(target):
                changed.append(path)
            else:
                bad.append(f"{path}: {'missing' if not os.path.exists(target) else 'sha256 differs'}")
            continue
        if method in ("url", "latest"):
            download(source, target)
        elif method.startswith("range:"):
            download(source, target, method.split(":", 1)[1])
        elif method == "derive":
            derive(source, args.data, target)
        elif method.startswith("gdrive:"):
            folder = method.split(":", 1)[1]
            if folder not in drive_folders:
                drive_folders.add(folder)
                if subprocess.run([sys.executable, "-c", "import gdown"], capture_output=True).returncode != 0:
                    sys.exit("gdown is needed for the Google Drive folder: run this with Data/venv-train/bin/python "
                             "(Training/requirements.txt)")
                os.makedirs(os.path.join(args.data, "corpus2", "openime"), exist_ok=True)
                subprocess.run([sys.executable, "-m", "gdown", "--folder", f"https://drive.google.com/drive/folders/{folder}"],
                               cwd=os.path.join(args.data, "corpus2", "openime"), check=True)
        if not os.path.exists(target):
            bad.append(f"{path}: not downloaded")
        elif sha256(target) != digest:
            if args.update:
                updates[path] = sha256(target)
            elif method == "latest":
                changed.append(path)
            else:
                bad.append(f"{path}: sha256 differs after download (the source has changed)")
    if updates:
        record(updates)
    matching = len(selected) - len(bad) - len(changed) - len(updates)
    print(f"{matching} of {len(selected)} sources in place and matching Data/manifest.tsv")
    if changed:
        print(f"{len(changed)} served only as their current version differ from the recorded one "
              f"(--update records them): " + ", ".join(changed[:5]) + (" …" if len(changed) > 5 else ""))
    if updates:
        print(f"{len(updates)} recorded anew in Data/manifest.tsv")
    for line in bad:
        print("FAIL", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
