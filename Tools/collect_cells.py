#!/usr/bin/env python3
"""Download popular cell dictionaries from the Baidu (.bdict) and QQ (.qpyd) input method sites.

    python3 Tools/collect_cells.py baidu|qq [pages per category, default 5]

Files go to Data/lexicons/<site>/<id>_<name>.<ext>; already downloaded ones are skipped.
Each download is recorded in Data/manifest.tsv (Tools/fetch_data.py).
"""
import html
import os
import re
import subprocess
import sys
import time
import urllib.parse

import fetch_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICONS = os.path.join(ROOT, "Data", "lexicons")
AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def fetch(url, referer):
    # The QQ site answers Python's own HTTP client with 403 but serves curl, so curl does the fetching.
    result = subprocess.run(
        ["curl", "-sSL", "--fail", "-m", "40", "-A", AGENT, "-e", referer, url],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.decode("utf-8", "ignore").strip() or f"curl exit {result.returncode}")
    return result.stdout


def page_text(url, referer):
    raw = fetch(url, referer)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", "ignore")


def save(site, name, extension, url, referer, magic):
    path = os.path.join(LEXICONS, site, re.sub(r"[/\\:]", "_", name) + extension)
    if os.path.exists(path):
        return False
    try:
        data = fetch(url, referer)
    except Exception as error:
        print(f"{name}: {error}", flush=True)
        return False
    if not data.startswith(magic):
        print(f"{name}: unexpected file contents", flush=True)
        return False
    with open(path, "wb") as f:
        f.write(data)
    fetch_data.add(f"lexicons/{site}/{os.path.basename(path)}", "latest", url)
    time.sleep(0.2)
    return True


def baidu(pages):
    site = "https://shurufa.baidu.com"
    # The category links only appear on list pages, so start from one and follow them.
    categories, pending = set(), ["157"]
    while pending:
        category = pending.pop()
        if category in categories:
            continue
        categories.add(category)
        try:
            linked = re.findall(r"dict_list\?cid=(\d+)", page_text(f"{site}/dict_list?cid={category}", site))
        except Exception as error:
            print(f"category {category}: {error}", flush=True)
            continue
        pending.extend(set(linked) - categories)
    print(f"baidu: {len(categories)} categories", flush=True)
    count = 0
    for category in sorted(categories, key=int):
        for page in range(1, pages + 1):
            try:
                listing = page_text(f"{site}/dict_list?cid={category}&orderby=download&page={page}", site)
            except Exception as error:
                print(f"category {category} page {page}: {error}", flush=True)
                break
            found = re.findall(r'dict-name="([^"]*)"\s+dict-innerid="(\d+)"', listing)
            if not found:
                break
            for name, inner in found:
                url = f"{site}/dict_innerid_download?innerid={inner}"
                count += save("baidu", f"{inner}_{html.unescape(name)}", ".bdict", url, site + "/dict", b"biptbdsw")
        print(f"baidu category {category}: {count} downloaded so far", flush=True)


def qq(pages):
    site = "https://cdict.qq.pinyin.cn"
    index = page_text(site + "/v1/", site)
    lists = set(re.findall(r'href="(/v1/list\?sort1=[^"]+)"', index))
    for link in sorted(lists):
        lists |= set(re.findall(r'href="(/v1/list\?sort1=[^"]+)"', page_text(site + urllib.parse.quote(html.unescape(link), safe="/?=&"), site)))
    count = 0
    for link in sorted(link for link in lists if "sort2" in link):
        base = site + urllib.parse.quote(html.unescape(link), safe="/?=&")
        seen = set()
        for page in range(1, pages + 1):
            try:
                listing = page_text(f"{base}&page={page}", site)
            except Exception as error:
                print(f"{link} page {page}: {error}", flush=True)
                break
            found = dict(re.findall(r'/v1/detail\?dict_id=(\d+)"[^>]*>\s*([^<]{1,60})<', listing))
            new = set(found) - seen
            if not new:
                break
            seen |= new
            for dict_id in sorted(new, key=int):
                url = f"{site}/v1/download?dict_id={dict_id}"
                count += save("qq", f"{dict_id}_{html.unescape(found[dict_id]).strip()}", ".qpyd", url, site + "/v1/", b"")
        print(f"qq {html.unescape(link)}: {count} downloaded so far", flush=True)


def main():
    which = sys.argv[1]
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    os.makedirs(os.path.join(LEXICONS, which), exist_ok=True)
    {"baidu": baidu, "qq": qq}[which](pages)


if __name__ == "__main__":
    main()
