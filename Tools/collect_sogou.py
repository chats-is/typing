#!/usr/bin/env python3
"""Download the most popular Sogou cell dictionaries (.scel) of every top-level category.

    python3 Tools/collect_sogou.py [pages per category, default 8]

Files go to Data/lexicons/sogou/<category>/<id>_<name>.scel; already downloaded ones are skipped.
Each download is recorded in Data/manifest.tsv (Tools/fetch_data.py).
"""
import html
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import fetch_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Data", "lexicons", "sogou")
SITE = "https://pinyin.sogou.com"
AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
CATEGORIES = {
    167: "城市信息", 1: "自然科学", 76: "社会科学", 96: "工程应用", 127: "农林渔畜", 132: "医学医药",
    436: "电子游戏", 154: "艺术设计", 389: "生活百科", 367: "运动休闲", 31: "人文科学", 403: "娱乐休闲",
}
LINK = re.compile(r'download_cell\.php\?id=(\d+)&(?:amp;)?name=([^"\']+)')


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "Referer": SITE + "/dict/"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    downloaded = 0
    for category, label in CATEGORIES.items():
        directory = os.path.join(OUT, label)
        os.makedirs(directory, exist_ok=True)
        for page in range(1, pages + 1):
            try:
                listing = fetch(f"{SITE}/dict/cate/index/{category}/default/{page}").decode("utf-8", "ignore")
            except Exception as error:
                print(f"{label} page {page}: {error}", flush=True)
                break
            links = LINK.findall(listing)
            if not links:
                break
            for cell_id, quoted in links:
                name = urllib.parse.unquote(html.unescape(quoted))
                path = os.path.join(directory, f"{cell_id}_{re.sub(r'[/\\\\:]', '_', name)}.scel")
                if os.path.exists(path):
                    continue
                url = f"{SITE}/d/dict/download_cell.php?id={cell_id}&name={quoted}"
                try:
                    data = fetch(url)
                except Exception as error:
                    print(f"{name}: {error}", flush=True)
                    continue
                if data[:4] != b"\x40\x15\x00\x00":
                    print(f"{name}: not a cell dictionary", flush=True)
                    continue
                with open(path, "wb") as f:
                    f.write(data)
                fetch_data.add(f"lexicons/sogou/{label}/{os.path.basename(path)}", "latest", url)
                downloaded += 1
                time.sleep(0.2)
        print(f"{label}: done, {downloaded} downloaded so far", flush=True)


if __name__ == "__main__":
    main()
