#!/usr/bin/env python3
"""Merge every collected lexicon into Data/lexicons/all_words.tsv.

    Data/venv-train/bin/python Tools/merge_lexicons.py

One line per (word, pinyin): "word<TAB>pin yin<TAB>weight<TAB>source count". The weight is the
largest frequency any source gives (rime-ice/rime-frost/jieba/THUOCL carry frequencies; cell
dictionaries do not and contribute 0). Pinyin missing from a source is filled in with pypinyin.
"""
import glob
import os
import re
import sys
import zipfile
from collections import defaultdict

from pypinyin import lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdict  # noqa: E402
import scel  # noqa: E402
from build_lexicon import parse as parse_rime  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICONS = os.path.join(ROOT, "Data", "lexicons")
OUT = os.path.join(LEXICONS, "all_words.tsv")
HAN = re.compile(r"^[一-鿿]+$")


def main():
    weights = defaultdict(float)
    sources = defaultdict(set)
    syllables = {line.split("\t")[0] for line in open(os.path.join(ROOT, "Resources", "readings.tsv"), encoding="utf-8")}

    def add(source, word, pinyin, weight):
        if not HAN.match(word) or len(word) > 12:
            return
        if not pinyin:
            pinyin = lazy_pinyin(word)
        if len(pinyin) != len(word) or any(s not in syllables for s in pinyin):
            return
        key = (word, " ".join(pinyin))
        weights[key] = max(weights[key], weight)
        sources[key].add(source)

    def done(source):
        print(f"{source}: {sum(1 for s in sources.values() if source in s)} entries so far", flush=True)

    for name in ["rime-ice", "rime-frost", "zhwiki", "moegirl"]:
        for path in glob.glob(os.path.join(LEXICONS, name, "*.dict.yaml")):
            if os.path.basename(path) in ("en.dict.yaml", "en_ext.dict.yaml"):
                continue
            for word, pinyin, weight in parse_rime(path):
                add(name, word, pinyin, weight)
        done(name)

    for path in glob.glob(os.path.join(LEXICONS, "THUOCL", "*.txt")):
        for line in open(path, encoding="utf-8", errors="ignore"):
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1].strip().isdigit():
                add("THUOCL", fields[0].strip(), None, float(fields[1]))
    done("THUOCL")

    for line in open(os.path.join(LEXICONS, "jieba", "dict.txt.big"), encoding="utf-8"):
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            add("jieba", fields[0], None, float(fields[1]))
    done("jieba")

    for path in glob.glob(os.path.join(LEXICONS, "jinghang", "*.zip")):
        with zipfile.ZipFile(path) as archive:
            for line in archive.read("dictionary.txt").decode("utf-8").split("\n"):
                fields = line.split("\t")
                if len(fields) >= 2 and not line.startswith("#"):
                    add("jinghang", fields[1], None, 0.0)
    for path in glob.glob(os.path.join(LEXICONS, "jinghang", "*.txt")):
        for line in open(path, encoding="utf-8", errors="ignore"):
            add("jinghang", line.strip(), None, 0.0)
    done("jinghang")

    for path in glob.glob(os.path.join(LEXICONS, "sogou", "*", "*.scel")):
        try:
            _, entries = scel.parse(path)
        except ValueError:
            continue
        for word, pinyin, _ in entries:
            add("sogou", word, pinyin, 0.0)
    done("sogou")

    # CustomPinyinDictionary, from the Gboard build's database: "word<TAB>pin'yin".
    for line in open(os.path.join(LEXICONS, "CustomPinyinDictionary", "words.tsv"), encoding="utf-8"):
        fields = line.rstrip("\n").split("\t")
        if len(fields) == 2:
            add("custom", fields[0], fields[1].split("'"), 0.0)
    done("custom")

    # QQ dictionaries already converted to text by the ciku-ime mirror: "word<TAB>pin yin<TAB>rank".
    for path in glob.glob(os.path.join(LEXICONS, "qq-mirror", "QQ*", "*", "*.txt")):
        for line in open(path, encoding="utf-8", errors="ignore"):
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and not line.startswith("#"):
                add("qq", fields[0], fields[1].split(), 0.0)
    done("qq")

    for path in glob.glob(os.path.join(LEXICONS, "baidu", "*.bdict")):
        for word, pinyin, _ in bdict.parse(path):
            add("baidu", word, pinyin, 0.0)
    done("baidu")

    with open(OUT, "w", encoding="utf-8") as f:
        for (word, pinyin), weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
            f.write(f"{word}\t{pinyin}\t{weight:g}\t{len(sources[(word, pinyin)])}\n")
    words = {word for word, _ in weights}
    print(f"wrote {OUT}: {len(weights)} entries, {len(words)} distinct words")


if __name__ == "__main__":
    main()
