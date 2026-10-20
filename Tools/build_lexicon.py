#!/usr/bin/env python3
"""Build the input method's lexicon, Data/lexicon/lexicon.sqlite.

    Data/venv-train/bin/python Tools/build_lexicon.py

Words come from the merged open lexicons (Data/lexicons/all_words.tsv); how usual each one is comes
from everyday text (Data/lexicon/word_counts_vit3.tsv, from Tools/count_words_viterbi.py: chat, real input,
forums first, news last), with a floor for words that several lexicons agree on, so that a real
but rarely written word (退格) still beats a nickname (推哥). Single characters carry one row per
reading, weighted by the readings table. English words carry their usual casing (iphone -> iPhone).

Tables:
  lexicon(key, n, wid, word, logp, abbr)       key: syllables joined by "'"; n: syllable count;
                                               abbr: first letters
  syllables(s)                                 every full syllable
  english(key, word, logp)                     key: the lowercase word; word: as it is written
  rare_characters(key, word, logp)             characters only the full list of a lone syllable shows
  symbols(key, name, mark, rank)               marks offered right after their name (yinhao: 引号 「」 『』)
"""
import json
import math
import os
import sqlite3

from pypinyin import Style, pinyin as to_pinyin
from pypinyin.constants import PINYIN_DICT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICONS = os.path.join(ROOT, "Data", "lexicons")
TABLE = os.path.join(LEXICONS, "all_words.tsv")
COUNTS = os.environ.get("COUNTS", os.path.join(ROOT, "Data", "lexicon", "word_counts_vit3.tsv"))  # Tools/count_words_viterbi.py, 3 EM passes (COUNTS=... overrides)
READINGS = os.path.join(ROOT, "Data", "lexicon", "readings.tsv")  # character → readings table
ENGLISH = os.path.join(LEXICONS, "english", "en_50k.txt")
CASING = os.path.join(ROOT, "Data", "lexicon", "english_casing.tsv")  # Tools/make_english_casing.py
OUT = os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite")
VOCAB = os.path.join(ROOT, "Data", "lm9", "vocab.json")  # the language model's characters

# The Unicode blocks of Chinese characters (GB 18030 maps all of them): the basic block, extensions
# A to H and the compatibility ideographs.
HANZI_BLOCKS = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2EBEF), (0x30000, 0x323AF), (0xF900, 0xFAFF)]
# Marks typed by their names (two syllables or more: one-syllable names such as du would crowd
# the usual characters): yinhao offers 「」『』 right after 引号, including the marks no key gives
# (the keys type the system Pinyin's ones; GB/T 19246-2003 table 1 has more). They are kept in a
# table of their own, symbols, and placed after their name once the candidates are ranked: as
# words they took part in the ranking, and after some text the model put "，" before 逗号.
SYMBOLS = {
    # Punctuation (GB/T 15834 names).
    "dou'hao": ("逗号", ["，"]),
    "ju'hao": ("句号", ["。", "．"]),
    "wen'hao": ("问号", ["？"]),
    "tan'hao": ("叹号", ["！"]),
    "gan'tan'hao": ("感叹号", ["！"]),
    "mao'hao": ("冒号", ["："]),
    "fen'hao": ("分号", ["；"]),
    "dun'hao": ("顿号", ["、"]),
    "yin'hao": ("引号", ["「」", "『』", "“”", "‘’"]),
    "shuang'yin'hao": ("双引号", ["“”", "『』"]),
    "dan'yin'hao": ("单引号", ["‘’", "「」"]),
    "kuo'hao": ("括号", ["（）", "【】", "〔〕", "［］", "｛｝"]),
    "yuan'kuo'hao": ("圆括号", ["（）"]),
    "fang'kuo'hao": ("方括号", ["［］", "【】"]),
    "liu'jiao'kuo'hao": ("六角括号", ["〔〕"]),
    "fang'tou'kuo'hao": ("方头括号", ["【】"]),
    "hua'kuo'hao": ("花括号", ["｛｝"]),
    "shu'ming'hao": ("书名号", ["《》", "〈〉"]),
    "sheng'lve'hao": ("省略号", ["……"]),
    "po'zhe'hao": ("破折号", ["——"]),
    "lian'jie'hao": ("连接号", ["—", "－", "～"]),
    "bo'lang'hao": ("波浪号", ["～"]),
    "jian'ge'hao": ("间隔号", ["·"]),
    "xie'gang": ("斜杠", ["／"]),
    "fan'xie'gang": ("反斜杠", ["＼"]),
    # Numbers and units.
    "bai'fen'hao": ("百分号", ["％"]),
    "qian'fen'hao": ("千分号", ["‰"]),
    "she'shi'du": ("摄氏度", ["℃"]),
    "du'shu": ("度数", ["°"]),
    "cheng'hao": ("乘号", ["×"]),
    "chu'hao": ("除号", ["÷"]),
    "jia'hao": ("加号", ["＋"]),
    "jian'hao": ("减号", ["－"]),
    "deng'hao": ("等号", ["＝"]),
    "zheng'fu'hao": ("正负号", ["±"]),
    "yue'deng'yu": ("约等于", ["≈"]),
    "bu'deng'yu": ("不等于", ["≠"]),
    "da'yu'deng'yu": ("大于等于", ["≥"]),
    "xiao'yu'deng'yu": ("小于等于", ["≤"]),
    # Money.
    "ren'min'bi": ("人民币", ["￥"]),
    "mei'yuan": ("美元", ["$"]),
    "ou'yuan": ("欧元", ["€"]),
    "ying'bang": ("英镑", ["£"]),
    # Marks.
    "jian'tou": ("箭头", ["→", "←", "↑", "↓"]),
    "dui'hao": ("对号", ["√", "✓"]),
    "cuo'hao": ("错号", ["×", "✗"]),
    "xing'hao": ("星号", ["＊", "★", "☆", "※"]),
    "yuan'quan": ("圆圈", ["○", "●"]),
    "fang'kuang": ("方框", ["□", "■"]),
    "san'jiao'xing": ("三角形", ["△", "▲"]),
}
MAX_SYLLABLES = 8
# Occurrences per million characters a word is credited with for being listed by this many lexicons.
FLOOR = {2: 0.03, 3: 0.08, 4: 0.2, 5: 0.3, 6: 0.5, 7: 0.7, 8: 1.0}
MIN_SOURCES_SHORT = 2  # words of 1-4 characters
MIN_SOURCES_LONG = 3  # longer phrases, which were not counted
MIN_SOURCES_SEEN = 4  # below this many lexicons a word must also occur in everyday text ...
MIN_RATE_FEW = 0.05  # ... at least this often per million characters
ENGLISH_WORDS = 50000  # the whole frequency list (30000 left out common words such as dialog, #30421)
ENGLISH_FLOOR = 0.02  # per million, for English words never seen in Chinese text
# How an ordinary English word (one the English frequency list knows) is offered: in lowercase first,
# unless one other casing is written this share of the time or more inside Chinese text (OK); any
# other casing written at least ENGLISH_OTHER_FORM of the time comes as a further candidate (Hello,
# IT). Deliberate mixed case (macOS, iPhone) keeps the casing make_english_casing.py chose.
ENGLISH_MOSTLY = 0.6
ENGLISH_OTHER_FORM = 0.2
ENGLISH_CAPITAL_RATE = 0.1  # per million: a capitalised letter string this common is a name or abbreviation (NBA, iPhone)


def floor_for(sources):
    return FLOOR.get(min(sources, 8), 0.0) if sources >= 2 else 0.0


def main():
    counts = {}
    with open(COUNTS, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                counts[parts[0]] = float(parts[1])
    print(f"{len(counts):,} counted words")

    # (key, word) -> rate per million; readings of one word share its count.
    entries = {}
    reading_use = {}
    with open(TABLE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            word, pinyin, sources = parts[0], parts[1], int(parts[3])
            syllables = pinyin.split()
            if len(syllables) != len(word) or len(word) > MAX_SYLLABLES or len(word) == 1:
                continue
            if not all(s.isascii() and s.isalpha() for s in syllables):
                continue
            if sources < (MIN_SOURCES_SHORT if len(word) <= 4 else MIN_SOURCES_LONG):
                continue
            # Listed by few lexicons and never seen in everyday text: most of these are noise.
            if sources < MIN_SOURCES_SEEN and counts.get(word, 0.0) < MIN_RATE_FEW:
                continue
            rate = counts.get(word, 0.0) + floor_for(sources)
            if len(word) > 4 and word not in counts:
                rate = floor_for(sources) * 3  # phrases were not counted; trust the lexicons a little more
            key = "'".join(syllables)
            if rate > entries.get((key, word), 0.0):
                entries[(key, word)] = rate
            # How often each character is read each way, from the words it is in: 长 is zhang in
            # 长大 and 成长, chang in 长城; the readings table only knows the character's total.
            for c, syllable in zip(word, syllables):
                reading_use.setdefault(c, {})
                reading_use[c][syllable] = reading_use[c].get(syllable, 0.0) + rate
    print(f"{len(entries):,} word readings")

    # Single characters: one row per reading, the readings table's weights split the character's count.
    by_char = {}
    with open(READINGS, encoding="utf-8") as f:
        for line in f:
            reading, char, weight = line.rstrip("\n").split("\t")[:3]
            if len(char) == 1 and reading.isascii() and reading.isalpha():
                by_char.setdefault(char, []).append((reading, float(weight)))
    # Every character the language model knows must be typeable: those the readings table lacks
    # (traditional forms, variants, dialect characters) get pypinyin's readings and a small rate.
    with open(VOCAB, encoding="utf-8") as f:
        vocab = json.load(f)
    added = 0
    for token in vocab:
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff" and token not in by_char:
            options = [r for r in to_pinyin(token, style=Style.NORMAL, heteronym=True)[0] if r.isascii() and r.isalpha()]
            if options:
                by_char[token] = [(r, 1.0) for r in options]
                added += 1
    print(f"{added} characters added from pypinyin")
    # Every Chinese character with a reading must be typeable (GB/T 19246-2003 4.2: the whole GB 18030
    # set): the characters pypinyin reads that neither the readings table nor the model's vocabulary
    # has, at the same small rate. They go into a table of their own, rare_characters, which only
    # the full list of a syllable typed alone reads: in the lexicon table they made every prefix and
    # initials lookup scan tens of thousands of extra rows (typing was 5-25% slower).
    # Only readings of the syllables already known (those of 《汉语拼音方案》, 4.3.3): dialect readings
    # (din, wong) and interjections (m, ng) would make single letters and common runs whole syllables.
    known_syllables = {r for readings in by_char.values() for r, _ in readings}
    rare_chars = set()
    for code in sorted(PINYIN_DICT):
        char = chr(code)
        if char in by_char or not any(a <= code <= b for a, b in HANZI_BLOCKS):
            continue
        options = [r for r in to_pinyin(char, style=Style.NORMAL, heteronym=True)[0] if r in known_syllables]
        if options:
            by_char[char] = [(r, 1.0) for r in options]
            rare_chars.add(char)
    print(f"{len(rare_chars)} more characters from pypinyin, to type every character with a reading")
    # ê is typed "ea" on the keyboard (GF 3006-2001, 4.3.4). The readings table writes it as e and
    # pypinyin as ê (not ASCII), so it had no key: the characters pypinyin reads ê get the reading ea.
    read_e_circumflex = 0
    for char, readings in by_char.items():
        if "ê" in to_pinyin(char, style=Style.NORMAL, heteronym=True)[0] and all(r != "ea" for r, _ in readings):
            readings.append(("ea", 1.0))
            read_e_circumflex += 1
    print(f"{read_e_circumflex} characters read ê, typed ea")
    chars = 0
    rare_entries = {}
    for char, readings in by_char.items():
        everyday = counts.get(char, 0.0) + (0.5 if char in reading_use or counts.get(char, 0.0) > 0 else 0.05)
        use = reading_use.get(char, {})
        listed = [r for r, _ in readings]
        total = sum(use.get(r, 0.0) for r in listed)
        for reading, _weight in readings:
            if total > 0:
                share = max(use.get(reading, 0.0) / total, 0.002)  # a reading no word uses is rare, not absent
            else:
                share = 1.0 / len(readings)
            target = rare_entries if char in rare_chars else entries
            target[(reading, char)] = max(target.get((reading, char), 0.0), everyday * share)
            chars += 1
    print(f"{chars:,} character readings")

    # English: as written inside Chinese text and how often, on the same per-million scale; the
    # English frequency list only adds coverage for words never seen there.
    # Letter strings in Chinese text are English words only when the English word list knows them,
    # or when they are usually written with capitals (iPhone, QQ, NBA): "hxy", "pz" are pinyin initials
    # written as letters, not words.
    known = {}
    with open(ENGLISH, encoding="utf-8") as f:
        for count, line in enumerate(f):
            word = line.split()[0]
            if count >= ENGLISH_WORDS:
                break
            if word.isalpha() and word.isascii():
                known.setdefault(word.lower(), count)
    english = {}  # lower -> [(written, rate)], the first the commonest way of writing it
    with open(CASING, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            lower, usual, rate = fields[0], fields[1], float(fields[2])
            if not (lower.isalpha() and lower.isascii()):
                continue
            if lower in known:
                forms = [(form, float(share)) for form, share in (item.rsplit(":", 1) for item in fields[3].split())] if len(fields) > 3 else []
                deliberate = usual != lower and not usual.isupper() and not usual.istitle()
                if deliberate or not forms:
                    english[lower] = [(usual, rate)]
                    continue
                top, top_share = max(((f, s) for f, s in forms if f != lower), key=lambda x: x[1], default=(lower, 0.0))
                first = top if top_share >= ENGLISH_MOSTLY else lower
                written = [(first, rate)]
                written += [(form, rate * share) for form, share in forms if form != first and share >= ENGLISH_OTHER_FORM]
                if first != lower and all(form != lower for form, _ in written):
                    written.append((lower, rate * dict(forms).get(lower, 0.0)))
                english[lower] = [(form, r) for form, r in written if r > 0]
            elif usual != lower and rate >= ENGLISH_CAPITAL_RATE:
                english[lower] = [(usual, rate)]
    for lower, count in known.items():
        if lower not in english:
            english[lower] = [(lower, ENGLISH_FLOOR * (1 - count / ENGLISH_WORDS))]
    print(f"{len(english):,} English words, {sum(len(v) for v in english.values()):,} ways of writing them")


    # A name the lexicon lacks (六角括号) is added as a word, so its marks have something to follow.
    symbol_words = set()
    symbol_rows = []
    for key, (name, marks) in SYMBOLS.items():
        if (key, name) not in entries:
            entries[(key, name)] = FLOOR[2]
            symbol_words.add(name)
        symbol_rows += [(key, name, mark, rank) for rank, mark in enumerate(marks)]
    print(f"{len(symbol_rows)} marks typed by their names ({len(symbol_words)} names added as words)")

    total = sum(entries.values()) + sum(rate for written in english.values() for _, rate in written)
    english_rows = [(lower, form, math.log(rate / total)) for lower, written in english.items() for form, rate in written]
    rare_rows = sorted(((key, word, math.log(rate / total)) for (key, word), rate in rare_entries.items()), key=lambda r: (r[0], -r[2]))
    # The added names' ids come after all others, so the ids the bigram model was counted on stay.
    wids = {}
    for _key, word in sorted(entries):
        if word not in symbol_words:
            wids.setdefault(word, len(wids) + 1)
    for _key, word in sorted(entries):
        wids.setdefault(word, len(wids) + 1)
    rows = sorted(
        ((key, key.count("'") + 1, wids[word], word, math.log(rate / total), "".join(s[0] for s in key.split("'"))) for (key, word), rate in entries.items()),
        key=lambda r: (r[0], -r[4]),
    )
    syllables = sorted({s for key, _ in entries for s in key.split("'")})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    db = sqlite3.connect(OUT)
    db.execute("CREATE TABLE lexicon(key TEXT NOT NULL, n INTEGER NOT NULL, wid INTEGER NOT NULL, word TEXT NOT NULL, logp REAL NOT NULL, abbr TEXT NOT NULL)")
    db.execute("CREATE TABLE syllables(s TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE english(key TEXT NOT NULL, word TEXT NOT NULL, logp REAL NOT NULL)")
    db.execute("CREATE TABLE rare_characters(key TEXT NOT NULL, word TEXT NOT NULL, logp REAL NOT NULL)")
    db.executemany("INSERT INTO rare_characters VALUES (?,?,?)", rare_rows)
    db.execute("CREATE INDEX idx_rare ON rare_characters(key, logp DESC)")
    db.execute("CREATE TABLE symbols(key TEXT NOT NULL, name TEXT NOT NULL, mark TEXT NOT NULL, rank INTEGER NOT NULL)")
    db.executemany("INSERT INTO symbols VALUES (?,?,?,?)", symbol_rows)
    db.execute("CREATE INDEX idx_symbols ON symbols(key, name, rank)")
    db.executemany("INSERT INTO lexicon VALUES (?,?,?,?,?,?)", rows)
    db.executemany("INSERT INTO syllables VALUES (?)", ((s,) for s in syllables))
    db.executemany("INSERT INTO english VALUES (?,?,?)", english_rows)
    db.execute("CREATE INDEX idx_key ON lexicon(key, logp DESC)")
    db.execute("CREATE INDEX idx_n_key ON lexicon(n, key)")
    db.execute("CREATE INDEX idx_abbr ON lexicon(abbr, logp DESC) WHERE n >= 2")
    db.execute("CREATE INDEX idx_english ON english(key, logp DESC)")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"wrote {OUT}: {len(rows):,} rows, {len(rare_rows):,} rare characters' readings, {len(syllables)} syllables, {os.path.getsize(OUT) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
