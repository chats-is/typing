#!/usr/bin/env python3
"""Build the input method's lexicon, Data/lexicon/lexicon.sqlite.

    Data/venv-train/bin/python Tools/build_lexicon.py

Words come from the merged open lexicons (Data/lexicons/all_words.tsv); how usual each one is comes
from everyday text (Data/lexicon/word_counts.tsv, from Training/count_words.py: chat, real input,
forums first, news last), with a floor for words that several lexicons agree on, so that a real
but rarely written word (退格) still beats a nickname (推哥). Single characters carry one row per
reading, weighted by the readings table. English words carry their usual casing (iphone -> iPhone).

Tables:
  lexicon(key, n, wid, word, logp, abbr, lm0)  key: syllables joined by "'"; n: syllable count;
                                               abbr: first letters; lm0: filled in by Tools/add_lm0.py
  syllables(s)                                 every full syllable
  english(key, word, logp)                     key: the lowercase word; word: as it is written
"""
import json
import math
import os
import sqlite3

from pypinyin import Style, pinyin as to_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICONS = os.path.join(ROOT, "Data", "lexicons")
TABLE = os.path.join(LEXICONS, "all_words.tsv")
COUNTS = os.environ.get("COUNTS", os.path.join(ROOT, "Data", "lexicon", "word_counts_vit.tsv"))  # Tools/count_words_viterbi.py (COUNTS=... overrides)
READINGS = os.path.join(ROOT, "Data", "engine_old", "readings.tsv")
ENGLISH = os.path.join(LEXICONS, "english", "en_50k.txt")
CASING = os.path.join(ROOT, "Data", "lexicon", "english_casing.tsv")  # Tools/make_english_casing.py
OUT = os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite")
VOCAB = os.path.join(ROOT, "Data", "lm9", "vocab.json")  # the language model's characters
LM0 = os.path.join(ROOT, "Data", "lexicon", "lm0.tsv")

MAX_SYLLABLES = 8
# Occurrences per million characters a word is credited with for being listed by this many lexicons.
FLOOR = {2: 0.03, 3: 0.08, 4: 0.2, 5: 0.3, 6: 0.5, 7: 0.7, 8: 1.0}
MIN_SOURCES_SHORT = 2  # words of 1-4 characters
MIN_SOURCES_LONG = 3  # longer phrases, which were not counted
MIN_SOURCES_SEEN = 4  # below this many lexicons a word must also occur in everyday text ...
MIN_RATE_FEW = 0.05  # ... at least this often per million characters
ENGLISH_WORDS = 30000
ENGLISH_FLOOR = 0.02  # per million, for English words never seen in Chinese text
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
    chars = 0
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
            entries[(reading, char)] = max(entries.get((reading, char), 0.0), everyday * share)
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
    english = {}
    with open(CASING, encoding="utf-8") as f:
        for line in f:
            lower, usual, rate = line.rstrip("\n").split("\t")[:3]
            rate = float(rate)
            if not (lower.isalpha() and lower.isascii()):
                continue
            if lower in known or (usual != lower and rate >= ENGLISH_CAPITAL_RATE):
                english[lower] = (usual, rate)
    for lower, count in known.items():
        if lower not in english:
            english[lower] = (lower, ENGLISH_FLOOR * (1 - count / ENGLISH_WORDS))
    print(f"{len(english):,} English words")

    # The language model's no-context score per word, computed once by Tools/add_lm0.py and cached.
    lm0 = {}
    if os.path.exists(LM0):
        with open(LM0, encoding="utf-8") as f:
            for line in f:
                word, value = line.rstrip("\n").split("\t")[:2]
                lm0[word] = float(value)
    print(f"{len(lm0):,} cached lm0 values")

    total = sum(entries.values()) + sum(rate for _, rate in english.values())
    english_rows = [(lower, usual, math.log(rate / total), lm0.get(usual, 0.0)) for lower, (usual, rate) in english.items()]
    wids = {}
    for _key, word in sorted(entries):
        wids.setdefault(word, len(wids) + 1)
    rows = sorted(
        ((key, key.count("'") + 1, wids[word], word, math.log(rate / total), "".join(s[0] for s in key.split("'")), lm0.get(word, 0.0)) for (key, word), rate in entries.items()),
        key=lambda r: (r[0], -r[4]),
    )
    syllables = sorted({s for key, _ in entries for s in key.split("'")})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    db = sqlite3.connect(OUT)
    db.execute("CREATE TABLE lexicon(key TEXT NOT NULL, n INTEGER NOT NULL, wid INTEGER NOT NULL, word TEXT NOT NULL, logp REAL NOT NULL, abbr TEXT NOT NULL, lm0 REAL NOT NULL)")
    db.execute("CREATE TABLE syllables(s TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE english(key TEXT NOT NULL, word TEXT NOT NULL, logp REAL NOT NULL, lm0 REAL NOT NULL DEFAULT 0)")
    db.executemany("INSERT INTO lexicon VALUES (?,?,?,?,?,?,?)", rows)
    db.executemany("INSERT INTO syllables VALUES (?)", ((s,) for s in syllables))
    db.executemany("INSERT INTO english VALUES (?,?,?,?)", english_rows)
    db.execute("CREATE INDEX idx_key ON lexicon(key, logp DESC)")
    db.execute("CREATE INDEX idx_n_key ON lexicon(n, key)")
    db.execute("CREATE INDEX idx_abbr ON lexicon(abbr, logp DESC) WHERE n >= 2")
    db.execute("CREATE INDEX idx_word ON lexicon(word, logp DESC)")
    db.execute("CREATE INDEX idx_wid ON lexicon(wid)")
    db.execute("CREATE INDEX idx_english ON english(key, logp DESC)")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"wrote {OUT}: {len(rows):,} rows, {len(syllables)} syllables, {os.path.getsize(OUT) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
