#!/usr/bin/env python3
"""The basic-input tests: the common words, each typed every way people type it.

    python3 Training/make_basic_eval.py --top 10000

Every common word is typed in all its forms (each syllable in full or by its initial; zh/ch/sh
also as z/c/s) and every common English word in lowercase. What the user actually types is a
string of letters, and one string often has several readings ("wan" is 玩 in full and 我爱你 by
initials), so the expected first candidate for a string is the word that is most usual *as
typed that way*: the lexicon's frequency, less the typing model's cost of the form (INITIAL_COST
per syllable typed by its initial, ENGLISH_COST for an English word in pinyin mode) - the same
assumptions the decoder makes, stated here once.

Writes to Data/:
  eval_full_first.tsv      strings whose best reading is a word typed in full      -> that word (must be first)
  eval_initials_first.tsv  strings whose best reading is a word typed by initials  -> that word (must be first)
  eval_mixed_first.tsv     ... a word typed in a mix of full syllables and initials -> that word (must be first)
  eval_english_first.tsv   strings whose best reading is an English word -> as it is usually written (must be first)
  eval_full_page.tsv       full pinyin of every common word of 2+ characters -> the word (must be on the first page)
  eval_char_page.tsv       full pinyin of every common single character -> the character (first page where possible)
  eval_user_first.tsv      the user's own examples (Data/basic_user.tsv: keys<TAB>word), kept as they are

"Common" is the top --top words (1-4 characters) of the lexicon (Data/lexicon/lexicon.sqlite) by its
everyday frequency, and the top --english English words. The rows are shuffled so a --limit run is a fair sample.
"""
import argparse
import itertools
import os
import random
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
EVAL_DIR = os.environ.get("EVAL_DIR", DATA)  # where the eval_*.tsv files go
INITIALS = ("zh", "ch", "sh")
INITIAL_COST = float(os.environ.get("INITIAL_COST", "3.0"))  # log-probability cost of typing a syllable by its initial (LexiconDecoder.initialPenalty)
ENGLISH_COST = float(os.environ.get("ENGLISH_COST", "1.0"))  # ... of typing an English word in pinyin mode (LexiconDecoder.englishPenalty)


def initials(syllable):
    for i in INITIALS:
        if syllable.startswith(i):
            return [i, i[0]]  # people type sh or s
    return [syllable[0]]


def forms(syllables):
    """(letters, kind, initials used) for every way of typing the word."""
    options = [[(s, "full")] + [(i, "initial") for i in initials(s)] for s in syllables]
    out = {}
    for choice in itertools.product(*options):
        letters = "".join(k for k, _ in choice)
        used = sum(1 for _, kind in choice if kind == "initial")
        kind = "full" if used == 0 else "initials" if used == len(choice) else "mixed"
        if letters not in out or out[letters][1] > used:
            out[letters] = (kind, used)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10000)
    parser.add_argument("--english", type=int, default=3000)
    parser.add_argument("--lexicon", default=os.path.join(DATA, "lexicon", "lexicon.sqlite"))
    args = parser.parse_args()

    # Every word of the lexicon competes for a string of letters; only strings whose best reading
    # is a common word (the top --top, or a top --english English word) are tested.
    db = sqlite3.connect(args.lexicon)
    seen = set()
    common = set()
    best = {}  # letters -> (score, word, kind): the word most usual as typed with these letters
    page, chars = [], []
    for word, key, logp in db.execute("SELECT word, key, logp FROM lexicon WHERE n <= 4 ORDER BY logp DESC"):
        if not all("一" <= c <= "鿿" for c in word):
            continue
        first = word not in seen  # a word's usual reading is its best row; other readings count too
        seen.add(word)
        if first and len(common) < args.top:
            common.add(word)
        syllables = key.split("'")
        for letters, (kind, used) in forms(syllables).items():
            # Nobody types a one-syllable word by its initial alone: that is an unfinished syllable.
            if len(syllables) == 1 and used:
                continue
            score = logp - used * INITIAL_COST
            if letters not in best or best[letters][0] < score:
                best[letters] = (score, word, kind)
            if kind == "full" and first and word in common:
                (page if len(word) >= 2 else chars).append((letters, word))
    print(f"{len(common)} common words of {len(seen)}")
    for letters, word, logp in db.execute("SELECT key, word, logp FROM english WHERE length(key) >= 2 ORDER BY logp DESC"):
        score = logp - ENGLISH_COST
        if letters not in best or best[letters][0] < score:
            best[letters] = (score, word, "english")
    # Common English words: the English word list's, or longer capitalised names; two-letter
    # abbreviations (UT, GI) are not words anyone expects first.
    listed = set()
    with open(os.path.join(DATA, "lexicons", "english", "en_50k.txt"), encoding="utf-8") as f:
        for count, line in enumerate(f):
            if count >= 30000:
                break
            listed.add(line.split()[0].lower())
    for letters, word, _ in db.execute("SELECT key, word, logp FROM english ORDER BY logp DESC LIMIT ?", (args.english,)):
        if letters in listed or len(letters) >= 3:
            common.add(word)

    rng = random.Random(7)
    sets = {"full": [], "initials": [], "mixed": [], "english": []}
    for letters, (_, word, kind) in best.items():
        if word in common:
            sets[kind].append((letters, word))
    for name, rows in sets.items():
        rng.shuffle(rows)
        path = os.path.join(EVAL_DIR, f"eval_{name}_first.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for letters, word in rows:
                f.write(f"{letters}\t{word}\n")
        print(f"{path}: {len(rows)} rows")
    for name, rows in (("full_page", page), ("char_page", chars)):
        rng.shuffle(rows)
        with open(os.path.join(EVAL_DIR, f"eval_{name}.tsv"), "w", encoding="utf-8") as f:
            for letters, word in rows:
                f.write(f"{letters}\t{word}\n")
        print(f"eval_{name}.tsv: {len(rows)} rows")

    user = os.path.join(DATA, "basic_user.tsv")
    if os.path.exists(user):
        rows = [line.rstrip("\n").split("\t") for line in open(user, encoding="utf-8") if "\t" in line]
        with open(os.path.join(EVAL_DIR, "eval_user_first.tsv"), "w", encoding="utf-8") as f:
            for keys, word in rows:
                f.write(f"{keys}\t{word}\n")
        print(f"eval_user_first.tsv: {len(rows)} rows")


if __name__ == "__main__":
    main()
