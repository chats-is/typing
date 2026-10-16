#!/usr/bin/env python3
"""Build Resources/readings.tsv, the character-to-pinyin table the model decoder needs.

One line per reading: "syllable<TAB>character<TAB>weight". It is taken from the single
character dictionary of rime-ice (Data/raw/8105.dict.yaml, fetched by build_lexicon.py).
"""
import os

from build_lexicon import RAW_DIR, ROOT, fetch, parse

OUT = os.path.join(ROOT, "Resources", "readings.tsv")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    rows = {}
    for word, syllables, weight in parse(fetch("8105.dict.yaml")):
        if len(word) == 1 and len(syllables) == 1:
            key = (syllables[0], word)
            rows[key] = max(weight, rows.get(key, 0.0))
    with open(OUT, "w", encoding="utf-8") as f:
        for (syllable, character), weight in sorted(rows.items(), key=lambda r: (r[0][0], -r[1])):
            f.write(f"{syllable}\t{character}\t{weight:g}\n")
    print(f"wrote {OUT}: {len(rows)} readings, {len({s for s, _ in rows})} syllables")


if __name__ == "__main__":
    main()
