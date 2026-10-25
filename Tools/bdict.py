#!/usr/bin/env python3
"""Read Baidu input method cell dictionaries (.bdict / .bcd).

    python3 Tools/bdict.py file.bdict [...]    # prints "word<TAB>pin yin<TAB>frequency"

As a module: `parse(path)` returns [(word, [syllables], frequency)]. Entries whose pinyin
cannot be decoded (English words, symbols) are skipped.
"""
import struct
import sys

INITIALS = ["c", "d", "b", "f", "g", "h", "ch", "j", "k", "l", "m", "n", "", "p", "q", "r", "s", "t", "sh", "zh", "w", "x", "y", "z"]
FINALS = ["uang", "iang", "iong", "ang", "eng", "ian", "iao", "ing", "ong", "uai", "uan", "ai", "an", "ao", "ei", "en", "er",
          "ua", "ie", "in", "iu", "ou", "ia", "ue", "ui", "un", "uo", "a", "e", "i", "o", "u", "v"]
ENTRIES = 0x350


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"biptbdsw":
        raise ValueError(f"{path}: not a Baidu cell dictionary")
    entries = []
    position = ENTRIES
    while position + 4 <= len(data):
        count, frequency = struct.unpack_from("<HH", data, position)
        position += 4
        if count == 0 or count > 32:
            break
        if position + count * 4 > len(data):
            break
        codes = data[position:position + count * 2]
        position += count * 2
        word = data[position:position + count * 2].decode("utf-16-le", "ignore")
        position += count * 2
        syllables = []
        for i in range(count):
            initial, final = codes[2 * i], codes[2 * i + 1]
            if initial >= len(INITIALS) or final >= len(FINALS):
                syllables = None
                break
            syllables.append(INITIALS[initial] + FINALS[final])
        if syllables and len(word) == count:
            entries.append((word, syllables, frequency))
    return entries


def main():
    for path in sys.argv[1:]:
        entries = parse(path)
        print(f"# {path}: {len(entries)} entries", file=sys.stderr)
        for word, syllables, frequency in entries:
            print(f"{word}\t{' '.join(syllables)}\t{frequency}")


if __name__ == "__main__":
    main()
