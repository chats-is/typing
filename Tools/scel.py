#!/usr/bin/env python3
"""Read Sogou cell dictionaries (.scel).

    python3 Tools/scel.py file.scel [...]     # prints "word<TAB>pin yin<TAB>rank" for each entry

As a module: `parse(path)` returns (title, [(word, [syllables], rank)]). The rank is the
entry's position-like number stored in the file, not a corpus frequency.
"""
import struct
import sys

PINYIN_TABLE = 0x1540
# Where the word table starts depends on the format revision in byte 4.
WORD_TABLE = {0x44: 0x2628, 0x45: 0x26C4}


def text(data):
    return data.decode("utf-16-le", "ignore").split("\x00")[0]


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x40\x15\x00\x00" or data[4] not in WORD_TABLE:
        raise ValueError(f"{path}: not a Sogou cell dictionary")
    title = text(data[0x130:0x338])

    syllables = {}
    position = PINYIN_TABLE + 4
    end = WORD_TABLE[data[4]]
    while position + 4 <= end:
        index, length = struct.unpack_from("<HH", data, position)
        position += 4
        if length == 0 or position + length > end:
            break
        syllables[index] = text(data[position:position + length])
        position += length

    entries = []
    position = end
    while position + 4 <= len(data):
        same, pinyin_bytes = struct.unpack_from("<HH", data, position)
        position += 4
        if position + pinyin_bytes > len(data):
            break
        indexes = struct.unpack_from(f"<{pinyin_bytes // 2}H", data, position)
        position += pinyin_bytes
        pinyin = [syllables.get(i, "") for i in indexes]
        for _ in range(same):
            if position + 2 > len(data):
                break
            (word_bytes,) = struct.unpack_from("<H", data, position)
            position += 2
            word = data[position:position + word_bytes].decode("utf-16-le", "ignore")
            position += word_bytes
            (extra_bytes,) = struct.unpack_from("<H", data, position)
            position += 2
            rank = struct.unpack_from("<H", data, position)[0] if extra_bytes >= 2 else 0
            position += extra_bytes
            if word and all(pinyin):
                entries.append((word, pinyin, rank))
    return title, entries


def main():
    for path in sys.argv[1:]:
        title, entries = parse(path)
        print(f"# {title}: {len(entries)} entries", file=sys.stderr)
        for word, pinyin, rank in entries:
            print(f"{word}\t{' '.join(pinyin)}\t{rank}")


if __name__ == "__main__":
    main()
