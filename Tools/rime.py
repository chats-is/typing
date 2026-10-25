"""Reads Rime dictionaries (*.dict.yaml): the entries after the "..." line, "word<TAB>pin yin<TAB>weight"."""

VOWELS = set("aeiouv")


def parse(path):
    """Yield (word, [syllables], weight) from a rime dict.yaml file."""
    in_body = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not in_body:
                in_body = line.strip() == "..."
                continue
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            word = fields[0].strip()
            syllables = fields[1].strip().lower().replace("ü", "v").split()
            try:
                weight = float(fields[2]) if len(fields) > 2 and fields[2].strip() else 1.0
            except ValueError:
                weight = 1.0
            if not word or not syllables:
                continue
            # Interjection-only syllables (m, n, ng, hm ...) wreck segmentation.
            if any(not s.isascii() or not s.isalpha() or not (VOWELS & set(s)) for s in syllables):
                continue
            yield word, syllables, weight
