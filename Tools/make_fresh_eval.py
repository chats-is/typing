#!/usr/bin/env python3
"""Fresh sentence sets: the ones that decide between models.

    Data/venv-train/bin/python Tools/make_fresh_eval.py [source ...]

From the cleaned held-out documents (Data/prepared-clean/holdout_*.txt): documents no older eval
set took sentences from (forum and novels have too few: other sentences of those), sentences that
occur nowhere in the cleaned training streams (Data/prepared-clean/*.bin, checked exactly). Same
sentence rule and context as Tools/make_genre_eval.py, seed 2026. Writes Data/fresh/eval_<source>.tsv
(300 each; novels 110) and, for dialogue, Data/fresh/general_dialogue.tsv: the learning check, from
later documents only, so no document gives sentences to both. Never tune anything on these.
"""
import glob, json, os, random, re, sys
import numpy as np
from pypinyin import lazy_pinyin

HAN_RUN = re.compile(r"[一-鿿]+")
CONTEXT_CHARS, MIN_CONTEXT, WANT, POOL = 60, 4, 300, 900
CHECK = 600  # learning check sentences (dialogue)
syllables = {l.split("\t")[0] for l in open("Data/lexicon/readings.tsv", encoding="utf-8")}
vocab = json.load(open("Data/prepared-clean/vocab.json")); ids = {t: i for i, t in enumerate(vocab)}

used_targets, used_pairs = set(), []
for f in glob.glob("Data/eval*.tsv"):
    for line in open(f, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            used_targets.add(p[1])
            if len(p) >= 3 and p[2]:
                used_pairs.append(p[2].replace("\\n", "\n")[-10:] + p[1])

only = sys.argv[1:]
sources = {name: f"Data/prepared-clean/holdout_{name}.txt" for name in ["dialogue", "news", "forum", "mixed", "novels", "web", "touchpal"]}
pools = {}
for name, path in sources.items():
    if only and name not in only: continue
    docs = [d for d in open(path, encoding="utf-8").read().split("\n\n") if d.strip()]
    rng = random.Random(2026); rng.shuffle(docs)
    out = []
    for number, doc in enumerate(docs):
        few = name in ("forum", "novels")  # too few held-out documents: used ones, other sentences
        if not few and any(u in doc for u in used_pairs if u[-5:] in doc):
            continue  # a document an existing set took sentences from
        if name == "touchpal":
            s = doc.strip()
            cands = [(s, "")] if HAN_RUN.fullmatch(s) and 2 <= len(s) <= 18 else []
        else:
            ms = [m for m in HAN_RUN.finditer(doc) if 5 <= len(m.group()) <= 18 and m.start() >= MIN_CONTEXT]
            rng.shuffle(ms)
            cands = [(m.group(), doc[max(0, m.start() - CONTEXT_CHARS):m.start()]) for m in ms[:5 if few else 3 if name == "dialogue" else 1]]
        for s, ctx in cands:
            py = lazy_pinyin(s)
            if s in used_targets or len(py) != len(s) or any(x not in syllables for x in py) or any(c not in ids for c in s):
                continue
            out.append((s, ctx, " ".join(py), number))
        if len(out) >= (3000 if name == "dialogue" else POOL): break
    pools[name] = out
    print(name, "candidates", len(out), flush=True)

# Which candidates occur in any training stream: first-5-token hash, then an exact check.
cands = sorted({c[0] for v in pools.values() for c in v})
enc = {s: np.array([ids[c] for c in s], dtype=np.uint16) for s in cands}
K = np.uint64(1000003)
def h5(a):
    a = a.astype(np.uint64); return (((a[:-4] * K + a[1:-3]) * K + a[2:-2]) * K + a[3:-1]) * K + a[4:]
first = {}
for s in cands:
    if len(enc[s]) >= 5: first.setdefault(int(h5(enc[s])[0]), []).append(s)
keys = np.array(sorted(first), dtype=np.uint64)
seen = set(s for s in cands if len(s) < 5)  # short ones: check directly below
for path in sorted(glob.glob("Data/prepared-clean/*.bin")):
    if path.endswith(("val.bin", "val.raw.bin")): continue
    data = np.memmap(path, dtype="<u2", mode="r")
    step = 50_000_000
    for start in range(0, len(data) - 4, step):
        chunk = np.asarray(data[start:start + step + 32])
        hs = h5(chunk)
        idx = np.searchsorted(keys, hs); idx[idx == len(keys)] = 0
        for pos in np.nonzero(keys[idx] == hs)[0]:
            for s in first[int(hs[pos])]:
                e = enc[s]
                if s not in seen and pos + len(e) <= len(chunk) and np.array_equal(chunk[pos:pos + len(e)], e):
                    seen.add(s)
    print("scanned", os.path.basename(path), "found so far", len(seen), flush=True)
# Short TouchPal messages (< 5 characters) are kept out: too common to be checked this way.
def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for s, c, p, _ in rows:
            f.write(p + "\t" + s + "\t" + c.replace("\t", " ").replace("\\", "/").replace("\n", "\\n") + "\n")

for name, v in pools.items():
    usable, taken = [c for c in v if c[0] not in seen], set()
    kept = []
    for c in usable:
        if c[0] not in taken and len(kept) < WANT: kept.append(c); taken.add(c[0])
    write(f"Data/fresh/eval_{name}.tsv", kept)
    print(f"Data/fresh/eval_{name}.tsv: {len(kept)} sentences ({len(v) - len(usable)} of {len(v)} candidates occur in training)")
    if name == "dialogue":
        # The learning check: later documents only, so no document gives sentences to both sets.
        last = max(c[3] for c in kept)
        check = []
        for c in usable:
            if c[3] > last and c[0] not in taken and len(check) < CHECK: check.append(c); taken.add(c[0])
        write("Data/fresh/general_dialogue.tsv", check)
        print(f"Data/fresh/general_dialogue.tsv: {len(check)} sentences")
