"""The text sources the language model is trained on, each yielding plain-text documents.

Everything is read from Data/corpus2 and Data/corpus3 (see Data/lexicons/sources.tsv for where
the files came from). A document is one dialogue, article, forum thread or stretch of a novel.
"""
import glob
import json
import os
import re
import zipfile
import gzip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "Data", "corpus2")

ASCII_WORD = re.compile(r"[0-9A-Za-z]")
NOVEL_CHUNK = 2000


def join_segmented(utterance):
    """LCCC text is word-segmented with spaces; drop them except between two ASCII words."""
    parts = utterance.split(" ")
    out = []
    for part in parts:
        if out and part and ASCII_WORD.match(out[-1][-1:]) and ASCII_WORD.match(part[:1]):
            out.append(" ")
        out.append(part)
    return "".join(out)


def dialogue():
    """Short everyday conversations (LCCC large, which contains the base set)."""
    with gzip.open(os.path.join(CORPUS, "lccc_large.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            try:
                turns = json.loads(line)
            except ValueError:
                continue
            yield "\n".join(join_segmented(turn) for turn in turns)


def mixed():
    """News, encyclopedia, reviews and community Q&A, one document per line (CLUECorpusSmall slice)."""
    with open(os.path.join(CORPUS, "clue_part.txt"), "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break  # the slice ends in the middle of a line
            yield raw.decode("utf-8", "ignore").strip()


def news():
    """Full news articles (THUCNews)."""
    with open(os.path.join(CORPUS, "THUCNews.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
            except ValueError:
                continue
            body = (article.get("text") or "").replace("　　", "\n")
            yield (article.get("title") or "") + "\n" + body


# Lines the forum software added around the posts: reply buttons, floor numbers,
# mentions, separators and table rows.
FORUM_NOISE = re.compile(r"^(回复|举报|楼主|#\d+楼|\d+楼|@\S+|[-=_—]{4,}.*|\|.*\||来自.{0,12}客户端.*)$")


def clean_post(post):
    return "\n".join(line for line in (l.strip() for l in post.split("\n")) if line and not FORUM_NOISE.match(line))


def forum():
    """Forum threads: the topic followed by its replies (Tianya)."""
    for path in sorted(glob.glob(os.path.join(CORPUS, "tianya", "*.zip"))):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                with archive.open(name) as f:
                    for raw in f:
                        try:
                            thread = json.loads(raw)
                        except ValueError:
                            continue
                        replies = [clean_post(reply.get("回复") or "") for reply in thread.get("回复") or []]
                        yield "\n".join([thread.get("主题") or ""] + [reply for reply in replies if reply])


def novels():
    """Classic novels, cut into stretches of about NOVEL_CHUNK characters at paragraph ends."""
    for path in sorted(glob.glob(os.path.join(CORPUS, "novels", "*.txt"))):
        if os.path.basename(path) == "list.txt":
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            chunk, size = [], 0
            for paragraph in f:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                chunk.append(paragraph)
                size += len(paragraph)
                if size >= NOVEL_CHUNK:
                    yield "\n".join(chunk)
                    chunk, size = [], 0
            if chunk:
                yield "\n".join(chunk)


def aligned_units(path):
    """Lines of "pinyin_汉字 pinyin_汉字 ..." (the OpenIME corpora): the Chinese side of each."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            characters = [pair.split("_", 1)[1] for pair in line.split() if "_" in pair]
            if characters:
                yield "".join(characters)


def touchpal():
    """Real input-method input, one input unit per document (TouchPal corpus)."""
    return aligned_units(os.path.join(CORPUS, "openime", "OpenIME", "dataset", "TP", "touchpal.segWithCYHZ.50w.adddict.ali"))


def peoples_daily():
    """People's Daily sentences (OpenIME's PD corpus)."""
    return aligned_units(os.path.join(CORPUS, "openime", "OpenIME", "dataset", "PD", "r9397.miu.segWithCYHZ.5kk.adddict.ali"))


CORPUS3 = os.path.join(ROOT, "Data", "corpus3")
HAN = re.compile(r"[一-鿿]")
# Gambling, adult and SEO spam that web crawls are full of.
WEB_SPAM = re.compile(r"博彩|娱乐城|百家乐|老虎机|六合彩|时时彩|赌场|真人视讯|葡京|威尼斯人|彩票网|代孕|成人视频|av女优|裸聊|一夜情|发票代开|代开发票|私服|外挂|seo优化|快速排名|蜘蛛池|激情|色情|无码")


def web():
    """Recent general web text (FineWeb-2, cmn_Hani, ODC-By): pages crawled from 2015 on, with
    the English words mixed into Chinese and their original case. Pages that are mostly
    not Chinese, very short, or spam are skipped."""
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(os.path.join(CORPUS3, "fineweb2", "*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["text", "date"]):
            for text, date in zip(batch.column("text").to_pylist(), batch.column("date").to_pylist()):
                if not date or date[:4] < "2015":
                    continue
                han = len(HAN.findall(text))
                if han < 100 or han < 0.4 * len(text.replace(" ", "").replace("\n", "")):
                    continue
                if WEB_SPAM.search(text.lower()):
                    continue
                yield text


FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
CODE_BLOCK = re.compile(r"^```.*?^```", re.S | re.M)
SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.S)
MACRO = re.compile(r"\{\{\s*\w+\(\s*\"([^\"]*)\"[^}]*\}\}")
OTHER_MACRO = re.compile(r"\{\{[^}]*\}\}")
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
TAG = re.compile(r"<[^>]+>")
MARKUP = re.compile(r"^\s*(#+|[-*+]|\d+\.|>)\s+|[*_`|]+", re.M)


def techdocs():
    """Chinese technical documentation, dense with English terms in their usual case: MDN's
    zh-CN pages (CC BY-SA 2.5) and the Kubernetes website's zh-cn pages (CC BY 4.0). Markdown,
    code blocks and links are stripped, and only lines with Chinese in them are kept."""
    paths = glob.glob(os.path.join(CORPUS3, "mdn", "repo", "files", "zh-cn", "**", "*.md"), recursive=True)
    paths += glob.glob(os.path.join(CORPUS3, "k8s", "repo", "content", "zh-cn", "**", "*.md"), recursive=True)
    for path in sorted(p for p in paths if os.path.isfile(p)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = CODE_BLOCK.sub("", FRONT_MATTER.sub("", text))
        text = OTHER_MACRO.sub("", MACRO.sub(r"\1", SHORTCODE.sub("", text)))
        text = MARKUP.sub("", TAG.sub("", LINK.sub(r"\1", text)))
        lines = [line.strip() for line in text.split("\n") if HAN.search(line)]
        if lines:
            yield "\n".join(lines)


def tatoeba():
    """Short everyday English sentences (Tatoeba, CC BY 2.0 FR), 30 to a document."""
    import bz2

    chunk = []
    with bz2.open(os.path.join(CORPUS3, "tatoeba", "eng_sentences.tsv.bz2"), "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                chunk.append(parts[2])
            if len(chunk) == 30:
                yield "\n".join(chunk)
                chunk = []
    if chunk:
        yield "\n".join(chunk)


# name -> (documents, share of each training batch drawn from it)
SOURCES = {
    "dialogue": (dialogue, 0.28),
    "mixed": (mixed, 0.18),
    "news": (news, 0.09),
    "forum": (forum, 0.12),
    "novels": (novels, 0.04),
    "touchpal": (touchpal, 0.05),
    "peoples_daily": (peoples_daily, 0.02),
    "web": (web, 0.14),
    "techdocs": (techdocs, 0.02),
    "tatoeba": (tatoeba, 0.02),
    "words": (None, 0.03),  # lexicon entries, written by prepare_data.py
    "english": (None, 0.01),  # common English words, written by prepare_data.py
}
