"""Cleaning of the training text for the language model (Training/prepare_data.py).

The input method types words, English words and short sentences, so the model is trained on
text people type, with nothing that identifies anyone in it:

- redact(): the same rules as the input method's own learning records (Engine/Redaction.swift):
  e-mail addresses, web addresses, six or more digits in a row (phone, ID card with its final X,
  bank card, codes) and card or phone numbers written in groups become "＊". Dates, times and
  numbers merely next to each other stay.
- clean(): redacts, then drops lines that are not something one types (mostly digits, symbols or
  redacted marks: tables, separators, bare addresses) and returns None for documents made mostly
  of numbers (price lists, statistics).

Only training documents are cleaned; the held-out documents stay as they were, so the evaluation
sets made from them keep their meaning.
"""
import re

MARK = "＊"

PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?i)\b(?:https?://|www\.)[^\s，。、；：！？）》」』]+"),
    re.compile(r"(?<!\d)\d{4}(?:[ \-]\d{4}){2,3}(?:[ \-]\d{1,4})?(?!\d)"),
    re.compile(r"(?<!\d)1\d{2}[ \-]\d{4}[ \-]\d{4}(?!\d)"),
    re.compile(r"\d{6,}[Xx]?"),
]
HAN_OR_LETTER = re.compile(r"[一-鿿A-Za-z]")
DIGIT = re.compile(r"\d")
# What a typed line is not made of: digits, redacted marks, box drawing and the symbols of tables,
# timestamps and code. Punctuation people type in chat, Chinese or ASCII (！ ？ 。 … . ! ? ~ , ;),
# does not count ("管住手！！！！", "唉......").
NOT_TYPED = re.compile(r"[0-9＊\u2500-\u257F#$%&*+/:<=>@\[\\\]^_`{|}\-]")

# A line shorter than this is kept whatever it holds (a reply such as "1" or "+1" is typed text).
MIN_JUDGED_LINE = 8
# A longer line is dropped when more than MAX_NOT_TYPED of it (spaces aside) is digits and symbols
# and less than MIN_TEXT_SHARE is Chinese characters or letters: tables, separators, timestamps.
MAX_NOT_TYPED = 0.6
MIN_TEXT_SHARE = 0.3
# A document of at least MIN_JUDGED_DOCUMENT characters with more than MAX_DIGIT_SHARE digits is
# dropped (lottery tips, statistics). Short ones stay: a chat with a few numbers is typed text.
MIN_JUDGED_DOCUMENT = 200
MAX_DIGIT_SHARE = 0.2


def redact(text):
    for pattern in PATTERNS:
        text = pattern.sub(MARK, text)
    return text


def typed_line(line):
    compact = line.replace(" ", "")
    if len(compact) < MIN_JUDGED_LINE:
        return True
    symbols = len(NOT_TYPED.findall(compact))
    text = len(HAN_OR_LETTER.findall(compact))
    return not (symbols > MAX_NOT_TYPED * len(compact) and text < MIN_TEXT_SHARE * len(compact))


def clean(text):
    """The document as it is trained on, or None when nothing worth typing is left."""
    text = redact(text)
    lines = [line for line in text.split("\n") if typed_line(line)]
    text = "\n".join(lines)
    compact = text.replace(" ", "").replace("\n", "")
    if not compact:
        return None
    if len(compact) >= MIN_JUDGED_DOCUMENT and len(DIGIT.findall(compact)) > MAX_DIGIT_SHARE * len(compact):
        return None
    return text
