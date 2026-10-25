"""Cleaning of the training text for the language model (Training/prepare_data.py).

The input method types words, English words and short sentences, so the model is trained on
typed text, with nothing identifying in it:

- redact(): the rules of Resources/redaction.json, the same file the input method's learning
  records are redacted by (Engine/Redaction.swift).
  Become "＊": e-mail addresses; web addresses, also right after Chinese text and bare domains
  (yys.163.com); six or more digits in a row (phone, ID card with its final X, bank card, codes);
  card or phone numbers written in groups; the account after 微信/QQ and the like; the name after
  姓名/联系人/收件人; the name after "@" (@＊); the building, unit and room of an address
  (11楼1211, 3号楼2单元401); licence plates. Dates, times, prices, street names and numbers merely
  next to each other stay.
- clean(): HTML entities read as their characters; HTML tags and forum markup ([size=14]) and the
  reply headers of forums (作者：… 回复日期：…, @someone 3楼 2015-09-09 08:18:00, 来自…客户端) are
  removed; then redacts, drops lines that are not something one types (mostly digits, symbols or
  redacted marks: tables, separators, timestamps) and returns None for documents made mostly of
  numbers (price lists, statistics).

Held-out documents are cleaned the same way: validation and the test sets made from
them measure the text the model is trained for. Training/test_cleaning.py checks every rule.
"""
import html
import json
import re
from pathlib import Path

# The redaction rules live in Resources/redaction.json, shared with the input method's learning
# records (Engine/Redaction.swift reads the same file): one list, so the two never drift apart.
_RULES = json.loads((Path(__file__).resolve().parent.parent / "Resources" / "redaction.json").read_text(encoding="utf-8"))
MARK = _RULES["mark"]


def _expand(pattern):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: _RULES["vars"][m.group(1)], pattern)


# (pattern, replacement) in the file's order; $1 there is \1 here.
PATTERNS = [
    (re.compile(_expand(rule["pattern"])),
     (lambda m: MARK if _random_run(m.group()) else m.group()) if rule.get("generated") else rule["replace"].replace("$1", "\\1"))
    for rule in _RULES["rules"]
]

_DATE = r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}日?|\d{4}年\d{1,2}月\d{1,2}日)"
_DATE_TIME = _DATE + r"(?:[ \t]*\d{1,2}:\d{2}(?::\d{2})?)?"
# Down to the second: written by a system, never typed (2017-04-18 23:01:00, 2018-10-01 00:49:18.422).
_TIMESTAMP = _DATE + r"[ \t]*\d{1,2}:\d{2}:\d{2}(?:\.\d+)?"
# Forum reply headers and client signatures, wherever they sit in a line.
STRUCTURE = [
    # Signatures, notices and comment headers with their names, before the timestamps go.
    re.compile(r"【?本[贴帖]由[^\n于]{1,30}于[ \t]*" + _DATE_TIME + r"[ \t]*在〖[^〗\n]{0,30}〗发表】?\.?"),
    re.compile(r"(?m)^.*(?:已被[^\n，,]{0,30}封杀|解封时间为).*$"),
    re.compile(r"[^\s:：]{1,30}[:：][ \t]*(?:举报[ \t]*)?" + _DATE_TIME + r"[ \t]*评论"),
    re.compile(r"(?:发布|更新|发表|上传)时间[:：][ \t]*" + _DATE_TIME + r"(?:[ \t]*(?:编辑|作者|来源)[:：][ \t]*[^\s，。]{1,20})?"),
    re.compile(r"(?:发布者|发帖人|楼主)[:：][ \t]*\S{1,30}(?=[ \t]+(?:发表日期|发表时间|日期|时间)[:：])"),
    re.compile(r"作者[:：][ \t]*\S{1,30}[ \t]+来自[:：][ \t]*\S{0,16}客户端[ \t]+时间[:：][ \t]*" + _DATE_TIME),
    re.compile(r"作者[:：][ \t]*\S{1,30}[ \t]+(?:回复日期|发表日期|日期|时间)[:：][ \t]*" + _DATE_TIME),
    re.compile(r"(?:回复日期|发表日期|发表时间|回复时间)[:：][ \t]*" + _DATE_TIME),
    re.compile(r"(?<![一-鿿])时间[:：][ \t]*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}[ \t]+\d{1,2}:\d{2}(?::\d{2})?"),
    re.compile(r"回复第\d+楼[ \t]*[(（][^()（）\n]{0,60}[)）]"),
    re.compile(r"本帖最后由[ \t]*\S{1,30}[ \t]*于[ \t]*" + _DATE_TIME + r"[ \t]*编辑"),
    re.compile(r"@\S{1,30}(?:[ \t]+\d+楼)?[ \t]+" + _DATE_TIME),
    re.compile(r"(?:\S{1,30}[ \t]+)?\d+楼[ \t]+" + _DATE_TIME),
    re.compile(r"(?:来自[:：][ \t]*\S{0,16}客户端|\[来自[^\]\n]{0,30}\])"),
    # Timestamps left over, after the headers that are recognised by them.
    re.compile(r"[(（]?" + _TIMESTAMP + r"[)）]?"),
    re.compile(_DATE + r"[ \t]*\d{1,2}:\d{2}[ \t]*(?:上传|发布|发表|更新|编辑|评论|回复|来自[^\s，。]{0,8})"),
    re.compile(r"(?i)\[/?(?:size|color|b|i|u|s|url|img|quote|font|align|attach|attachimg|backcolor|table|tr|td|list|hr|media|flash|code|email|indent|float|hide|free)(?:=[^\]\n]{0,80})?\]"),
    re.compile(r"(?i)</?(?:p|div|span|a|img|br|hr|table|tbody|thead|tr|td|th|ul|ol|li|strong|em|b|i|u|font|h[1-6]|iframe|script|style|section|article|header|footer|figure|figcaption|blockquote|center|sup|sub|small|video|source|embed|meta|link)(?:\s[^<>\n]{0,300})?/?>"),
    # Tags the corpus cut short, the text going on right after (after whole tags): <img data-rawwidth正文…
    re.compile(r"(?i)<(?:img|a|figure|noscript|iframe|video|span|div|p)(?:[ \t]+(?:data-[a-z\-]+|class|href|src|style|id)(?:=\"[^\"\n]{0,200}\"?)?)+"),
    re.compile(r"(?im)</(?:p|div|span|a)$"),
    re.compile(r"(?im)\[/?(?:size|color|font)(?:=[^\]\n]{0,20})?$"),
]

SPACES = re.compile(r"[ \t]{2,}")
HAN_OR_LETTER = re.compile(r"[一-鿿A-Za-z]")
DIGIT = re.compile(r"\d")
# What a typed line is not made of: digits, redacted marks, box drawing and the symbols of tables,
# timestamps and code. Punctuation typed in chat, Chinese or ASCII (！ ？ 。 … . ! ? ~ , ;),
# does not count ("管住手！！！！", "唉......").
NOT_TYPED = re.compile(r"[0-9＊─-╿#$%&*+/:<=>@\[\\\]^_`{|}\-]")

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


def _random_run(run):
    """Whether a run of letters and digits looks generated rather than written: a quarter of it
    digits, or 40 or more long with a tenth of it digits."""
    share = sum(c.isdigit() for c in run) / len(run)
    return share >= 0.25 or (len(run) >= 40 and share >= 0.1)


def redact(text):
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def typed_line(line):
    compact = line.replace(" ", "")
    if len(compact) < MIN_JUDGED_LINE:
        return True
    symbols = len(NOT_TYPED.findall(compact))
    text = len(HAN_OR_LETTER.findall(compact))
    return not (symbols > MAX_NOT_TYPED * len(compact) and text < MIN_TEXT_SHARE * len(compact))


def strip_structure(text):
    """HTML entities as their characters; markup and forum reply headers removed."""
    text = html.unescape(text).replace(" ", " ")
    for pattern in STRUCTURE:
        text = pattern.sub("", text)
    return text


def clean(text):
    """The document as it is trained on, or None when nothing worth typing is left."""
    text = redact(strip_structure(text))
    lines = [SPACES.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line and typed_line(line)]
    text = "\n".join(lines)
    compact = text.replace(" ", "").replace("\n", "")
    if not compact:
        return None
    if len(compact) >= MIN_JUDGED_DOCUMENT and len(DIGIT.findall(compact)) > MAX_DIGIT_SHARE * len(compact):
        return None
    return text
