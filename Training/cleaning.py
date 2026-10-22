"""Cleaning of the training text for the language model (Training/prepare_data.py).

The input method types words, English words and short sentences, so the model is trained on
text people type, with nothing that identifies anyone in it:

- redact(): the same rules as the input method's own learning records (Engine/Redaction.swift).
  Become "＊": e-mail addresses; web addresses, also right after Chinese text and bare domains
  (yys.163.com); six or more digits in a row (phone, ID card with its final X, bank card, codes);
  card or phone numbers written in groups; the account after 微信/QQ and the like; the name after
  姓名/联系人/收件人; the person after "@" (@＊); the building, unit and room of an address
  (11楼1211, 3号楼2单元401); licence plates. Dates, times, prices, street names and numbers merely
  next to each other stay.
- clean(): HTML entities read as their characters; HTML tags and forum markup ([size=14]) and the
  reply headers of forums (作者：… 回复日期：…, @someone 3楼 2015-09-09 08:18:00, 来自…客户端) are
  removed; then redacts, drops lines that are not something one types (mostly digits, symbols or
  redacted marks: tables, separators, timestamps) and returns None for documents made mostly of
  numbers (price lists, statistics).

Held-out documents are cleaned the same way (2026-10-22): validation and the test sets made from
them measure the text the model is trained for. Training/test_cleaning.py checks every rule.
"""
import html
import re

MARK = "＊"

# Web addresses and domains end before Chinese punctuation, quotes and brackets. Domain suffixes:
# every country code and the usual generic ones (.com, .tk, .icu, .guru…).
_URL_END = r"[^\s，。、；：！？）》」』\"'<>（(【\[一-鿿,]*"
_TLD = r"(?:monster|online|click|cloud|store|asia|blog|bond|buzz|club|cyou|guru|info|link|live|loan|mobi|name|shop|site|tech|wang|work|app|bid|biz|cfd|com|dev|edu|fun|gov|icu|ink|kim|ltd|net|org|pro|ren|sbs|top|vip|win|xin|xyz|ac|ad|ae|af|ag|al|ao|aq|ar|au|aw|ax|az|ba|bb|bd|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bw|bz|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cu|cv|cw|cx|cy|cz|de|dj|dk|dm|dz|ec|ee|eg|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|ie|il|im|io|iq|ir|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|me|mg|mh|mm|mn|mp|mq|mr|ms|mt|mu|mv|mw|mx|mz|na|nc|ne|nf|ng|ni|nl|np|nr|nu|nz|om|pa|pe|pf|pg|pk|pn|pr|pt|pw|qa|re|ro|ru|rw|sa|sb|sc|sd|se|sg|si|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"
# Suffixes that are also file extensions or words in code and text (index.md, req.params.id,
# memory.oom.group): a domain only with a path or a query after it (now.to/x5c).
_TLD_PATH = r"(?:company|network|science|center|design|review|travel|email|games|group|media|party|press|space|today|tools|trade|world|best|chat|city|cool|date|game|host|life|love|news|page|plus|rest|show|team|wiki|zone|art|bar|fit|int|men|mil|one|pub|red|run|tel|ai|am|as|at|be|by|ca|cs|do|go|id|in|is|it|lb|md|mk|ml|mo|my|no|ph|pl|pm|ps|py|rs|sh|to)"
_PROVINCE = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_NUMERAL = "[0-9一二三四五六七八九十]+"

# (pattern, replacement), in this order: e-mail before "@someone", accounts before plain digits.
PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), MARK),
    # Addresses whose "://" the corpus lost: httpwww.bilibili.com…, httpmp.weixin.qq.com…
    (re.compile(r"(?i)(?<![A-Za-z])https?:?/{0,2}(?=www\.|(?:[a-z0-9\-]+\.)+" + _TLD + r")" + _URL_END), MARK),
    # From wherever http:// or www. starts, no \b: Chinese characters count as word characters, so
    # 网址http://… had none, and glued text (7http://…, increasewww.…) has none either.
    (re.compile(r"(?i)(?:https?://|www\.)" + _URL_END), MARK),
    # Bare domains in small letters, as they are written (user.Info in code is not one).
    (re.compile(r"(?<![A-Za-z0-9@.])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+" + _TLD + r"(?:\.[a-z]{2})?(?::\d+)?(?![A-Za-z0-9])(?:/" + _URL_END + ")?"), MARK),
    (re.compile(r"(?<![A-Za-z0-9@.])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+" + _TLD_PATH + r"(?:\.[a-z]{2})?(?::\d+)?(?=[/?])" + _URL_END), MARK),
    # Within one line: [ \t], never \s, which would reach into the next line.
    (re.compile(r"((?:微信|[vV]信|威信|薇信|扣扣|抖音|快手|支付宝|淘宝|旺旺|钉钉|小红书)[ \t]*(?:号码|号|群)?[ \t]*[:：]?[ \t]*)[A-Za-z0-9][\-_A-Za-z0-9]{4,19}(?![\-_A-Za-z0-9])"), r"\1" + MARK),
    # Latin names only before 号 or a colon, or before a space and an account with a digit
    # (wxWidgets and "go on QQ immediately" are not accounts).
    (re.compile(r"(?i)((?<![A-Za-z\-])(?:weixin|vx|wx|qq|line|skype|telegram|tg)(?:[ \t]*(?:号码|号|群|id)[ \t]*[:：]?[ \t]*|[ \t]*[:：][ \t]*))[A-Za-z0-9][\-_.A-Za-z0-9]{4,29}(?![\-_A-Za-z0-9])"), r"\1" + MARK),
    (re.compile(r"(?i)((?<![A-Za-z])(?:weixin|vx|wx|qq)[ \t]+)(?=[\-_A-Za-z]*\d)[A-Za-z0-9][\-_A-Za-z0-9]{4,19}(?![\-_A-Za-z0-9])"), r"\1" + MARK),
    (re.compile(r"((?:真实姓名|姓名|联系人|收件人|收货人|户名)(?:[ \t]*[:：][ \t]*|[ \t]+))[一-鿿·]{2,5}"), r"\1" + MARK),
    (re.compile(r"@[^\s@，。、；：！？,;:!?（）()【】\[\]《》<>“”\"'…]{1,20}"), "@" + MARK),
    (re.compile(r"\d+(?:号楼|栋|幢|座)[ \t]*(?:" + _NUMERAL + r"单元)?[ \t]*\d{2,4}(?:室|号)?"), MARK),
    (re.compile(r"(?:第?" + _NUMERAL + r"单元)[ \t]*\d{3,4}(?:室|号)?(?!\d)"), MARK),
    (re.compile(r"(?<!\d)\d+楼\d{3,4}(?:室|号)?(?![\d元块万千百个人次年米])"), MARK),
    (re.compile(r"(?<!\d)\d{3,4}(?=室)"), MARK),
    (re.compile(r"[" + _PROVINCE + r"][A-Z][·•]?[A-HJ-NP-Z0-9]{5,6}(?![A-Za-z0-9])"), MARK),
    # Letters and digits mixed, 20 or more, with many digits: link parameters, keys, check codes,
    # file names; nobody types them. Names with a digit or two stay (CanvasRenderingContext2D).
    (re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{20,}(?![A-Za-z0-9])"), lambda m: MARK if _random_run(m.group()) else m.group()),
    (re.compile(r"(?<!\d)\d{4}(?:[ \-]\d{4}){2,3}(?:[ \-]\d{1,4})?(?!\d)"), MARK),
    (re.compile(r"(?<!\d)1\d{2}[ \-]\d{4}[ \-]\d{4}(?!\d)"), MARK),
    (re.compile(r"\d{6,}[Xx]?"), MARK),
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
# timestamps and code. Punctuation people type in chat, Chinese or ASCII (！ ？ 。 … . ! ? ~ , ;),
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
