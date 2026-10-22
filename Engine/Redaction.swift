import Foundation

/// What is taken out of the learning records before anything is written, by the same rules as the
/// training text (Training/cleaning.py, redact()): e-mail addresses; web addresses, also right
/// after Chinese text, and bare domains; six or more digits in a row; card or phone numbers written
/// in groups; the account after 微信/QQ; the name after 姓名/联系人/收件人; the person after "@";
/// the building, unit and room of an address; licence plates. Each becomes "＊"; a commit typed
/// right after a prompt for a password or a code is not recorded at all. Shorter numbers (years,
/// prices, versions) and street names stay. The original text is never stored.
public enum Redaction {
    /// The mark that stands for what was taken out.
    public static let mark = "＊"

    private static let urlEnd = #"[^\s，。、；：！？）》」』\"'<>（(【\[一-鿿,]*"#
    private static let tld = "(?:monster|online|click|cloud|store|asia|blog|bond|buzz|club|cyou|guru|info|link|live|loan|mobi|name|shop|site|tech|wang|work|app|bid|biz|cfd|com|dev|edu|fun|gov|icu|ink|kim|ltd|net|org|pro|ren|sbs|top|vip|win|xin|xyz|ac|ad|ae|af|ag|al|ao|aq|ar|au|aw|ax|az|ba|bb|bd|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bw|bz|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cu|cv|cw|cx|cy|cz|de|dj|dk|dm|dz|ec|ee|eg|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|ie|il|im|io|iq|ir|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|me|mg|mh|mm|mn|mp|mq|mr|ms|mt|mu|mv|mw|mx|mz|na|nc|ne|nf|ng|ni|nl|np|nr|nu|nz|om|pa|pe|pf|pg|pk|pn|pr|pt|pw|qa|re|ro|ru|rw|sa|sb|sc|sd|se|sg|si|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"
    /// Suffixes that are also file extensions or words in code and text (index.md, req.params.id):
    /// a domain only with a path or a query after it (now.to/x5c).
    private static let tldPath = "(?:company|network|science|center|design|review|travel|email|games|group|media|party|press|space|today|tools|trade|world|best|chat|city|cool|date|game|host|life|love|news|page|plus|rest|show|team|wiki|zone|art|bar|fit|int|men|mil|one|pub|red|run|tel|ai|am|as|at|be|by|ca|cs|do|go|id|in|is|it|lb|md|mk|ml|mo|my|no|ph|pl|pm|ps|py|rs|sh|to)"
    private static let numeral = "[0-9一二三四五六七八九十]+"

    /// A replacement by template, or the letters-and-digits runs that look generated.
    private enum Step { case replace(NSRegularExpression, String), generatedRuns(NSRegularExpression) }

    /// In this order: e-mail before "@someone", accounts before plain digits.
    private static let steps: [Step] = {
        func r(_ pattern: String, _ template: String = mark) -> Step { .replace(try! NSRegularExpression(pattern: pattern), template) }
        return [
            r(#"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"#),
            // Addresses whose "://" the corpus lost: httpwww.bilibili.com…, httpmp.weixin.qq.com…
            r(#"(?i)(?<![A-Za-z])https?:?/{0,2}(?=www\.|(?:[a-z0-9\-]+\.)+"# + tld + ")" + urlEnd),
            // From wherever http:// or www. starts, no \b: Chinese characters count as word
            // characters, so 网址http://… had none, and glued text (7http://…) has none either.
            r(#"(?i)(?:https?://|www\.)"# + urlEnd),
            // Bare domains in small letters, as they are written (user.Info in code is not one).
            r(#"(?<![A-Za-z0-9@.])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"# + tld + #"(?:\.[a-z]{2})?(?::\d+)?(?![A-Za-z0-9])(?:/"# + urlEnd + ")?"),
            r(#"(?<![A-Za-z0-9@.])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"# + tldPath + #"(?:\.[a-z]{2})?(?::\d+)?(?=[/?])"# + urlEnd),
            // Within one line: [ \t], never \s, which would reach into the next line.
            r(#"((?:微信|[vV]信|威信|薇信|扣扣|抖音|快手|支付宝|淘宝|旺旺|钉钉|小红书)[ \t]*(?:号码|号|群)?[ \t]*[:：]?[ \t]*)[A-Za-z0-9][\-_A-Za-z0-9]{4,19}(?![\-_A-Za-z0-9])"#, "$1" + mark),
            // Latin names only before 号 or a colon, or before a space and an account with a digit
            // (wxWidgets and "go on QQ immediately" are not accounts).
            r(#"(?i)((?<![A-Za-z\-])(?:weixin|vx|wx|qq|line|skype|telegram|tg)(?:[ \t]*(?:号码|号|群|id)[ \t]*[:：]?[ \t]*|[ \t]*[:：][ \t]*))[A-Za-z0-9][\-_.A-Za-z0-9]{4,29}(?![\-_A-Za-z0-9])"#, "$1" + mark),
            r(#"(?i)((?<![A-Za-z])(?:weixin|vx|wx|qq)[ \t]+)(?=[\-_A-Za-z]*\d)[A-Za-z0-9][\-_A-Za-z0-9]{4,19}(?![\-_A-Za-z0-9])"#, "$1" + mark),
            r(#"((?:真实姓名|姓名|联系人|收件人|收货人|户名)(?:[ \t]*[:：][ \t]*|[ \t]+))[一-鿿·]{2,5}"#, "$1" + mark),
            r(#"@[^\s@，。、；：！？,;:!?（）()【】\[\]《》<>“”"'…]{1,20}"#, "@" + mark),
            r(#"\d+(?:号楼|栋|幢|座)[ \t]*(?:"# + numeral + #"单元)?[ \t]*\d{2,4}(?:室|号)?"#),
            r(#"(?:第?"# + numeral + #"单元)[ \t]*\d{3,4}(?:室|号)?(?!\d)"#),
            r(#"(?<!\d)\d+楼\d{3,4}(?:室|号)?(?![\d元块万千百个人次年米])"#),
            r(#"(?<!\d)\d{3,4}(?=室)"#),
            r(#"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][·•]?[A-HJ-NP-Z0-9]{5,6}(?![A-Za-z0-9])"#),
            // Letters and digits mixed, 20 or more, with many digits: link parameters, keys, check
            // codes, file names. Names with a digit or two stay (CanvasRenderingContext2D).
            .generatedRuns(try! NSRegularExpression(pattern: #"(?<![A-Za-z0-9])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{20,}(?![A-Za-z0-9])"#)),
            // A bank card number written in groups of four, and a phone number written 3-4-4.
            r(#"(?<!\d)\d{4}(?:[ \-]\d{4}){2,3}(?:[ \-]\d{1,4})?(?!\d)"#),
            r(#"(?<!\d)1\d{2}[ \-]\d{4}[ \-]\d{4}(?!\d)"#),
            // Six or more digits in a row (phone, ID card with its final X, bank card, codes). Dates
            // and times such as 2016-07-20 11:30, or numbers merely next to each other, are not one run.
            r(#"\d{6,}[Xx]?"#),
        ]
    }()

    /// Whether a run of letters and digits looks generated rather than written: a quarter of it
    /// digits, or 40 or more long with a tenth of it digits.
    private static func generated(_ run: Substring) -> Bool {
        let share = Double(run.filter(\.isNumber).count) / Double(run.count)
        return share >= 0.25 || (run.count >= 40 && share >= 0.1)
    }

    /// Words that, just before the cursor, mean a secret is being typed (English ones as whole
    /// words, so pinyin or spinning do not count).
    private static let secretPrompt = try! NSRegularExpression(pattern: #"(?i)密码|口令|验证码|校验码|动态码|\b(?:password|passcode|pin|otp)\b"#)

    /// The text with every sensitive run replaced by `mark`.
    public static func redact(_ text: String) -> String {
        var result = text
        for step in steps {
            let range = NSRange(result.startIndex..., in: result)
            switch step {
            case .replace(let pattern, let template):
                result = pattern.stringByReplacingMatches(in: result, range: range, withTemplate: template)
            case .generatedRuns(let pattern):
                for match in pattern.matches(in: result, range: range).reversed() {
                    guard let span = Range(match.range, in: result), generated(result[span]) else { continue }
                    result.replaceSubrange(span, with: mark)
                }
            }
        }
        return result
    }

    /// Whether the text in front of the cursor asks for a password or a code: then what is typed
    /// next is not recorded.
    public static func asksForSecret(_ context: String) -> Bool {
        let tail = String(context.suffix(12))
        return secretPrompt.firstMatch(in: tail, range: NSRange(tail.startIndex..., in: tail)) != nil
    }
}
