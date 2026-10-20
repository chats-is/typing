import Foundation

/// What is taken out of the learning records before anything is written: e-mail addresses, web
/// addresses, six or more digits in a row and card or phone numbers written in groups become one
/// "＊"; a commit typed right after a prompt for a password or a code is not recorded at all.
/// Shorter numbers (years, prices, versions) stay. The original text is never stored.
public enum Redaction {
    /// The mark that stands for what was taken out.
    public static let mark = "＊"

    private static let patterns: [NSRegularExpression] = [
        // E-mail addresses.
        #"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"#,
        // Web addresses.
        #"(?i)\b(?:https?://|www\.)[^\s，。、；：！？）》」』]+"#,
        // A bank card number written in groups of four, and a phone number written 3-4-4.
        #"(?<!\d)\d{4}(?:[ \-]\d{4}){2,3}(?:[ \-]\d{1,4})?(?!\d)"#,
        #"(?<!\d)1\d{2}[ \-]\d{4}[ \-]\d{4}(?!\d)"#,
        // Six or more digits in a row (phone, ID card with its final X, bank card, codes). Dates and
        // times such as 2016-07-20 11:30, or numbers merely next to each other, are not one run.
        #"\d{6,}[Xx]?"#,
    ].map { try! NSRegularExpression(pattern: $0) }

    /// Words that, just before the cursor, mean a secret is being typed (English ones as whole
    /// words, so pinyin or spinning do not count).
    private static let secretPrompt = try! NSRegularExpression(pattern: #"(?i)密码|口令|验证码|校验码|动态码|\b(?:password|passcode|pin|otp)\b"#)

    /// The text with every sensitive run replaced by `mark`.
    public static func redact(_ text: String) -> String {
        var result = text
        for pattern in patterns {
            let range = NSRange(result.startIndex..., in: result)
            result = pattern.stringByReplacingMatches(in: result, range: range, withTemplate: mark)
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
