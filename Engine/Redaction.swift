import Foundation

/// What is taken out of the learning records before anything is written: the rules of
/// Resources/redaction.json, the same file the training text is redacted by (Training/cleaning.py),
/// so the two never drift apart. E-mail and web addresses, long digit runs, accounts, names after
/// 姓名/联系人, the name after "@", rooms and plates each become "＊"; a commit typed right after
/// a prompt for a password or a code is not recorded at all. Shorter numbers (years, prices,
/// versions) and street names stay. The original text is never stored.
public enum Redaction {
    /// The rules file as written; see its "about".
    private struct File: Decodable {
        struct Rule: Decodable { let pattern: String; let replace: String?; let generated: Bool? }
        let mark: String
        let vars: [String: String]
        let rules: [Rule]
    }

    /// A replacement by template, or the letters-and-digits runs that look generated.
    private enum Step { case replace(NSRegularExpression, String), generatedRuns(NSRegularExpression) }

    /// Resources/redaction.json: in the app's bundle, or in the repository when a tool runs from
    /// its root. Without it nothing could be redacted, so there is no carrying on (build.sh copies it).
    private static let file: File = {
        let url = Bundle.main.url(forResource: "redaction", withExtension: "json") ?? URL(fileURLWithPath: "Resources/redaction.json")
        guard let data = try? Data(contentsOf: url), let file = try? JSONDecoder().decode(File.self, from: data) else {
            fatalError("redaction.json missing or unreadable at \(url.path)")
        }
        return file
    }()

    /// The mark that stands for what was taken out.
    public static var mark: String { file.mark }

    /// The rules in the file's order, {{NAME}} replaced by its variable; $1 in a template is the first group.
    private static let steps: [Step] = file.rules.map { rule in
        var pattern = rule.pattern
        for (name, value) in file.vars { pattern = pattern.replacingOccurrences(of: "{{\(name)}}", with: value) }
        let regex = try! NSRegularExpression(pattern: pattern)
        return rule.generated == true ? .generatedRuns(regex) : .replace(regex, rule.replace ?? file.mark)
    }

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
