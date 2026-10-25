import Foundation

/// Results typed by a short form instead of character by character.
/// - v and a number: its Chinese forms (v1234.5 → 一千二百三十四点五, 壹仟贰佰叁拾肆点伍,
///   壹仟贰佰叁拾肆元伍角, 1,234.5).
/// - v and a sum: its result (v12*3+5 → 41, 12*3+5=41).
/// - vu and a code point: the character (vu4e00 → 一).
/// - The names 日期 时间 星期 农历 are followed by today's date, the time, the weekday, the lunar date.
/// No syllable starts with v, so v followed by a digit or a sign is never pinyin; English words
/// (video) go on with letters.
public enum QuickInput {
    private static let signs = Set("+-*/()%^.=")

    /// Whether code points are typed with vu: not with shuangpin, where vu is a syllable (zhu in
    /// Xiaohe) and a digit after it picks a candidate.
    public static var unicodeForm: Bool { InputSession.shuangpin == nil }

    /// Whether the composition is a quick form: v, then digits and signs (or u and hex digits).
    public static func isQuick(_ buffer: String) -> Bool {
        guard buffer.hasPrefix("v"), buffer.count > 1 else { return false }
        let rest = buffer.dropFirst()
        // vu and hex digits with at least one decimal digit: vue stays an English word.
        if rest.hasPrefix("u"), rest.count > 1 {
            guard unicodeForm else { return false }
            let code = rest.dropFirst()
            return code.allSatisfy(\.isHexDigit) && code.contains(where: \.isNumber)
        }
        return rest.allSatisfy { $0.isASCII && ($0.isNumber || signs.contains($0)) }
    }

    /// Whether a key that is not a letter goes into the composition: a digit or a sign after v
    /// and digits or signs (the digits are then no candidate numbers), a hex digit after vu.
    public static func accepts(_ key: Character, after buffer: String) -> Bool {
        guard buffer.hasPrefix("v") else { return false }
        if buffer.hasPrefix("vu") {
            return unicodeForm && key.isNumber && buffer.dropFirst(2).allSatisfy(\.isHexDigit) && buffer.count < 8
        }
        return key.isASCII && (key.isNumber || signs.contains(key)) && (buffer == "v" || isQuick(buffer))
    }

    /// What a quick form gives, best first; empty while it is not complete (v12+).
    public static func candidates(for buffer: String) -> [String] {
        guard isQuick(buffer) else { return [] }
        let rest = String(buffer.dropFirst())
        if rest.hasPrefix("u") {
            // Four to six hex digits, as code points are written (4e00, 1F600); no control characters.
            let code = rest.dropFirst()
            guard (4...6).contains(code.count), let value = UInt32(code, radix: 16), let scalar = Unicode.Scalar(value),
                  !scalar.properties.generalCategory.isControlOrUnassigned else { return [] }
            return [String(Character(scalar))]
        }
        let expression = rest.hasSuffix("=") ? String(rest.dropLast()) : rest
        if let number = Decimal(string: expression), expression.allSatisfy({ $0.isNumber || $0 == "." }), !expression.hasPrefix(".") {
            return numberForms(expression, number)
        }
        var calculator = Calculator(expression)
        guard let value = calculator.evaluate(), value.isFinite else { return [] }
        let result = format(value)
        return [result, expression + "=" + result]
    }

    // MARK: - Numbers


    private static let lowerDigits = Array("零一二三四五六七八九")
    private static let upperDigits = Array("零壹贰叁肆伍陆柒捌玖")
    private static let lowerUnits = ["", "十", "百", "千"]
    private static let upperUnits = ["", "拾", "佰", "仟"]
    private static let groupUnits = ["", "万", "亿", "万亿"]

    private static func numberForms(_ text: String, _ number: Decimal) -> [String] {
        let parts = text.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        let integer = String(parts[0].drop { $0 == "0" })
        let fraction = parts.count > 1 ? String(parts[1]) : ""
        guard integer.count <= 16 else { return [] }
        var forms: [String] = []
        let lower = chinese(integer, digits: lowerDigits, units: lowerUnits, plainTen: true)
            + (fraction.isEmpty ? "" : "点" + String(fraction.map { lowerDigits[$0.wholeNumberValue!] }))
        let upper = chinese(integer, digits: upperDigits, units: upperUnits, plainTen: false)
            + (fraction.isEmpty ? "" : "点" + String(fraction.map { upperDigits[$0.wholeNumberValue!] }))
        forms += [lower, upper]
        // Money: yuan, then jiao and fen (the third decimal and beyond is dropped).
        let jiao = fraction.first?.wholeNumberValue ?? 0, fen = fraction.dropFirst().first?.wholeNumberValue ?? 0
        var money = (integer.isEmpty ? "" : chinese(integer, digits: upperDigits, units: upperUnits, plainTen: false) + "元")
        if jiao == 0 && fen == 0 {
            money += integer.isEmpty ? "零元整" : "整"
        } else {
            if jiao > 0 { money += String(upperDigits[jiao]) + "角" } else if !integer.isEmpty { money += "零" }
            if fen > 0 { money += String(upperDigits[fen]) + "分" }
        }
        forms.append(money)
        let grouped = NumberFormatter.localizedString(from: number as NSDecimalNumber, number: .decimal)
        if grouped != text { forms.append(grouped) }
        var seen = Set<String>()
        return forms.filter { seen.insert($0).inserted }
    }

    /// 1234 → 一千二百三十四 / 壹仟贰佰叁拾肆; zeros inside become one 零 (1004 → 一千零四).
    private static func chinese(_ integer: String, digits: [Character], units: [String], plainTen: Bool) -> String {
        guard !integer.isEmpty else { return String(digits[0]) }
        let values = integer.compactMap(\.wholeNumberValue)
        var groups: [[Int]] = []
        var end = values.count
        while end > 0 {
            groups.insert(Array(values[max(0, end - 4)..<end]), at: 0)
            end -= 4
        }
        var result = ""
        var zeroPending = false
        for (index, group) in groups.enumerated() {
            let groupUnit = groupUnits[groups.count - 1 - index]
            if group.allSatisfy({ $0 == 0 }) { zeroPending = !result.isEmpty; continue }
            var text = ""
            for (position, digit) in group.enumerated() {
                let unit = units[group.count - 1 - position]
                if digit == 0 {
                    zeroPending = !result.isEmpty || !text.isEmpty
                } else {
                    if zeroPending { text += String(digits[0]); zeroPending = false }
                    text += String(digits[digit]) + unit
                }
            }
            // A group after a larger one that starts below the thousands reads with a 零 (1,0200 → 一万零二百).
            if !result.isEmpty, group.count == 4, group[0] == 0, !text.hasPrefix(String(digits[0])) { text = String(digits[0]) + text }
            result += text + groupUnit
            zeroPending = false
        }
        // 十二, not 一十二, when it begins the number (only for the plain digits).
        if plainTen, result.hasPrefix("一十") { result.removeFirst() }
        return result
    }

    private static func format(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 { return String(Int64(value)) }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = false
        formatter.maximumFractionDigits = 10
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }

    // MARK: - Calculator

    /// + - * / ^ %, brackets and decimals, by the usual precedence; nil when the sum is not complete.
    private struct Calculator {
        private let tokens: [Character]
        private var position = 0
        init(_ text: String) { tokens = Array(text) }

        mutating func evaluate() -> Double? {
            guard let value = sum(), position == tokens.count else { return nil }
            return value
        }
        private mutating func sum() -> Double? {
            guard var value = product() else { return nil }
            while position < tokens.count, tokens[position] == "+" || tokens[position] == "-" {
                let sign = tokens[position]; position += 1
                guard let next = product() else { return nil }
                value = sign == "+" ? value + next : value - next
            }
            return value
        }
        private mutating func product() -> Double? {
            guard var value = power() else { return nil }
            while position < tokens.count, tokens[position] == "*" || tokens[position] == "/" {
                let sign = tokens[position]; position += 1
                guard let next = power() else { return nil }
                if sign == "/" && next == 0 { return nil }
                value = sign == "*" ? value * next : value / next
            }
            return value
        }
        private mutating func power() -> Double? {
            guard let base = unary() else { return nil }
            if position < tokens.count, tokens[position] == "^" {
                position += 1
                guard let exponent = power() else { return nil }
                return pow(base, exponent)
            }
            return base
        }
        private mutating func unary() -> Double? {
            if position < tokens.count, tokens[position] == "-" { position += 1; return unary().map { -$0 } }
            guard var value = atom() else { return nil }
            while position < tokens.count, tokens[position] == "%" { position += 1; value /= 100 }
            return value
        }
        private mutating func atom() -> Double? {
            guard position < tokens.count else { return nil }
            if tokens[position] == "(" {
                position += 1
                guard let value = sum(), position < tokens.count, tokens[position] == ")" else { return nil }
                position += 1
                return value
            }
            let start = position
            while position < tokens.count, tokens[position].isNumber || tokens[position] == "." { position += 1 }
            return position > start ? Double(String(tokens[start..<position])) : nil
        }
    }

    // MARK: - Today

    /// The names followed by today's date, the time, the weekday or the lunar date.
    public static let dateNames: Set<String> = ["日期", "时间", "星期", "农历"]

    /// What the name 日期, 时间, 星期 or 农历 stands for now.
    public static func dynamic(forName name: String, now: Date = Date()) -> [String] {
        let calendar = Calendar(identifier: .gregorian)
        let parts = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second, .weekday], from: now)
        func two(_ n: Int?) -> String { String(format: "%02d", n ?? 0) }
        let (year, month, day) = (parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
        switch name {
        case "日期":
            let digitsYear = String(String(year).map { lowerDigits[$0.wholeNumberValue!] }).replacingOccurrences(of: "零", with: "〇")
            return ["\(year)-\(two(month))-\(two(day))", "\(year)年\(month)月\(day)日", "\(year)/\(month)/\(day)",
                    digitsYear + "年" + chinese(String(month), digits: lowerDigits, units: lowerUnits, plainTen: true) + "月"
                        + chinese(String(day), digits: lowerDigits, units: lowerUnits, plainTen: true) + "日"]
        case "时间":
            return ["\(two(parts.hour)):\(two(parts.minute))", "\(two(parts.hour)):\(two(parts.minute)):\(two(parts.second))",
                    "\(year)-\(two(month))-\(two(day)) \(two(parts.hour)):\(two(parts.minute))"]
        case "星期":
            let names = ["日", "一", "二", "三", "四", "五", "六"]
            let today = names[((parts.weekday ?? 1) - 1) % 7]
            return ["星期" + today, "周" + today]
        case "农历":
            // The Chinese calendar with its day names (初十, 十一, 廿一): 丙午年九月十一.
            let formatter = DateFormatter()
            formatter.calendar = Calendar(identifier: .chinese)
            formatter.locale = Locale(identifier: "zh_Hans_CN@numbers=hanidays")
            formatter.dateFormat = "U年MMMd"
            let full = formatter.string(from: now)
            formatter.dateFormat = "MMMd"
            return [full, "农历" + formatter.string(from: now)]
        default:
            return []
        }
    }
}

private extension Unicode.GeneralCategory {
    var isControlOrUnassigned: Bool { self == .control || self == .unassigned || self == .surrogate || self == .format }
}
