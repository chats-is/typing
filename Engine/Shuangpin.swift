import Foundation

/// Shuangpin: every syllable typed with two keys, its initial and its final, by one of the usual
/// layouts. The keys are read as full pinyin (nihc → ni'hao in Xiaohe) and decoded as such, so the
/// lexicon, the model, learning and fuzzy sounds work unchanged.
/// The layouts follow the published schemes, checked against fcitx5's libime and the Rime
/// configuration rime-ice (雾凇拼音).
public enum ShuangpinLayout: String, CaseIterable {
    case xiaohe, ziranma, microsoft, sogou

    public var label: String {
        switch self {
        case .xiaohe: return "小鹤双拼"
        case .ziranma: return "自然码"
        case .microsoft: return "微软双拼"
        case .sogou: return "搜狗双拼"
        }
    }

    /// The key of each final that is not its own single letter.
    fileprivate var finals: [String: Character] {
        switch self {
        case .microsoft, .sogou:
            // Sogou is Microsoft's layout with ve on t only (rime-ice double_pinyin_sogou).
            var keys: [String: Character] = ["ai": "l", "an": "j", "ang": "h", "ao": "k", "ei": "z", "en": "f", "eng": "g", "er": "r", "ia": "w", "ian": "m",
                    "iang": "d", "iao": "c", "ie": "x", "in": "n", "ing": ";", "iong": "s", "iu": "q", "ong": "s", "ou": "b", "ua": "w",
                    "uai": "y", "uan": "r", "uang": "d", "ue": "t", "ui": "v", "un": "p", "uo": "o", "ve": "v", "v": "y"]
            if self == .sogou { keys["ve"] = "t" }
            return keys
        case .ziranma:
            return ["ai": "l", "an": "j", "ang": "h", "ao": "k", "ei": "z", "en": "f", "eng": "g", "er": "r", "ia": "w", "ian": "m",
                    "iang": "d", "iao": "c", "ie": "x", "in": "n", "ing": "y", "iong": "s", "iu": "q", "ong": "s", "ou": "b", "ua": "w",
                    "uai": "y", "uan": "r", "uang": "d", "ue": "t", "ui": "v", "un": "p", "uo": "o", "ve": "t", "v": "v"]
        case .xiaohe:
            return ["ai": "d", "an": "j", "ang": "h", "ao": "c", "ei": "w", "en": "f", "eng": "g", "ia": "x", "ian": "m", "iang": "l",
                    "iao": "n", "ie": "p", "in": "b", "ing": "k", "iong": "s", "iu": "q", "ong": "s", "ou": "z", "ua": "x", "uai": "k",
                    "uan": "r", "uang": "l", "ue": "t", "ui": "v", "un": "y", "uo": "o", "ve": "t", "v": "v"]
        }
    }

    /// The keys of zh, ch and sh.
    fileprivate var retroflex: [String: Character] {
        ["zh": "v", "ch": "i", "sh": "u"]
    }

    /// How a syllable without an initial is typed: after a lead key ("o"); and/or spelled: its
    /// single letter doubled (aa), its two letters as they are (ai), or its first letter and the
    /// final's key (ah for ang); and/or, for finals starting with a or e, that letter and the
    /// final's key (aj for an, aa for a), as rime-ice types Microsoft and Sogou.
    fileprivate var zeroInitial: (lead: Character?, spelled: Bool, aeLeading: Bool) {
        switch self {
        case .xiaohe: return (nil, true, false)
        case .ziranma: return ("o", true, false)
        case .microsoft, .sogou: return ("o", false, true)
        }
    }
}

/// A layout's two-key table, for the syllables the lexicon knows.
public final class Shuangpin {
    public let layout: ShuangpinLayout
    /// Whether ";" is a final's key (ing in Microsoft and Sogou): then it is a letter after an initial.
    public var usesSemicolon: Bool { layout.finals.values.contains(";") }

    /// Whether the keys end with an initial's key still waiting for its final (x in Microsoft's
    /// x;), as the keys are read: capitals and "'" do not count.
    public func waitsForFinal(_ keys: String) -> Bool {
        let typed = Array(keys)
        var i = 0
        var waiting = false
        while i < typed.count {
            if typed[i] == "'" || typed[i].isUppercase { i += 1; waiting = false; continue }
            if i + 1 < typed.count, !typed[i + 1].isUppercase, pairs[String(typed[i...i + 1])] != nil {
                i += 2; waiting = false
            } else {
                waiting = singles[typed[i]] != nil
                i += 1
            }
        }
        return waiting
    }
    /// Two keys → the syllable they spell (the first of a few when the final key is shared and both
    /// are syllables; the rare ones are dropped).
    private var pairs: [String: String] = [:]
    /// One key at the end: the initial it starts (v → zh in Xiaohe).
    private var singles: [Character: String] = [:]
    /// The other way: the two keys of each syllable (for tests that type full-pinyin sentences).
    private var keysOf: [String: String] = [:]

    private static let initials = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w", "zh", "ch", "sh"]
    private static let finals = ["a", "o", "e", "i", "u", "v", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "er",
                                 "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong", "ua", "uo", "uai", "ui", "uan", "un", "uang",
                                 "ue", "ve"]
    private static let zeroFinals = ["a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"]

    /// `weights` (how many characters each syllable has) decides between two syllables on the
    /// same two keys: lo is luo (罗 落 …; 咯 is read luo as well), not lo.
    public init(_ layout: ShuangpinLayout, syllables: Set<String>, weights: [String: Double] = [:]) {
        self.layout = layout
        func place(_ typed: String, _ syllable: String) {
            if let other = pairs[typed], (weights[other] ?? -.infinity) >= (weights[syllable] ?? -.infinity) { return }
            pairs[typed] = syllable
        }
        let finalKeys = layout.finals
        func key(ofFinal final: String) -> Character? { finalKeys[final] ?? (final.count == 1 ? final.first : nil) }
        for initial in Self.initials {
            guard let initialKey = layout.retroflex[initial] ?? (initial.count == 1 ? initial.first : nil) else { continue }
            singles[initialKey] = singles[initialKey] ?? initial
            for final in Self.finals {
                // ü is written u after j, q, x and y (ju, xue), v after n and l (lv, nve).
                let spelled = ["j", "q", "x", "y"].contains(initial) ? final.replacingOccurrences(of: "v", with: "u") : final
                let syllable = initial + spelled
                guard syllables.contains(syllable), let finalKey = key(ofFinal: final) else { continue }
                place(String([initialKey, finalKey]), syllable)
            }
        }
        let zero = layout.zeroInitial
        for final in Self.zeroFinals where syllables.contains(final) {
            let finalKey = key(ofFinal: final)
            var ways: [String] = []
            if let lead = zero.lead, let finalKey { ways.append(String([lead, finalKey])) }
            if zero.spelled {
                // aa; ai and er as they are (also a and ai's key); ah for ang (its first letter, its final's key).
                if final.count == 1 { ways.append(final + final) }
                if final.count == 2 { ways.append(final) }
                if final.count >= 2, let finalKey { ways.append(String([final.first!, finalKey])) }
            }
            if zero.aeLeading, final.first == "a" || final.first == "e" {
                if final.count == 1 { ways.append(final + final) } else if let finalKey { ways.append(String([final.first!, finalKey])) }
            }
            for typed in ways where pairs[typed] == nil { pairs[typed] = final }
        }
        // The keys of each syllable: two keys, the shortest way when there are several (oa and aa).
        for (typed, syllable) in pairs.sorted(by: { $0.key < $1.key }) where keysOf[syllable] == nil { keysOf[syllable] = typed }
    }

    /// The two keys of a full-pinyin syllable, nil when the layout has none for it.
    public func keys(for syllable: String) -> String? { keysOf[syllable] }

    /// The keys read as full pinyin, syllables apart ("nihc" → "ni'hao"), and for each letter of
    /// that reading how many keys it stands at (to give the candidates' lengths back in keys).
    public func read(_ keys: String) -> (pinyin: String, keysAt: [Int]) {
        var pinyin = ""
        var keysAt: [Int] = [0]
        let typed = Array(keys)
        var i = 0
        // keysAt[n]: the keys behind the first n letters of the reading (a syllable's letters all
        // stand at the end of its two keys; a separator at the end of the syllable before it).
        func append(_ text: String, upTo keyIndex: Int) {
            if !pinyin.isEmpty { pinyin += "'"; keysAt.append(keysAt.last!) }
            for _ in text { keysAt.append(keyIndex) }
            pinyin += text
        }
        while i < typed.count {
            if typed[i] == "'" { i += 1; continue }
            // Capitals stay as they are, as with full pinyin (对PC行业): no key pair.
            if typed[i].isUppercase {
                var end = i
                while end < typed.count, typed[end].isUppercase { end += 1 }
                append(String(typed[i..<end]), upTo: end)
                i = end
                continue
            }
            if i + 1 < typed.count, !typed[i + 1].isUppercase, let syllable = pairs[String(typed[i...i + 1])] {
                append(syllable, upTo: i + 2)
                i += 2
            } else {
                // A key left on its own: the initial it starts, or the letter itself.
                append(singles[typed[i]] ?? String(typed[i]), upTo: i + 1)
                i += 1
            }
        }
        return (pinyin, keysAt)
    }
}
