import Foundation

/// Pairs of sounds the user may ask to be treated as interchangeable.
public enum FuzzyRule: String, CaseIterable {
    case zZh = "z-zh", cCh = "c-ch", sSh = "s-sh", nL = "n-l"
    case anAng = "an-ang", enEng = "en-eng", inIng = "in-ing"

    /// How the rule reads in the settings window.
    public var label: String {
        switch self {
        case .zZh: return "z = zh"
        case .cCh: return "c = ch"
        case .sSh: return "s = sh"
        case .nL: return "n = l"
        case .anAng: return "an = ang"
        case .enEng: return "en = eng"
        case .inIng: return "in = ing"
        }
    }
}

/// Maps a run of typed letters to the syllables it can stand for: itself, a fuzzy-sound
/// equivalent, or (in tolerant mode) a syllable one keystroke slip away.
struct SyllableMatcher {
    struct Match {
        let syllable: String
        /// Log-probability cost of reading the typed letters this way.
        let penalty: Double
    }

    /// Longest run of letters that can stand for one syllable (a 6-letter syllable plus one stray key).
    static let maxTypedLength = 7

    private let syllables: Set<String>
    private var fuzzy: [String: [Match]] = [:]
    private var typos: [String: [Match]] = [:]

    private static let fuzzyPenalty = 1.0
    /// Every typing-slip penalty is at least this, every fuzzy-sound penalty is below it.
    static let minimumTypoPenalty = 2.5
    private static let transpositionPenalty = 2.5
    private static let doubledLetterPenalty = 2.5
    private static let substitutionPenalty = 3.0
    private static let missingLetterPenalty = 3.5
    private static let strayLetterPenalty = 3.5
    /// Extra cost when the mistyped letters spell a valid syllable themselves.
    private static let realSyllablePenalty = 1.0

    private static let keyboardRows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    /// Keys next to each key on a QWERTY keyboard.
    private static let neighbors: [Character: [Character]] = {
        let rows = keyboardRows.map(Array.init)
        // Each row sits about half a key to the right of the row above.
        var result: [Character: [Character]] = [:]
        for (r, row) in rows.enumerated() {
            for (c, key) in row.enumerated() {
                var near: [Character] = []
                if c > 0 { near.append(row[c - 1]) }
                if c + 1 < row.count { near.append(row[c + 1]) }
                if r > 0 {
                    for above in [c, c + 1] where above < rows[r - 1].count { near.append(rows[r - 1][above]) }
                }
                if r + 1 < rows.count {
                    for below in [c - 1, c] where below >= 0 && below < rows[r + 1].count { near.append(rows[r + 1][below]) }
                }
                result[key] = near
            }
        }
        return result
    }()

    init(syllables: Set<String>, fuzzyRules: Set<FuzzyRule>) {
        self.syllables = syllables
        for syllable in syllables {
            for variant in Self.fuzzyVariants(of: syllable, rules: fuzzyRules) where variant != syllable {
                fuzzy[variant, default: []].append(Match(syllable: syllable, penalty: Self.fuzzyPenalty))
            }
            for (variant, slipPenalty) in Self.typoVariants(of: syllable) where variant != syllable {
                // A slip that happens to spell another syllable ("nu" for "bu", "an" for "kan") is
                // only caught from context, so it has to explain the text clearly better.
                // Single letters are left alone: "a", "e" and "o" are one slip from too many syllables.
                guard variant.count >= 2 else { continue }
                let isSyllable = syllables.contains(variant)
                let penalty = slipPenalty + (isSyllable ? Self.realSyllablePenalty : 0)
                var matches = typos[variant, default: []]
                if let index = matches.firstIndex(where: { $0.syllable == syllable }) {
                    if matches[index].penalty > penalty { matches[index] = Match(syllable: syllable, penalty: penalty) }
                } else {
                    matches.append(Match(syllable: syllable, penalty: penalty))
                }
                typos[variant] = matches
            }
        }
    }

    /// Syllables that `typed` can stand for. Slips of the finger are only considered when `tolerant`.
    func matches(_ typed: String, tolerant: Bool) -> [Match] {
        var result: [Match] = []
        if syllables.contains(typed) { result.append(Match(syllable: typed, penalty: 0)) }
        if let alternatives = fuzzy[typed] { result += alternatives }
        if tolerant, let corrections = typos[typed] { result += corrections }
        return result
    }

    private static func fuzzyVariants(of syllable: String, rules: Set<FuzzyRule>) -> Set<String> {
        var initials: [(String, String)] = []
        if rules.contains(.zZh) { initials.append(("z", "zh")) }
        if rules.contains(.cCh) { initials.append(("c", "ch")) }
        if rules.contains(.sSh) { initials.append(("s", "sh")) }
        var finals: [(String, String)] = []
        if rules.contains(.anAng) { finals.append(("an", "ang")) }
        if rules.contains(.enEng) { finals.append(("en", "eng")) }
        if rules.contains(.inIng) { finals.append(("in", "ing")) }

        var heads: Set<String> = [syllable]
        for (plain, retroflex) in initials {
            if syllable.hasPrefix(retroflex) {
                heads.insert(plain + syllable.dropFirst(retroflex.count))
            } else if syllable.hasPrefix(plain) {
                heads.insert(retroflex + syllable.dropFirst(plain.count))
            }
        }
        if rules.contains(.nL) {
            if syllable.hasPrefix("n") { heads.insert("l" + syllable.dropFirst()) }
            if syllable.hasPrefix("l") { heads.insert("n" + syllable.dropFirst()) }
        }
        var result = heads
        for head in heads {
            for (front, back) in finals {
                if head.hasSuffix(back) {
                    result.insert(String(head.dropLast()))
                } else if head.hasSuffix(front) {
                    result.insert(head + "g")
                }
            }
        }
        return result
    }

    /// Everything one slip away from `syllable`: swapped neighbours, a doubled, missing or
    /// stray letter, or a hit on an adjacent key.
    private static func typoVariants(of syllable: String) -> [(String, Double)] {
        let letters = Array(syllable)
        var result: [(String, Double)] = []
        for i in letters.indices {
            var missing = letters
            missing.remove(at: i)
            result.append((String(missing), missingLetterPenalty))

            if i + 1 < letters.count, letters[i] != letters[i + 1] {
                var swapped = letters
                swapped.swapAt(i, i + 1)
                result.append((String(swapped), transpositionPenalty))
            }

            var doubled = letters
            doubled.insert(letters[i], at: i)
            result.append((String(doubled), doubledLetterPenalty))

            for near in neighbors[letters[i]] ?? [] {
                var substituted = letters
                substituted[i] = near
                result.append((String(substituted), substitutionPenalty))

                // A neighbouring key caught together with the intended one, before or after it.
                for offset in [i, i + 1] {
                    var stray = letters
                    stray.insert(near, at: offset)
                    result.append((String(stray), strayLetterPenalty))
                }
            }
        }
        return result
    }
}
