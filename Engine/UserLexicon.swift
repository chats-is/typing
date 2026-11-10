import Foundation

/// Words from learning records that the shipped lexicon does not contain.
/// Built in memory from those records. The lexicon file is not written.
struct UserLexicon {
    struct Word {
        let text: String
        let key: String
        let syllables: [String]
    }

    /// A fixed low unigram. The record count does not raise it; training does the ranking.
    static let logp = -12.0
    private static let maxSyllables = 8
    private static let searchLimit = 2048

    private var words: [Word] = []
    private var byKey: [String: [Word]] = [:]
    private var readings: [Character: [String]] = [:]
    private var readingsLexicon: ObjectIdentifier?

    mutating func rebuild(entries: [LearningStore.Entry], lexicon: Lexicon) {
        prepareReadings(lexicon)
        var keysByText: [String: [String]] = [:]
        for entry in entries {
            let text = entry.text
            guard text.count >= 2, text.count <= Self.maxSyllables, readings(for: text) != nil else { continue }
            let keys = entry.keys.lowercased().filter { $0.isASCII && $0.isLetter }
            guard !keys.isEmpty else { continue }
            if keysByText[text]?.contains(keys) != true { keysByText[text, default: []].append(keys) }
        }
        var built: [Word] = []
        for (text, keys) in keysByText {
            guard let syllables = choose(text: text, keys: keys) else { continue }
            let key = syllables.joined(separator: "'")
            if lexicon.exact(key).contains(where: { $0.word == text }) { continue }
            built.append(Word(text: text, key: key, syllables: syllables))
        }
        words = built
        byKey = Dictionary(grouping: built, by: \.key)
    }

    /// Replaces one word. A record write touches only that word, not the whole list.
    mutating func replace(text: String, keys: [String], lexicon: Lexicon) {
        prepareReadings(lexicon)
        var updated = words.filter { $0.text != text }
        let usable = keys.filter { !$0.isEmpty }
        if text.count >= 2, text.count <= Self.maxSyllables, let syllables = choose(text: text, keys: usable) {
            let key = syllables.joined(separator: "'")
            if !lexicon.exact(key).contains(where: { $0.word == text }) {
                updated.append(Word(text: text, key: key, syllables: syllables))
            }
        }
        words = updated
        byKey = Dictionary(grouping: updated, by: \.key)
    }

    func exact(_ key: String) -> [Lexicon.Entry] {
        (byKey[key] ?? []).map(entry)
    }

    func hasPrefix(_ key: String) -> Bool {
        let upper = key + "("
        return words.contains { $0.key >= key && $0.key < upper }
    }

    func partial(syllableCount: Int, keyPrefix: String) -> [Lexicon.Entry] {
        let upper = keyPrefix + "{"
        return words.filter { $0.syllables.count == syllableCount && $0.key >= keyPrefix && $0.key < upper }.map(entry)
    }

    /// Same shape as `Lexicon.abbreviated`: one segment per syllable, complete or a prefix of it.
    func abbreviated(_ segments: [(text: String, complete: Bool)]) -> [Lexicon.Entry] {
        guard segments.count >= 2 else { return [] }
        return words.filter { word in
            guard word.syllables.count == segments.count else { return false }
            for (syllable, segment) in zip(word.syllables, segments) {
                if segment.complete {
                    if syllable != segment.text { return false }
                } else if !syllable.hasPrefix(segment.text) {
                    return false
                }
            }
            return true
        }.map(entry)
    }

    private func entry(_ word: Word) -> Lexicon.Entry {
        Lexicon.Entry(word: word.text, wid: 0, key: word.key, logp: Self.logp)
    }

    /// Readings of each character, from the shipped lexicon. Cached per lexicon instance.
    private mutating func prepareReadings(_ lexicon: Lexicon) {
        let id = ObjectIdentifier(lexicon)
        if readingsLexicon == id, !readings.isEmpty { return }
        var map: [Character: [String]] = [:]
        for syllable in lexicon.syllables {
            for item in lexicon.characters(syllable) where item.word.count == 1 {
                let character = Character(item.word)
                if map[character]?.contains(syllable) != true { map[character, default: []].append(syllable) }
            }
        }
        readings = map
        readingsLexicon = id
    }

    private func readings(for text: String) -> [[String]]? {
        var lists: [[String]] = []
        for character in text {
            guard let list = readings[character], !list.isEmpty else { return nil }
            lists.append(list)
        }
        return lists
    }

    /// The reading every recorded key can be typed as. Nil when none fits, more than one fits equally,
    /// or the search stops before every reading has been tried.
    private func choose(text: String, keys: [String]) -> [String]? {
        guard !keys.isEmpty, let lists = readings(for: text) else { return nil }
        let longest = keys.map(\.count).max() ?? 0
        var matches: [[String]] = []
        // Full-syllable counts of the longest keys, largest first. A higher list wins;
        // an equal list is a tie and the word is not built.
        var bestScore: [Int] = []
        var visited = 0
        var aborted = false
        func search(_ index: Int, _ syllables: [String]) {
            if aborted { return }
            visited += 1
            if visited > Self.searchLimit {
                aborted = true
                return
            }
            if index == lists.count {
                var score: [Int] = []
                for key in keys where key.count == longest {
                    guard let count = consumedFully(key, syllables: syllables) else { return }
                    score.append(count)
                }
                score.sort(by: >)
                for key in keys where key.count != longest {
                    if consumedFully(key, syllables: syllables) == nil { return }
                }
                if matches.isEmpty || Self.outranks(score, bestScore) {
                    bestScore = score
                    matches = [syllables]
                } else if score == bestScore {
                    matches.append(syllables)
                }
                return
            }
            for reading in lists[index] {
                let next = syllables + [reading]
                guard keys.allSatisfy({ consumed(keys: $0, syllables: next) }) else { continue }
                search(index + 1, next)
                if aborted { return }
            }
        }
        search(0, [])
        guard !aborted, matches.count == 1 else { return nil }
        return matches[0]
    }

    /// `score` is better when an earlier full-syllable count is higher.
    private static func outranks(_ score: [Int], _ best: [Int]) -> Bool {
        for (a, b) in zip(score, best) where a != b { return a > b }
        return score.count > best.count
    }

    /// True while `keys` still matches this unfinished reading. Later syllables may eat what is left.
    private func consumed(keys: String, syllables: [String]) -> Bool {
        var rest = Substring(keys)
        for syllable in syllables {
            if rest.hasPrefix(syllable) {
                rest = rest.dropFirst(syllable.count)
            } else {
                let initial = Self.initial(syllable)
                guard rest.hasPrefix(initial) else { return false }
                rest = rest.dropFirst(initial.count)
            }
        }
        return true
    }

    /// How many syllables were typed in full. Nil when `keys` is not this reading.
    private func consumedFully(_ keys: String, syllables: [String]) -> Int? {
        var rest = Substring(keys)
        var full = 0
        for syllable in syllables {
            if rest.hasPrefix(syllable) {
                rest = rest.dropFirst(syllable.count)
                full += 1
            } else {
                let initial = Self.initial(syllable)
                guard rest.hasPrefix(initial) else { return nil }
                rest = rest.dropFirst(initial.count)
            }
        }
        return rest.isEmpty ? full : nil
    }

    /// The letters an abbreviated syllable keeps: zh, ch, sh, or the first letter.
    private static func initial(_ syllable: String) -> String {
        if syllable.hasPrefix("zh") || syllable.hasPrefix("ch") || syllable.hasPrefix("sh") { return String(syllable.prefix(2)) }
        return String(syllable.prefix(1))
    }
}
