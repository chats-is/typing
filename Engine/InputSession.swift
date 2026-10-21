import Foundation

/// The state of one composition: the pinyin still to convert plus the candidates already
/// picked for the front of it. UI-agnostic so it can be driven from tests or the IME.
public final class InputSession {
    private let decoder: CandidateDecoding?
    /// Candidates the user has picked so far, with the raw pinyin each one replaced.
    private var confirmed: [(candidate: Candidate, raw: String)] = []
    /// Raw pinyin not converted yet.
    public private(set) var remaining = ""
    private var result = DecodeResult.empty
    /// The text in front of the composition, set by the host when a composition starts.
    public var precedingText = ""
    /// The letters typed and the text before the cursor for the composition committed last (what learning records).
    public private(set) var lastCommittedKeys = ""
    public private(set) var lastCommittedContext = ""
    /// Called when a deferred second look changed the candidates after `insert`/`backspace` returned.
    public var onUpdate: (() -> Void)?
    private var generation = 0

    private static let maxBufferLength = 80

    /// The shuangpin layout in use for every composition, nil for full pinyin (set from Settings).
    public static var shuangpin: Shuangpin?

    /// The full pinyin the keys spell (the keys themselves without shuangpin), "'" left out: what
    /// learning records them by.
    public static func pinyinKeys(_ keys: String) -> String {
        let pinyin = shuangpin.map { $0.read(keys).pinyin } ?? keys
        return pinyin.replacingOccurrences(of: "'", with: "")
    }

    /// The full pinyin of the first `consumed` keys still being composed.
    public func pinyinKeys(consumed: Int) -> String { Self.pinyinKeys(String(remaining.prefix(consumed))) }

    public init(decoder: CandidateDecoding?) {
        self.decoder = decoder
    }

    public var isComposing: Bool { !confirmed.isEmpty || !remaining.isEmpty }
    public var candidates: [Candidate] { result.candidates }

    /// Text to show inline while composing.
    public var preedit: String {
        confirmed.map(\.candidate.text).joined() + (result.candidates.isEmpty ? remaining : result.segmented)
    }

    /// Picked candidates followed by the unconverted pinyin as typed.
    public var rawText: String {
        confirmed.map(\.candidate.text).joined() + remaining
    }

    /// What sending the letters as typed gives: picked candidates, then the letters not converted
    /// with their "'" as typed (xi'an, don't); a "'" left in front of them by a pick is dropped.
    public var typedText: String {
        let letters = confirmed.isEmpty ? Substring(remaining) : remaining.drop { $0 == "'" }
        return confirmed.map(\.candidate.text).joined() + letters
    }

    /// Appends a letter (capitals are kept as typed inside the composition) or a "'" separator.
    public func insert(_ character: Character) {
        guard remaining.count < Self.maxBufferLength else { return }
        if character == "'" && (remaining.isEmpty || remaining.hasSuffix("'")) { return }
        remaining.append(character)
        refresh()
    }

    /// Removes the last letter, or un-picks the last candidate when no letters are left.
    public func backspace() {
        if !remaining.isEmpty {
            remaining.removeLast()
        } else if let last = confirmed.popLast() {
            remaining = last.raw
        }
        refresh()
    }

    /// Picks a candidate. Returns the full text to commit once the whole buffer is converted.
    public func select(_ index: Int) -> String? {
        guard result.candidates.indices.contains(index) else { return nil }
        let candidate = result.candidates[index]
        confirmed.append((candidate, String(remaining.prefix(candidate.consumed))))
        remaining.removeFirst(candidate.consumed)
        if remaining.isEmpty {
            let text = rawText
            lastCommittedKeys = confirmed.map { Self.pinyinKeys($0.raw) }.joined()
            lastCommittedContext = precedingText
            lastCommitNotLearned = pickedCharacter || confirmed.contains { $0.candidate.words.isEmpty }
            reset()
            return text
        }
        refresh()
        return nil
    }

    /// Picks one character of a candidate for all the letters the candidate stands for (以词定字:
    /// the first or the last character of 编辑 for bianji). Returns the full text once the whole
    /// buffer is converted, as `select` does.
    public func selectCharacter(_ index: Int, last: Bool) -> String? {
        guard result.candidates.indices.contains(index) else { return nil }
        let candidate = result.candidates[index]
        guard let character = last ? candidate.text.last : candidate.text.first else { return nil }
        let picked = Candidate(text: String(character), consumed: candidate.consumed, words: [], score: candidate.score)
        confirmed.append((picked, String(remaining.prefix(candidate.consumed))))
        remaining.removeFirst(candidate.consumed)
        pickedCharacter = true
        if remaining.isEmpty {
            let text = rawText
            lastCommittedKeys = confirmed.map { Self.pinyinKeys($0.raw) }.joined()
            lastCommittedContext = precedingText
            lastCommitNotLearned = true
            reset()
            return text
        }
        refresh()
        return nil
    }
    /// Whether the composition committed last is no reading to learn: a character picked out of a
    /// word (its letters spell the word) or a quick result (v1234 → 壹仟贰佰叁拾肆).
    public private(set) var lastCommitNotLearned = false
    private var pickedCharacter = false

    /// Drops the composition without committing anything.
    public func reset() {
        pickedCharacter = false
        confirmed.removeAll()
        remaining = ""
        result = .empty
        generation += 1
    }

    /// Reads the letters again (after what was learned for them changed).
    public func refreshCandidates() { refresh() }

    private func refresh() {
        let before = precedingText + confirmed.map(\.candidate.text).joined()
        if let shuangpin = Self.shuangpin, !QuickInput.isQuick(remaining) {
            // Two keys a syllable: decoded as the full pinyin they spell, lengths given back in keys.
            let (pinyin, keysAt) = shuangpin.read(remaining)
            result = Self.inKeys(decoder?.decode(pinyin, precedingText: before) ?? .empty, keysAt: keysAt)
        } else {
            result = decoder?.decode(remaining, precedingText: before) ?? .empty
        }
        generation += 1
        guard let refine = result.refine else { return }
        // Show the quick result first; the slower look (the model's rerank) runs off the main
        // thread so the next keystroke is never queued behind it, and lands only if still current.
        let current = generation
        let isCancelled: () -> Bool = { [weak self] in self.map { $0.generation != current } ?? true }
        // One serial queue: a burst of keys leaves a line of looks, each of which sees at once that
        // it is stale and returns, instead of all of them queueing on the model.
        Self.refineQueue.async { [weak self] in
            guard let self, current == self.generation, let better = refine(isCancelled) else { return }
            DispatchQueue.main.async {
                guard current == self.generation else { return }
                self.result = better
                self.onUpdate?()
            }
        }
    }

    private static let refineQueue = DispatchQueue(label: "typing.refine", qos: .userInteractive)

    /// A result decoded from the full pinyin of shuangpin keys, its lengths counted in keys again
    /// and its syllables shown apart (ni hao); so is the result of its second look.
    private static func inKeys(_ result: DecodeResult, keysAt: [Int]) -> DecodeResult {
        func keys(_ candidate: Candidate) -> Candidate {
            var copy = Candidate(text: candidate.text, consumed: keysAt[min(candidate.consumed, keysAt.count - 1)],
                                 words: candidate.words, score: candidate.score)
            copy.penalty = candidate.penalty
            return copy
        }
        let refine = result.refine.map { look in
            { (isCancelled: @escaping () -> Bool) -> DecodeResult? in look(isCancelled).map { inKeys($0, keysAt: keysAt) } }
        }
        return DecodeResult(candidates: result.candidates.map(keys), sentences: result.sentences.map(keys),
                            segmented: result.segmented.replacingOccurrences(of: "'", with: " "), refine: refine)
    }
}
