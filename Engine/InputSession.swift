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
            lastCommittedKeys = confirmed.map(\.raw).joined().replacingOccurrences(of: "'", with: "")
            lastCommittedContext = precedingText
            reset()
            return text
        }
        refresh()
        return nil
    }

    /// Drops the composition without committing anything.
    public func reset() {
        confirmed.removeAll()
        remaining = ""
        result = .empty
        generation += 1
    }

    private func refresh() {
        let before = precedingText + confirmed.map(\.candidate.text).joined()
        result = decoder?.decode(remaining, precedingText: before) ?? .empty
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
}
