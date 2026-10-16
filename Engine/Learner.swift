import Foundation

/// Runs a learning session: tunes the model's top blocks on what the person has confirmed, checks
/// the result on the person's most recent entries and on a bundled general set, and keeps it only
/// if the person's input comes out better without the general input getting worse.
public final class Learner {
    public typealias Readings = (_ keys: String, _ context: String) -> [(text: String, score: Double, penalty: Double)]

    public struct Outcome {
        public var accepted = false
        public var reason = ""
        public var steps = 0
        public var personalBefore = 0.0, personalAfter = 0.0, personalCount = 0
        public var generalBefore = 0.0, generalAfter = 0.0, generalCount = 0
        public var wordsBefore = 0.0, wordsAfter = 0.0, wordsCount = 0
        public var seconds = 0.0
        /// The tuned model that was kept, ready for the engine (nil when nothing was kept).
        public var scorer: GraphScorer?
    }

    public let store: LearningStore
    public let weights: GraphScorer.WeightsSource
    public let vocabularyURL: URL
    public let generalLatticesURL: URL?
    /// Lone common words (lexicon order expected): learning may move them only within tolerance.
    public var generalWordsURL: URL?
    private let readings: Readings
    private let ids: [Character: Int32]

    /// A fixed seed makes a session reproducible (tests); nil draws the training windows at random.
    public var seed: UInt64?
    /// Session policy.
    public var minimumNewEntries = 20
    public var minimumInterval: TimeInterval = 600
    public var maximumSteps = 300
    /// Enough steps for a small amount of text to be learned (validation and rollback guard the rest).
    public var minimumSteps = 100
    public var charactersPerStep = 250
    public var learningRate: Float = 2e-5
    public var heldOutFraction = 0.1
    public var heldOutMinimum = 20
    public var heldOutMaximum = 100
    /// Allowed drops (percentage points) before a result is rejected.
    public var personalTolerance = 0.5
    public var generalTolerance = 1.0
    public var wordsTolerance = 1.0
    /// Set from another thread to stop a running session early (checked between steps and before
    /// anything is written).
    public var cancelled: Bool {
        get { pauseLock.lock(); defer { pauseLock.unlock() }; return isCancelled }
        set { pauseLock.lock(); isCancelled = newValue; pauseLock.broadcast(); pauseLock.unlock() }
    }
    private var isCancelled = false
    /// Set while the person is typing: the session waits between steps and cases so the GPU and
    /// the graph lock stay free for the input method; cleared when the keyboard is idle again.
    public var paused: Bool {
        get { pauseLock.lock(); defer { pauseLock.unlock() }; return isPaused }
        set { pauseLock.lock(); isPaused = newValue; pauseLock.broadcast(); pauseLock.unlock() }
    }
    private var isPaused = false
    private let pauseLock = NSCondition()
    private func waitWhilePaused() {
        pauseLock.lock()
        while isPaused && !isCancelled { pauseLock.wait(until: Date(timeIntervalSinceNow: 0.5)) }
        pauseLock.unlock()
    }
    /// The model the engine is ranking with right now (the shipped one, or the shipped one plus
    /// the kept tuning): the session's "before", so no second copy of it has to be built.
    public var engineScorer: (() -> SentenceScorer?)?

    private static let contextWeight = 2.0, latticeWeight = 1.0

    public init(store: LearningStore, weights: GraphScorer.WeightsSource, vocabularyURL: URL, generalLatticesURL: URL?, readings: @escaping Readings) throws {
        self.store = store; self.weights = weights; self.vocabularyURL = vocabularyURL
        self.generalLatticesURL = generalLatticesURL; self.readings = readings
        let vocabulary = try JSONDecoder().decode([String].self, from: Data(contentsOf: vocabularyURL))
        var ids: [Character: Int32] = [:]
        for (index, token) in vocabulary.enumerated() where token.count == 1 { ids[Character(token)] = Int32(index) }
        self.ids = ids
    }

    private func encode(_ text: String) -> [Int32] { text.map { ids[$0] ?? $0.lowercased().first.flatMap { ids[$0] } ?? 2 } }

    /// Whether enough new material has arrived since the last session.
    public func due(now: Date = Date()) -> Bool {
        let state = store.state()
        let newEntries = store.count - state.entriesAtLastSession
        guard newEntries >= minimumNewEntries else { return false }
        return now.timeIntervalSince1970 - state.lastSessionTime >= minimumInterval || newEntries >= 6 * minimumNewEntries
    }

    /// A general set: lattice dumps "keys<TAB>expected<TAB>context<TAB>text|score|penalty ;; ...".
    struct Case { let expected: String; let context: String; let readings: [(text: String, score: Double, penalty: Double)] }

    private func generalCases() -> [Case] { cases(at: generalLatticesURL) }
    private func generalWordCases() -> [Case] { cases(at: generalWordsURL) }

    private func cases(at url: URL?) -> [Case] {
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            let f = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard f.count >= 4 else { return nil }
            let readings = f[3].split(separator: " ;; ").compactMap { item -> (String, Double, Double)? in
                let parts = item.split(separator: "|")
                guard parts.count >= 3, let s = Double(parts[parts.count - 2]), let p = Double(parts[parts.count - 1]) else { return nil }
                return (parts.dropLast(2).joined(separator: "|"), s, p)
            }
            return Case(expected: String(f[1]), context: f[2].replacingOccurrences(of: "\\n", with: "\n"), readings: readings.map { (text: $0.0, score: $0.1, penalty: $0.2) })
        }
    }

    /// Exact first-choice rate of `cases` when `scorer` reranks the lattice readings the engine's way.
    func exactRate(_ cases: [Case], scorer: SentenceScorer) -> Double {
        guard !cases.isEmpty else { return 0 }
        var exact = 0
        for c in cases where !c.readings.isEmpty {
            waitWhilePaused()
            let usable = c.context.replacingOccurrences(of: "\n", with: "")
            let prefix: [Int32] = usable.isEmpty ? [1] : [1] + scorer.contextTokens(usable, reserving: 1)
            var best: (score: Double, text: String)?
            var start = 0
            while start < c.readings.count {
                let batch = Array(c.readings[start ..< min(start + scorer.batchSize, c.readings.count)])
                guard let lm = scorer.score(prefix: prefix, candidates: batch.map(\.text)) else { break }
                for (reading, value) in zip(batch, lm) {
                    let final = Self.latticeWeight * (reading.score + reading.penalty) + Self.contextWeight * value - (Self.latticeWeight + Self.contextWeight) * reading.penalty
                    if best == nil || final > best!.score { best = (final, reading.text) }
                }
                start += batch.count
            }
            if best?.text == c.expected { exact += 1 }
        }
        return 100 * Double(exact) / Double(cases.count)
    }

    /// Exact rate of lone words under the engine's lone-word rule with learning: lexicon score +
    /// contextWeight × (log P tuned − log P shipped); `tuned == shipped` gives the lexicon order.
    func loneExactRate(_ cases: [Case], tuned: SentenceScorer, shipped: SentenceScorer) -> Double {
        guard !cases.isEmpty else { return 0 }
        let learned = store.learnedStrings()
        var exact = 0
        for c in cases where !c.readings.isEmpty {
            waitWhilePaused()
            var best: (score: Double, text: String)?
            var start = 0
            while start < c.readings.count {
                let batch = Array(c.readings[start ..< min(start + tuned.batchSize, c.readings.count)])
                guard let a = tuned.score(prefix: [1], candidates: batch.map(\.text)), let b = shipped.score(prefix: [1], candidates: batch.map(\.text)) else { break }
                for (reading, (ta, tb)) in zip(batch, zip(a, b)) {
                    let final = reading.score + (learned.contains(reading.text) ? Self.contextWeight * (ta - tb) : 0)
                    if best == nil || final > best!.score { best = (final, reading.text) }
                }
                start += batch.count
            }
            if best?.text == c.expected { exact += 1 }
        }
        return 100 * Double(exact) / Double(cases.count)
    }

    /// Runs one session; blocking, meant for a background thread.
    public func runSession(log: ((String) -> Void)? = nil) -> Outcome {
        var outcome = Outcome()
        let started = Date()
        var state = store.state()
        let entries = store.entries()
        // Only a session that trained is booked (a cancelled one writes nothing: the store may
        // just have been reset); one that found too little text is simply tried again later.
        var trained = false
        defer {
            if trained && !cancelled {
                state.lastSessionTime = Date().timeIntervalSince1970
                state.entriesAtLastSession = entries.count
                state.sessions += 1
                if outcome.accepted { state.accepted += 1 }
                state.lastPersonalBefore = outcome.personalBefore; state.lastPersonalAfter = outcome.personalAfter
                state.lastGeneralBefore = outcome.generalBefore; state.lastGeneralAfter = outcome.generalAfter
                state.lastDecision = outcome.reason
                store.save(state)
            }
        }
        guard entries.count >= minimumNewEntries else { outcome.reason = "too little text"; return outcome }

        // Held out: the most recent entries, judged with their real keys and context.
        let heldOutCount = min(heldOutMaximum, max(heldOutMinimum, Int(Double(entries.count) * heldOutFraction)))
        var heldOut: [LearningStore.Entry] = []
        var training: [LearningStore.Entry] = []
        for entry in entries.reversed() {
            if heldOut.count < heldOutCount, !entry.keys.isEmpty, heldOut.count + training.count < entries.count - heldOutCount || training.isEmpty {
                heldOut.append(entry)
            } else {
                training.append(entry)
            }
        }
        heldOut.reverse(); training.reverse()
        if heldOut.count >= entries.count { training = entries; heldOut = [] }
        let text = training.map(\.text).joined(separator: "\n")
        let stream = encode(text)
        guard stream.count > 2 * 64 + 2 else { outcome.reason = "too little text"; return outcome }  // two training windows

        do {
            // Each session learns the current material from the shipped weights, so what was learned is
            // always exactly a function of the entries the person can see and delete.
            let shippedWeights = try weights.load()
            let hasTuned = store.hasTunedWeights
            waitWhilePaused()
            let personalCases = heldOut.map { Case(expected: $0.text, context: $0.context, readings: readings($0.keys, $0.context)) }
            let general = generalCases()
            let words = generalWordCases()
            outcome.personalCount = personalCases.count
            outcome.generalCount = general.count
            outcome.wordsCount = words.count
            // Before and after are judged on the same runtime (the GPU one); the shipped model on
            // that runtime is the reference of the lone-word rule, so that with nothing learned
            // the rule moves nothing at all.
            let shipped: GraphScorer? = hasTuned ? try GraphScorer(weights: shippedWeights, vocabularyURL: vocabularyURL) : nil
            do {
                // "Before" is what the engine ranks with now when that is the GPU runtime;
                // otherwise built here and released again before training.
                let before: SentenceScorer
                if let engine = engineScorer?() {
                    before = engine
                } else if hasTuned {
                    var current = shippedWeights
                    current.apply(tuned: try GraphScorer.Weights(path: store.tunedBase))
                    before = try GraphScorer(weights: current, vocabularyURL: vocabularyURL)
                } else if let shipped {
                    before = shipped
                } else {
                    before = try GraphScorer(weights: shippedWeights, vocabularyURL: vocabularyURL)
                }
                outcome.personalBefore = exactRate(personalCases, scorer: before)
                outcome.generalBefore = exactRate(general, scorer: before)
                outcome.wordsBefore = loneExactRate(words, tuned: before, shipped: shipped ?? before)
            }
            log?("before: personal \(outcome.personalBefore) general \(outcome.generalBefore) words \(outcome.wordsBefore)")

            waitWhilePaused()
            if cancelled { outcome.reason = "cancelled"; return outcome }
            trained = true
            let candidateBase = store.tunedBase + ".new"
            do {
                // The trainer (fp32 copies, Adam state) lives only until its result is saved.
                let trainer = try GraphTrainer(weights: shippedWeights)
                let steps = min(maximumSteps, max(minimumSteps, stream.count / charactersPerStep))
                var generator: any RandomNumberGenerator = seed.map { SplitMix64(seed: $0) as any RandomNumberGenerator } ?? SystemRandomNumberGenerator()
                for step in 1...steps {
                    waitWhilePaused()
                    if cancelled { outcome.reason = "cancelled"; return outcome }
                    let starts = (0..<trainer.batch).map { _ in Int.random(in: 0 ..< max(1, stream.count - trainer.rows - 1), using: &generator) }
                    let loss = trainer.step(stream: stream, starts: starts, learningRate: learningRate)
                    if step % 50 == 0 { log?("step \(step)/\(steps) loss \(loss)") }
                }
                outcome.steps = steps
                waitWhilePaused()
                if cancelled { outcome.reason = "cancelled"; return outcome }
                try trainer.save(to: candidateBase, config: shippedWeights.config)
            }
            var tunedWeights = shippedWeights
            tunedWeights.apply(tuned: try GraphScorer.Weights(path: candidateBase))
            let after = try GraphScorer(weights: tunedWeights, vocabularyURL: vocabularyURL)
            outcome.personalAfter = exactRate(personalCases, scorer: after)
            outcome.generalAfter = exactRate(general, scorer: after)
            outcome.wordsAfter = try loneExactRate(words, tuned: after, shipped: shipped ?? GraphScorer(weights: shippedWeights, vocabularyURL: vocabularyURL))
            log?("after: personal \(outcome.personalAfter) general \(outcome.generalAfter) words \(outcome.wordsAfter)")

            let wordsOK = words.isEmpty || outcome.wordsAfter >= outcome.wordsBefore - wordsTolerance
            let generalOK = (general.isEmpty || outcome.generalAfter >= outcome.generalBefore - generalTolerance) && wordsOK
            let personalOK = personalCases.count < heldOutMinimum || outcome.personalAfter >= outcome.personalBefore - personalTolerance
            if cancelled { outcome.reason = "cancelled"; return outcome }
            if generalOK && personalOK {
                // The .bin goes first: a .json without its .bin would be taken for tuned weights.
                for ext in [".bin", ".json"] {
                    let from = URL(fileURLWithPath: candidateBase + ext), to = URL(fileURLWithPath: store.tunedBase + ext)
                    _ = try FileManager.default.replaceItemAt(to, withItemAt: from)
                    try? FileManager.default.removeItem(at: from)
                }
                outcome.accepted = true
                outcome.reason = "kept"
                outcome.scorer = after
            } else {
                for ext in [".json", ".bin"] { try? FileManager.default.removeItem(atPath: candidateBase + ext) }
                outcome.reason = !wordsOK ? "common words got worse" : generalOK ? "personal input got worse" : "general input got worse"
            }
        } catch {
            outcome.reason = "error: \(error)"
        }
        outcome.seconds = Date().timeIntervalSince(started)
        return outcome
    }
}

/// A small seeded generator (SplitMix64) for reproducible sessions.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
