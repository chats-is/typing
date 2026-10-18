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
        /// The person's own inputs typed alone: how many the tuned model learned the wrong way round
        /// (it prefers a reading never typed with those keys that the shipped model scored lower).
        public var typedReversed = 0, typedCount = 0
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
    /// Allowed share (percent) of the person's own lone inputs the tuned model may learn the wrong
    /// way round.
    public var typedTolerance = 1.0
    /// How many of the most recent distinct inputs the typed check looks at.
    public var typedMaximum = 300
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
    /// The fingerprint of the shipped model (written beside kept tuned tensors, so that tensors
    /// learned on another model are never applied to this one).
    public var shippedFingerprint: String?
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
            // The context the engine itself reads when ranking (its last characters, decoration out).
            let usable = LexiconDecoder.usableContext(c.context)
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

    /// The tuned and shipped log probabilities of each reading of a lone word (no context).
    private func loneScores(_ readings: [(text: String, score: Double, penalty: Double)], tuned: SentenceScorer, shipped: SentenceScorer) -> (tuned: [Double], shipped: [Double])? {
        var a: [Double] = [], b: [Double] = []
        var start = 0
        while start < readings.count {
            let batch = readings[start ..< min(start + tuned.batchSize, readings.count)].map(\.text)
            guard let x = tuned.score(prefix: [1], candidates: batch), let y = shipped.score(prefix: [1], candidates: batch) else { return nil }
            a += x; b += y
            start += batch.count
        }
        return (a, b)
    }

    /// Exact rate of lone words under the engine's lone-word rule with learning: lexicon score +
    /// contextWeight × the learned lift (words the person has written only); `tuned == shipped`
    /// gives the lexicon order.
    func loneExactRate(_ cases: [Case], tuned: SentenceScorer, shipped: SentenceScorer) -> Double {
        guard !cases.isEmpty else { return 0 }
        let learned = store.learnedStrings()
        var exact = 0
        for c in cases where !c.readings.isEmpty {
            waitWhilePaused()
            guard let (a, b) = loneScores(c.readings, tuned: tuned, shipped: shipped) else { continue }
            let lift = LexiconDecoder.learnedLift(tuned: a, base: b)
            let final = c.readings.indices.map { c.readings[$0].score + (learned.contains(c.readings[$0].text) ? Self.contextWeight * lift[$0] : 0) }
            if let best = final.indices.max(by: { final[$0] < final[$1] }), c.readings[best].text == c.expected { exact += 1 }
        }
        return 100 * Double(exact) / Double(cases.count)
    }

    /// One of the person's inputs, typed alone, whose text the lexicon puts first.
    struct TypedCase { let text: String; let typedWithKeys: Set<String>; let readings: [(text: String, score: Double, penalty: Double)] }

    /// The most recent distinct inputs with keys whose text is the lexicon's first reading of those
    /// keys alone.
    func typedCases(_ entries: [LearningStore.Entry]) -> [TypedCase] {
        var typed: [String: Set<String>] = [:]
        for entry in entries where !entry.keys.isEmpty { typed[entry.keys, default: []].insert(entry.text) }
        var seen = Set<String>(), cases: [TypedCase] = []
        for entry in entries.reversed() where !entry.keys.isEmpty && seen.insert(entry.keys + "\u{1F}" + entry.text).inserted {
            if cases.count >= typedMaximum { break }
            waitWhilePaused()
            let readings = readings(entry.keys, "")
            guard let first = readings.max(by: { $0.score < $1.score }), first.text == entry.text else { continue }
            cases.append(TypedCase(text: entry.text, typedWithKeys: typed[entry.keys] ?? [], readings: readings))
        }
        return cases
    }

    /// How many of `cases` the tuned model itself learned the wrong way round: the shipped model
    /// scores the person's word above a reading never typed with those keys, the tuned one below it.
    /// This judges the model, not the ranking rule (which never moves a word down).
    func typedReversed(_ cases: [TypedCase], tuned: SentenceScorer, shipped: SentenceScorer) -> Int {
        var reversed = 0
        for c in cases {
            waitWhilePaused()
            guard let (a, b) = loneScores(c.readings, tuned: tuned, shipped: shipped), let t = c.readings.firstIndex(where: { $0.text == c.text }) else { continue }
            let flipped = c.readings.indices.contains { u in
                u != t && !c.typedWithKeys.contains(c.readings[u].text) && b[t] >= b[u] && a[t] < a[u]
            }
            if flipped { reversed += 1 }
        }
        return reversed
    }

    /// What a session trains on: the sentences the inputs were typed in (the text before each on its
    /// line, decoration out, then the input), one per time it was typed in it, for the times whose
    /// context is kept; a sentence that only starts a longer one is left out. Then each input on its
    /// own, once, unless it already stands as a sentence. A sentence said again is learned again; a
    /// lone word is never repeated (measured 2026-10-18 on the Mac mini: repeating short inputs by
    /// how often they were typed gathered the model's probability on a few frequent characters, with
    /// more readings learned the wrong way round, general text worse and nothing gained in context,
    /// while repeated whole sentences, as in the forum2 demo, help).
    static func material(_ inputs: [LearningStore.Entry]) -> [String] {
        var said: [(time: Double, sentence: String)] = []
        for input in inputs {
            let kept = input.contexts.count
            for (time, context) in zip(input.times.suffix(kept), input.contexts.suffix(input.times.count)) {
                said.append((time, sentenceBefore(context) + input.text))
            }
        }
        said.sort { $0.time < $1.time }
        // In sorted order a string that starts a longer one is followed by one that starts with it.
        let sorted = Set(said.map(\.sentence)).sorted()
        var starts = Set<String>()
        for (s, next) in zip(sorted, sorted.dropFirst()) where next.hasPrefix(s) { starts.insert(s) }
        var material = said.map(\.sentence).filter { !starts.contains($0) }
        let standing = Set(material)
        var alone = Set<String>()
        for input in inputs where !standing.contains(input.text) && alone.insert(input.text).inserted { material.append(input.text) }
        return material
    }

    /// The text before the cursor on its own line, without box drawing, prompt marks and the
    /// spaces around them.
    static func sentenceBefore(_ context: String) -> String {
        let line = context.split(separator: "\n", omittingEmptySubsequences: false).last.map(String.init) ?? ""
        var text = String(String.UnicodeScalarView(line.unicodeScalars.filter { !LexiconDecoder.isDecoration($0) })).trimmingCharacters(in: .whitespaces)
        while let first = text.first, ">$%#".contains(first) { text = String(text.dropFirst()).trimmingCharacters(in: .whitespaces) }
        return text
    }

    /// Runs one session; blocking, meant for a background thread.
    public func runSession(log: ((String) -> Void)? = nil) -> Outcome {
        var outcome = Outcome()
        let started = Date()
        var state = store.state()
        // Each input once, with the times it was typed and the sentences it was typed in.
        let inputs = store.entries()
        // Only a session that trained is booked (a cancelled one writes nothing: the store may
        // just have been reset); one that found too little text is simply tried again later.
        var trained = false
        defer {
            if trained && !cancelled {
                state.lastSessionTime = Date().timeIntervalSince1970
                state.entriesAtLastSession = inputs.count  // distinct inputs, as `due` counts them
                state.sessions += 1
                if outcome.accepted { state.accepted += 1 }
                state.lastPersonalBefore = outcome.personalBefore; state.lastPersonalAfter = outcome.personalAfter
                state.lastGeneralBefore = outcome.generalBefore; state.lastGeneralAfter = outcome.generalAfter
                state.lastDecision = outcome.reason
                store.save(state)
            }
        }
        guard inputs.count >= minimumNewEntries else { outcome.reason = "too little text"; return outcome }

        // Held out: the most recent distinct inputs, each once as last typed, judged with their real
        // keys and context; none of their occurrences is trained on, so the check never sees an
        // answer it was taught.
        let heldOutCount = min(heldOutMaximum, max(heldOutMinimum, Int(Double(inputs.count) * heldOutFraction)))
        var heldOut = Array(inputs.reversed().filter { !$0.keys.isEmpty }.prefix(heldOutCount).reversed())
        let heldOutInputs = Set(heldOut.map(\.identity))
        var training = inputs.filter { !heldOutInputs.contains($0.identity) }
        if training.isEmpty { training = inputs; heldOut = [] }
        let text = Self.material(training).joined(separator: "\n")
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
            let typed = typedCases(inputs)
            outcome.typedCount = typed.count
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
            let reference = try shipped ?? GraphScorer(weights: shippedWeights, vocabularyURL: vocabularyURL)
            outcome.wordsAfter = loneExactRate(words, tuned: after, shipped: reference)
            outcome.typedReversed = typedReversed(typed, tuned: after, shipped: reference)
            log?("after: personal \(outcome.personalAfter) general \(outcome.generalAfter) words \(outcome.wordsAfter) typed reversed \(outcome.typedReversed)/\(outcome.typedCount)")

            let typedOK = typed.isEmpty || 100 * Double(outcome.typedReversed) / Double(typed.count) <= typedTolerance
            let wordsOK = words.isEmpty || outcome.wordsAfter >= outcome.wordsBefore - wordsTolerance
            let generalOK = (general.isEmpty || outcome.generalAfter >= outcome.generalBefore - generalTolerance) && wordsOK
            let personalOK = personalCases.count < heldOutMinimum || outcome.personalAfter >= outcome.personalBefore - personalTolerance
            if cancelled { outcome.reason = "cancelled"; return outcome }
            if generalOK && personalOK && typedOK {
                // The .bin goes first: a .json without its .bin would be taken for tuned weights.
                for ext in [".bin", ".json"] {
                    let from = URL(fileURLWithPath: candidateBase + ext), to = URL(fileURLWithPath: store.tunedBase + ext)
                    _ = try FileManager.default.replaceItemAt(to, withItemAt: from)
                    try? FileManager.default.removeItem(at: from)
                }
                // Last: tensors without it are taken for another model's and relearned.
                if let shippedFingerprint { try shippedFingerprint.write(to: store.tunedModelURL, atomically: true, encoding: .utf8) }
                else { try? FileManager.default.removeItem(at: store.tunedModelURL) }
                outcome.accepted = true
                outcome.reason = "kept"
                outcome.scorer = after
            } else {
                for ext in [".json", ".bin"] { try? FileManager.default.removeItem(atPath: candidateBase + ext) }
                outcome.reason = !typedOK ? "typed words learned the wrong way round" : !wordsOK ? "common words got worse" : !generalOK ? "general input got worse" : "personal input got worse"
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
