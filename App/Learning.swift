import Cocoa
import Engine

/// Learning on this machine: records confirmed text, runs a learning session when the
/// keyboard has been idle for a while and enough new text has arrived, and switches the engine to
/// the tuned model when a session is kept. Everything stays in the app's own folder; the setting
/// turns it off at once and the reset button forgets everything.
enum Learning {
    static let store = LearningStore()
    private static var learner: Learner?
    private static var idleTimer: Timer?
    private static var running = false
    private static let queue = DispatchQueue(label: "typing.learning", qos: .utility)
    /// Bumped by reset and by switching learning off: a session that started before is not installed.
    private static var generation = 0
    private static let idleSeconds: TimeInterval = 20

    static var enabled: Bool { UserDefaults.standard.object(forKey: "LearningEnabled") as? Bool ?? true }

    /// The shipped weights: read out of the bundled Core ML model (one copy serves both runtimes).
    private static var weightsSource: GraphScorer.WeightsSource? {
        guard let model = Bundle.main.url(forResource: "LanguageModel", withExtension: "mlmodelc"),
              let positions = Bundle.main.url(forResource: "positions", withExtension: "bin"),
              let header = Bundle.main.url(forResource: "weights", withExtension: "json") else { return nil }
        return .compiledModel(model, positions: positions, header: header)
    }

    /// The fingerprint of the shipped model, written by the build (Scripts/build.sh).
    static let modelFingerprint: String? = Bundle.main.url(forResource: "LanguageModel", withExtension: "sha256")
        .flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines)

    /// The tuned model to start with, if learning is on and a kept session exists.
    static func initialScorer() -> SentenceScorer? {
        guard enabled, store.hasTunedWeights, let weightsSource, let vocabulary = Bundle.main.url(forResource: "vocab", withExtension: "json") else { return nil }
        do {
            // Tuned tensors learned on another shipped model are dropped; the entries are relearned.
            store.discardTunedWeights(unlessLearnedOn: modelFingerprint)
            guard store.hasTunedWeights else { return nil }
            return try tunedCoreMLScorer(vocabulary: vocabulary) ?? GraphScorer(source: weightsSource, tunedPath: store.tunedBase, vocabularyURL: vocabulary)
        } catch {
            NSLog("Typing: cannot load the tuned model: \(error)")
            return nil
        }
    }

    /// The tuned model for the Neural Engine: the shipped compiled model with the learned
    /// tensors written in, built once per kept session and loaded like the shipped one. The GPU
    /// runtime is the fallback when it cannot be built.
    private static func tunedCoreMLScorer(vocabulary: URL) -> SentenceScorer? {
        guard let shipped = Bundle.main.url(forResource: "LanguageModel", withExtension: "mlmodelc") else { return nil }
        let destination = store.directory.appendingPathComponent("TunedModel.mlmodelc")
        let stampURL = store.directory.appendingPathComponent("TunedModel.stamp")
        let signature = TunedModel.signature(tunedBase: store.tunedBase, shipped: shipped)
        do {
            if (try? String(contentsOf: stampURL, encoding: .utf8)) != signature || !FileManager.default.fileExists(atPath: destination.path) {
                try TunedModel.build(shipped: shipped, tuned: try GraphScorer.Weights(path: store.tunedBase), destination: destination)
                try signature.write(to: stampURL, atomically: true, encoding: .utf8)
            }
            return try NeuralScorer(modelURL: destination, vocabularyURL: vocabulary)
        } catch {
            NSLog("Typing: tuned Core ML model unavailable, using the GPU runtime: \(error)")
            return nil
        }
    }

    static func recordCommit(keys: String, text: String, context: String) {
        guard enabled else { return }
        store.record(keys: keys, text: text, context: context)
        scheduleIdleCheck()
    }

    static func noteDelete() {
        store.deleted(characters: 1)
    }

    /// Any key: the idle clock restarts, and a running session pauses so that typing never shares
    /// the GPU or the graph lock with it; it goes on when the keyboard is idle again, so a session finishes even
    /// while typing goes on.
    static func noteActivity() {
        if running { learner?.paused = true }
        activityLock.lock(); lastActivity = Date(); activityLock.unlock()
        scheduleIdleCheck()
    }

    private static let activityLock = NSLock()
    private static var lastActivity = Date.distantPast
    /// Background work that needs the decoder waits here while keys are being typed, so no key
    /// ever queues behind it; returns false when `generation` is no longer the latest request.
    private static func yieldToTyping(generation: Int) -> Bool {
        while true {
            activityLock.lock(); let current = changesGeneration; let quiet = Date().timeIntervalSince(lastActivity); activityLock.unlock()
            if generation != current { return false }
            if quiet >= 0.4 { return true }
            Thread.sleep(forTimeInterval: 0.1)
        }
    }
    private static var changesGeneration = 0  // under activityLock
    private static var changesCache: (count: Int, excluded: Int, scorer: ObjectIdentifier?, changes: [(keys: String, learned: String, general: String)])?  // scanQueue only
    private static let scanQueue = DispatchQueue(label: "typing.learning.scan", qos: .userInitiated)

    private static func scheduleIdleCheck() {
        idleTimer?.invalidate()
        guard enabled else { return }
        idleTimer = Timer.scheduledTimer(withTimeInterval: idleSeconds, repeats: false) { _ in
            if running { learner?.paused = false } else { startSessionIfDue() }
        }
    }

    private static func makeLearner() -> Learner? {
        guard let weightsSource, let vocabulary = Bundle.main.url(forResource: "vocab", withExtension: "json"), let decoder = EngineHost.lexiconDecoder else { return nil }
        let general = Bundle.main.url(forResource: "general_lattices", withExtension: "tsv")
        let learner = try? Learner(store: store, weights: weightsSource, vocabularyURL: vocabulary, generalLatticesURL: general) { keys, context in
            // The lattice's own order and typing costs; the model's look is deferred and not taken here.
            let quick = decoder.decode(keys, precedingText: context)
            return quick.sentences.map { (text: $0.text, score: $0.score, penalty: $0.penalty) }
        }
        learner?.generalWordsURL = Bundle.main.url(forResource: "general_words", withExtension: "tsv")
        learner?.shippedFingerprint = modelFingerprint
        // The session compares before and after on the same runtime (GPU); the engine's own
        // scorer serves as "before" only when it is that runtime.
        learner?.engineScorer = { EngineHost.sentenceScorer as? GraphScorer }
        return learner
    }

    /// Whether a session is running now (the settings window shows it).
    static var isRunning: Bool { running }
    /// Called on the main thread when a session ends (the settings window refreshes).
    static var sessionEnded: (() -> Void)?

    /// Runs a session now, whether or not one is due (the button in Settings). Nothing happens while
    /// learning is off or a session is already running.
    static func trainNow() { startSession(force: true) }

    private static func startSessionIfDue() { startSession(force: false) }

    private static func startSession(force: Bool) {
        guard enabled, !running else { return }
        store.flush()
        if learner == nil { learner = makeLearner() }
        guard let learner, force || learner.due() else { return }
        learner.cancelled = false
        learner.paused = false
        running = true
        let started = generation
        queue.async {
            let outcome = learner.runSession { NSLog("Typing learning: \($0)") }
            NSLog("Typing learning: \(outcome.accepted ? "kept" : "discarded") (\(outcome.reason)) steps \(outcome.steps) personal \(outcome.personalBefore)→\(outcome.personalAfter) general \(outcome.generalBefore)→\(outcome.generalAfter) in \(Int(outcome.seconds)) s")
            // A kept model is written into a copy of the Core ML model here, off the main thread
            // (its first load compiles it for the Neural Engine), checked against the model that
            // was validated, and handed to the engine; the GPU model stands in if the copy differs.
            var kept: SentenceScorer? = outcome.scorer
            if outcome.accepted, let graph = outcome.scorer, let vocabulary = Bundle.main.url(forResource: "vocab", withExtension: "json"),
               let coreML = tunedCoreMLScorer(vocabulary: vocabulary) {
                if agrees(coreML, graph) { kept = coreML } else { NSLog("Typing learning: the Core ML copy disagrees with the validated model; using the GPU runtime") }
            }
            DispatchQueue.main.async {
                running = false
                defer { sessionEnded?() }
                guard started == generation else { return }  // reset or switched off meanwhile
                if outcome.accepted {
                    store.clearHeldKeys()  // this model was learned without the deleted records
                    install(tuned: kept)
                }
            }
        }
    }

    /// Whether two runtimes of what should be the same model agree (log-probabilities within
    /// half-precision noise) on a few everyday continuations.
    private static func agrees(_ a: SentenceScorer, _ b: SentenceScorer) -> Bool {
        let probes: [(String, [String])] = [("今天天气", ["很好", "不错", "怎么样"]), ("我觉得这个", ["问题", "方案", "东西"]), ("", ["你好", "我们", "不知道"]), ("明天", ["见", "再说", "有空吗"])]
        var worst = 0.0
        for (context, candidates) in probes {
            let prefix: [Int32] = [LexiconDecoder.bos] + a.contextTokens(context, reserving: 1)
            guard let x = a.score(prefix: prefix, candidates: candidates), let y = b.score(prefix: prefix, candidates: candidates), x.count == y.count else { return false }
            for (p, q) in zip(x, y) { worst = max(worst, abs(p - q)) }
        }
        return worst < 0.2
    }

    /// Which lone words learning may move: those in the learning records and not undone.
    static func learnedWordFilter() -> (String) -> Bool { { store.hasLearned($0) } }
    /// Keys whose learned order was deleted (they keep the general order until a kept session).
    static func heldKeysFilter() -> (String) -> Bool { { store.isHeld(keys: $0) } }

    /// Puts a tuned model (or, with nil, the shipped one) into the engine. With a tuned model the
    /// shipped one stays beside it so lone words move only by what was learned.
    private static func install(tuned: SentenceScorer?) {
        let scorer: SentenceScorer? = tuned ?? EngineHost.scorer
        EngineHost.sentenceScorer = scorer
        EngineHost.lexiconDecoder?.replaceScorer(scorer)
        EngineHost.lexiconDecoder?.setLearning(base: tuned == nil ? nil : EngineHost.scorer, words: tuned == nil ? nil : learnedWordFilter())
    }

    /// Loads the tuned model from disk (or none) off the main thread and installs it; a later
    /// reset or switch cancels an earlier load.
    private static func reloadScorer() {
        generation += 1
        let started = generation
        queue.async {
            let tuned = initialScorer()
            DispatchQueue.main.async { if started == generation { install(tuned: tuned) } }
        }
    }

    static func enabledChanged() {
        if !enabled { learner?.cancelled = true; learner?.paused = false; idleTimer?.invalidate() }
        reloadScorer()
    }

    static func reset() {
        learner?.cancelled = true
        learner?.paused = false
        store.reset()
        for name in ["TunedModel.mlmodelc", "TunedModel.stamp"] { try? FileManager.default.removeItem(at: store.directory.appendingPathComponent(name)) }
        reloadScorer()
    }

    /// What learning changed: recorded inputs whose first choice now differs from the
    /// general order (keys, learned first choice, general first choice).
    /// Each input is decoded twice under the decoder's lock, so this runs on a background queue,
    /// pauses whenever a key arrives, looks only at the most recent inputs, and is reused until the
    /// entries change. A newer request stops an older one.
    static func requestLearnedChanges(_ completion: @escaping ([(keys: String, learned: String, general: String)]) -> Void) {
        activityLock.lock(); changesGeneration += 1; let generation = changesGeneration; activityLock.unlock()
        let scorerID = EngineHost.sentenceScorer.map { ObjectIdentifier($0) }
        let active = EngineHost.lexiconDecoder?.baseScorer != nil
        scanQueue.async {
            let found = learnedChanges(generation: generation, scorerID: scorerID, active: active)
            activityLock.lock(); let latest = changesGeneration == generation; activityLock.unlock()
            if latest { DispatchQueue.main.async { completion(found) } }  // an older, stopped scan says nothing
        }
    }

    private static func learnedChanges(generation: Int, scorerID: ObjectIdentifier?, active: Bool) -> [(keys: String, learned: String, general: String)] {
        guard active, let decoder = EngineHost.lexiconDecoder else { return [] }
        let entries = store.entries()
        // A deletion that found no record still holds its keys back: they are part of the key.
        let excluded = store.heldKeys().count
        if let cache = changesCache, cache.count == entries.count, cache.excluded == excluded, cache.scorer == scorerID { return cache.changes }
        var seen = Set<String>()
        var changes: [(keys: String, learned: String, general: String)] = []
        for entry in entries.reversed() where !entry.keys.isEmpty && seen.insert(entry.keys).inserted {
            guard yieldToTyping(generation: generation) else { return changes }
            let learned = decoder.firstChoice(entry.keys, learned: true)
            let general = decoder.firstChoice(entry.keys, learned: false)
            if learned != general && !learned.isEmpty { changes.append((entry.keys, learned, general)) }
            if changes.count >= 100 || seen.count >= 400 { break }
        }
        changesCache = (entries.count, excluded, scorerID, changes)
        return changes
    }

    /// Deleting a learned first choice deletes its records and holds its keys: until a session has
    /// learned the remaining records and been kept, these keys typed alone keep the general order
    /// (the tuned model in use still ranks them the old way, and the change may come from another
    /// word moving, not from the one shown); that session clears the hold.
    static func undo(keys: String, word: String) {
        store.removeEntries(keys: keys, text: word)
        store.hold(keys: keys)
    }
    static func deleteEntries(times: Set<Double>) { store.removeEntries(times: times) }

    static func statusText() -> String {
        let state = store.state()
        if state.accepted == 0 { return store.count == 0 ? "还没有学习" : "已记录 \(store.count) 条输入" }
        return "已根据你的输入调整 \(state.accepted) 次"
    }
}
