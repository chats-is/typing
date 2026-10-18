import Foundation

/// What the person typed and confirmed, kept on this machine only, as the material the model
/// learns from: one line of JSON per committed composition (the keys typed, the text committed,
/// the text before it). Nothing leaves the Mac; the user can turn learning off and wipe this.
public final class LearningStore {
    public struct Entry: Codable, Equatable {
        /// When it was last typed.
        public var time: Double
        /// The pinyin letters as typed (no separators).
        public var keys: String
        /// The text committed for them.
        public var text: String
        /// Up to 60 characters in front of the cursor at the time (the latest one).
        public var context: String
        /// How many times these keys produced this text. The same input typed again is counted
        /// here, not stored again, so the material lists each input once.
        public var count: Int
        /// When it was typed, oldest first (the most recent `keptTimes` of them): the learner lays the
        /// material out along this timeline, so repeats stay spread out as they were typed.
        public var times: [Double]
        public static let keptTimes = 64
        /// The text in front of the cursor each of the most recent `keptContexts` times it was typed,
        /// oldest first (the last is `context`): the sentences it was typed in, which the material is
        /// built from. Kept apart from `context` so that an input typed often keeps more than its
        /// last sentence.
        public var contexts: [String]
        public static let keptContexts = 16
        public init(time: Double, keys: String, text: String, context: String, count: Int = 1, times: [Double]? = nil, contexts: [String]? = nil) {
            self.time = time; self.keys = keys; self.text = text; self.context = context; self.count = count
            self.times = times ?? [time]
            self.contexts = contexts ?? [context]
        }
        enum CodingKeys: String, CodingKey { case time, keys, text, context, count, times, contexts }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            time = try c.decode(Double.self, forKey: .time)
            keys = try c.decode(String.self, forKey: .keys)
            text = try c.decode(String.self, forKey: .text)
            context = try c.decode(String.self, forKey: .context)
            count = try c.decodeIfPresent(Int.self, forKey: .count) ?? 1
            times = try c.decodeIfPresent([Double].self, forKey: .times) ?? [time]
            contexts = try c.decodeIfPresent([String].self, forKey: .contexts) ?? [context]
        }
        /// What identifies one input: the keys and the text they produced.
        var identity: String { keys + "\u{1F}" + text }
    }

    public let directory: URL
    public var entriesURL: URL { directory.appendingPathComponent("entries.jsonl") }
    /// The tuned tensors (<base>.json + <base>.bin) and a note of the last learning session.
    public var tunedBase: String { directory.appendingPathComponent("tuned").path }
    /// The fingerprint of the shipped model the tuned tensors were learned on.
    public var tunedModelURL: URL { directory.appendingPathComponent("tuned.model") }
    public var stateURL: URL { directory.appendingPathComponent("state.json") }

    public struct State: Codable {
        public var lastSessionTime: Double = 0
        public var entriesAtLastSession: Int = 0
        public var sessions: Int = 0
        public var accepted: Int = 0
        public var lastPersonalBefore: Double = 0, lastPersonalAfter: Double = 0
        public var lastGeneralBefore: Double = 0, lastGeneralAfter: Double = 0
        public var lastDecision: String = ""
        public init() {}
    }

    private let lock = NSLock()
    private var handle: FileHandle?
    private var pending: Entry?
    private var pendingDeadline: Date = .distantPast
    /// Characters deleted since the held-back entry was committed (deletes come one key at a time).
    private var pendingDeleted = 0
    /// Distinct inputs on file (the number the learner and the settings count by).
    public private(set) var count = 0
    /// Identities of the inputs on file, so a repeat is recognised without rereading the file.
    private var identities = Set<String>()
    /// Lines in the file; repeats are appended as lines and folded in when the file is read, and
    /// the file is rewritten folded once the repeats pile up.
    private var rawLines = 0
    private static let compactAfterRepeats = 200
    private var heldKeysCache: Set<String>?
    /// Past this many distinct inputs the least typed, oldest ones are dropped (down to `pruneTo`);
    /// the most recent `keepRecent` are never dropped, new words being exactly what is being learned.
    public static let maxEntries = 5000
    static let pruneTo = 4500
    static let keepRecent = 500

    /// Default location: Application Support/Typing/learning (inside the sandbox container when sandboxed).
    public static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSHomeDirectory())
        return base.appendingPathComponent("Typing/learning", isDirectory: true)
    }

    public init(directory: URL = LearningStore.defaultDirectory()) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let merged = entriesLocked()
        count = merged.count
        identities = Set(merged.map(\.identity))
        if count > Self.maxEntries { rewriteLocked(Self.pruned(merged)) }
        else if rawLines - count > Self.compactAfterRepeats { rewriteLocked(merged) }
        // A file written before inputs were counted once: the session bookkeeping counted lines.
        var state = state()
        if state.entriesAtLastSession > count { state.entriesAtLastSession = count; save(state) }
    }

    /// Records a commit. It is held back for `undoWindow` seconds first: text deleted again right
    /// after being committed was not what the person wanted and is not learned.
    public func record(keys: String, text: String, context: String, undoWindow: TimeInterval = 3) {
        guard !text.isEmpty, text.contains(where: { ("\u{4E00}"..."\u{9FFF}").contains($0) }) else { return }
        lock.lock(); defer { lock.unlock() }
        flushPendingLocked()
        pending = Entry(time: Date().timeIntervalSince1970, keys: keys, text: text, context: String(context.suffix(60)))
        pendingDeadline = Date(timeIntervalSinceNow: undoWindow)
        pendingDeleted = 0
    }

    /// The person deleted `characters` characters right after a commit: drop the held-back entry
    /// when most of it is gone.
    public func deleted(characters: Int) {
        lock.lock(); defer { lock.unlock() }
        guard let entry = pending, Date() < pendingDeadline else { flushPendingLocked(); return }
        pendingDeleted += characters
        if pendingDeleted * 2 >= entry.text.count { pending = nil }
    }

    /// Writes a held-back entry whose undo window has passed.
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        if Date() >= pendingDeadline { flushPendingLocked() }
    }

    private func flushPendingLocked() {
        guard let entry = pending else { return }
        pending = nil
        guard let data = try? JSONEncoder().encode(entry), var line = String(data: data, encoding: .utf8) else { return }
        line += "\n"
        if handle == nil {
            if !FileManager.default.fileExists(atPath: entriesURL.path) { FileManager.default.createFile(atPath: entriesURL.path, contents: nil) }
            handle = try? FileHandle(forWritingTo: entriesURL)
            _ = try? handle?.seekToEnd()
        }
        if let data = line.data(using: .utf8), (try? handle?.write(contentsOf: data)) != nil {
            rawLines += 1
            if identities.insert(entry.identity).inserted { count += 1 }
            learnedCache = nil
            if count > Self.maxEntries {
                rewriteLocked(Self.pruned(entriesLocked()))
            } else if rawLines - count > Self.compactAfterRepeats {
                rewriteLocked(entriesLocked())
            }
        }
    }

    /// Drops the least typed, then oldest, inputs until `pruneTo` remain; the newest `keepRecent`
    /// (by last typing) stay regardless. Entries come and go back in time order.
    static func pruned(_ entries: [Entry]) -> [Entry] {
        guard entries.count > pruneTo else { return entries }
        let recent = Set(entries.suffix(keepRecent).map(\.identity))
        var candidates = entries.filter { !recent.contains($0.identity) }
        candidates.sort { $0.count != $1.count ? $0.count < $1.count : $0.time < $1.time }
        let dropped = Set(candidates.prefix(entries.count - pruneTo).map(\.identity))
        return entries.filter { !dropped.contains($0.identity) }
    }

    public func entries() -> [Entry] {
        lock.lock(); defer { lock.unlock() }
        flushPendingLockedIfDue()
        return entriesLocked()
    }

    private func entriesLocked() -> [Entry] {
        guard let text = try? String(contentsOf: entriesURL, encoding: .utf8) else { rawLines = 0; return [] }
        let decoder = JSONDecoder()
        var merged: [String: Entry] = [:]
        var lines = 0
        for line in text.split(separator: "\n") {
            lines += 1
            guard let entry = decoder.decode(Entry.self, fromLine: String(line)) else { continue }
            if var seen = merged[entry.identity] {
                seen.count += entry.count
                if entry.time >= seen.time {
                    seen.time = entry.time; seen.context = entry.context
                    seen.contexts = Array((seen.contexts + entry.contexts).suffix(Entry.keptContexts))
                } else {
                    seen.contexts = Array((entry.contexts + seen.contexts).suffix(Entry.keptContexts))
                }
                seen.times = Array((seen.times + entry.times).sorted().suffix(Entry.keptTimes))
                merged[entry.identity] = seen
            } else {
                merged[entry.identity] = entry
            }
        }
        rawLines = lines
        return merged.values.sorted { $0.time < $1.time }
    }

    /// Writes the folded entries back as the file.
    private func rewriteLocked(_ entries: [Entry]) {
        handle = nil
        let encoder = JSONEncoder()
        let text = entries.compactMap { try? encoder.encode($0) }.compactMap { String(data: $0, encoding: .utf8) }.joined(separator: "\n")
        try? (text.isEmpty ? "" : text + "\n").write(to: entriesURL, atomically: true, encoding: .utf8)
        rawLines = entries.count
        count = entries.count
        identities = Set(entries.map(\.identity))
        learnedCache = nil
    }

    private func flushPendingLockedIfDue() { if Date() >= pendingDeadline { flushPendingLocked() } }

    public func state() -> State {
        (try? JSONDecoder().decode(State.self, from: Data(contentsOf: stateURL))) ?? State()
    }

    public func save(_ state: State) {
        if let data = try? JSONEncoder().encode(state) { try? data.write(to: stateURL, options: .atomic) }
    }

    /// Forgets everything learned: the entries and the tuned tensors.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        pending = nil
        handle = nil
        for name in ["entries.jsonl", "tuned.json", "tuned.bin", "tuned.model", "tuned.new.json", "tuned.new.bin", "state.json", "exclusions.json"] {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
        count = 0
        rawLines = 0
        identities = []
        heldKeysCache = []
    }

    public var hasTunedWeights: Bool { FileManager.default.fileExists(atPath: tunedBase + ".json") }

    /// Drops tuned tensors unless they were learned on the shipped model with this fingerprint (a
    /// retrained model may have the same shape, and the tuned top blocks of one model do not fit
    /// another). The entries stay, and the next session, due at once, relearns them on this model.
    public func discardTunedWeights(unlessLearnedOn model: String?) {
        guard hasTunedWeights else { return }
        if let model, (try? String(contentsOf: tunedModelURL, encoding: .utf8)) == model { return }
        for ext in [".json", ".bin"] { try? FileManager.default.removeItem(atPath: tunedBase + ext) }
        try? FileManager.default.removeItem(at: tunedModelURL)
        var s = state()
        s.lastDecision = "model changed, relearning"
        s.entriesAtLastSession = 0
        s.lastSessionTime = 0
        save(s)
    }

    // MARK: - Held back until the next learning session

    public var exclusionsURL: URL { directory.appendingPathComponent("exclusions.json") }

    /// Keys whose learned first choice the person deleted. Until a session has learned the remaining
    /// records and been kept, the tuned model in use still ranks them the old way, so typed alone
    /// they keep the general order meanwhile. Cleared when such a session is kept.
    public func heldKeys() -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        if let heldKeysCache { return heldKeysCache }
        let set = (try? JSONDecoder().decode(Set<String>.self, from: Data(contentsOf: exclusionsURL))) ?? []
        heldKeysCache = set
        return set
    }

    public func hold(keys: String) { var set = heldKeys(); set.insert(keys); saveHeldKeys(set) }

    /// A kept session has learned without the deleted records: nothing is held back any more.
    public func clearHeldKeys() { saveHeldKeys([]) }

    private func saveHeldKeys(_ set: Set<String>) {
        lock.lock(); defer { lock.unlock() }
        heldKeysCache = set
        if set.isEmpty { try? FileManager.default.removeItem(at: exclusionsURL) }
        else if let data = try? JSONEncoder().encode(set) { try? data.write(to: exclusionsURL, options: .atomic) }
    }

    /// Cheap enough for every lone-word decode (the set is cached).
    public func isHeld(keys: String) -> Bool { heldKeys().contains(keys) }

    /// Deletes recorded entries (by time stamp), so the next session learns without them.
    public func removeEntries(times: Set<Double>) {
        lock.lock(); defer { lock.unlock() }
        flushPendingLocked()
        rewriteLocked(entriesLocked().filter { !times.contains($0.time) })
    }

    /// Deletes every record of `keys` producing `text`.
    public func removeEntries(keys: String, text: String) {
        lock.lock(); defer { lock.unlock() }
        flushPendingLocked()
        rewriteLocked(entriesLocked().filter { !($0.keys == keys && $0.text == text) })
    }

    // MARK: - What the person has written

    private var learnedCache: Set<String>?
    private var learnedCacheCount = -1
    public static let learnedMaxLength = 4

    /// Every run of 1 to 4 characters inside the texts the person has confirmed: the words whose
    /// lone-word order learning may change. Words the person never wrote keep the general order.
    public func learnedStrings() -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        if let learnedCache, learnedCacheCount == count { return learnedCache }
        var set = Set<String>()
        if let text = try? String(contentsOf: entriesURL, encoding: .utf8) {
            let decoder = JSONDecoder()
            for line in text.split(separator: "\n") {
                guard let entry = decoder.decode(Entry.self, fromLine: String(line)) else { continue }
                let chars = Array(entry.text)
                for i in chars.indices {
                    for n in 1...Self.learnedMaxLength where i + n <= chars.count {
                        set.insert(String(chars[i ..< i + n]))
                    }
                }
            }
        }
        learnedCache = set
        learnedCacheCount = count
        return set
    }

    public func hasLearned(_ word: String) -> Bool { learnedStrings().contains(word) }
}

private extension JSONDecoder {
    func decode<T: Decodable>(_ type: T.Type, fromLine line: String) -> T? {
        guard let data = line.data(using: .utf8) else { return nil }
        return try? decode(type, from: data)
    }
}
