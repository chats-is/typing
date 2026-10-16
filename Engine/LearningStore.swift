import Foundation

/// What the person typed and confirmed, kept on this machine only, as the material the model
/// learns from: one line of JSON per committed composition (the keys typed, the text committed,
/// the text before it). Nothing leaves the Mac; the user can turn learning off and wipe this.
public final class LearningStore {
    public struct Entry: Codable, Equatable {
        public var time: Double
        /// The pinyin letters as typed (no separators).
        public var keys: String
        /// The text committed for them.
        public var text: String
        /// Up to 60 characters in front of the cursor at the time.
        public var context: String
        public init(time: Double, keys: String, text: String, context: String) {
            self.time = time; self.keys = keys; self.text = text; self.context = context
        }
    }

    public let directory: URL
    public var entriesURL: URL { directory.appendingPathComponent("entries.jsonl") }
    /// The tuned tensors (<base>.json + <base>.bin) and a note of the last learning session.
    public var tunedBase: String { directory.appendingPathComponent("tuned").path }
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
    private var exclusionsCache: Set<String>?
    public private(set) var count = 0

    /// Default location: Application Support/Typing/learning (inside the sandbox container when sandboxed).
    public static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSHomeDirectory())
        return base.appendingPathComponent("Typing/learning", isDirectory: true)
    }

    public init(directory: URL = LearningStore.defaultDirectory()) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        count = (try? String(contentsOf: entriesURL, encoding: .utf8))?.split(separator: "\n").count ?? 0
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
        if let data = line.data(using: .utf8), (try? handle?.write(contentsOf: data)) != nil { count += 1; learnedCache = nil }
    }

    public func entries() -> [Entry] {
        lock.lock(); defer { lock.unlock() }
        flushPendingLockedIfDue()
        return entriesLocked()
    }

    private func entriesLocked() -> [Entry] {
        guard let text = try? String(contentsOf: entriesURL, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(separator: "\n").compactMap { decoder.decode(Entry.self, fromLine: String($0)) }
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
        for name in ["entries.jsonl", "tuned.json", "tuned.bin", "tuned.new.json", "tuned.new.bin", "state.json", "exclusions.json"] {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
        count = 0
        exclusionsCache = []
    }

    public var hasTunedWeights: Bool { FileManager.default.fileExists(atPath: tunedBase + ".json") }

    /// Drops tuned tensors that were learned on another model (the shipped model changed); the
    /// entries stay and the next session relearns them on the new model.
    public func discardTunedWeightsUnlessMatching(_ shipped: GraphScorer.Weights) {
        guard hasTunedWeights else { return }
        if let tuned = try? GraphScorer.Weights(path: tunedBase), shipped.matches(tuned: tuned) { return }
        for ext in [".json", ".bin"] { try? FileManager.default.removeItem(atPath: tunedBase + ext) }
        var s = state(); s.lastDecision = "model changed, relearning"; save(s)
    }

    // MARK: - Corrections by the person

    public var exclusionsURL: URL { directory.appendingPathComponent("exclusions.json") }

    /// Words whose order the person told learning to leave alone (they keep the general order).
    public func exclusions() -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        if let exclusionsCache { return exclusionsCache }
        let set = (try? JSONDecoder().decode(Set<String>.self, from: Data(contentsOf: exclusionsURL))) ?? []
        exclusionsCache = set
        return set
    }

    public func exclude(_ word: String) { var set = exclusions(); set.insert(word); saveExclusions(set) }
    public func include(_ word: String) { var set = exclusions(); set.remove(word); saveExclusions(set) }

    private func saveExclusions(_ set: Set<String>) {
        lock.lock(); defer { lock.unlock() }
        exclusionsCache = set
        if let data = try? JSONEncoder().encode(set) { try? data.write(to: exclusionsURL, options: .atomic) }
    }

    /// Cheap enough for the ranking of every lone-word candidate (the set is cached).
    public func isExcluded(_ word: String) -> Bool { exclusions().contains(word) }

    /// Deletes recorded entries (by time stamp), so the next session learns without them.
    public func removeEntries(times: Set<Double>) {
        lock.lock(); defer { lock.unlock() }
        flushPendingLocked()
        let kept = entriesLocked().filter { !times.contains($0.time) }
        handle = nil
        let encoder = JSONEncoder()
        let text = kept.compactMap { try? encoder.encode($0) }.compactMap { String(data: $0, encoding: .utf8) }.joined(separator: "\n")
        try? (text.isEmpty ? "" : text + "\n").write(to: entriesURL, atomically: true, encoding: .utf8)
        count = kept.count
        learnedCache = nil
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
