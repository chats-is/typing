import Foundation
import SQLite3

/// Read-only pinyin lexicon backed by the SQLite file produced by Tools/build_lexicon.py.
/// Keys are syllables joined by "'" (e.g. "zhong'guo"). Queries from any thread are serialised
/// by `queryLock`.
public final class Lexicon {
    public struct Entry {
        public let word: String
        /// Id shared by every reading of the word; indexes the bigram model.
        public let wid: UInt32
        public let key: String
        public let logp: Double
    }

    public enum LexiconError: Error {
        case cannotOpen(String)
        case badFile(String)
    }

    /// Every valid full syllable.
    public let syllables: Set<String>
    /// Every proper prefix of a valid syllable ("zh", "zho", ...).
    public let syllablePrefixes: Set<String>

    private var db: OpaquePointer?
    private var exactStmt: OpaquePointer?
    private var prefixStmt: OpaquePointer?
    private var partialStmt: OpaquePointer?
    private var abbreviationStmt: OpaquePointer?
    private var leadingStmt: OpaquePointer?
    private var englishStmt: OpaquePointer?

    private var exactCache: [String: [Entry]] = [:]
    private var prefixCache: [String: Bool] = [:]
    private var partialCache: [String: [Entry]] = [:]
    private var abbreviationCache: [String: [Entry]] = [:]
    private var englishCache: [String: [Entry]] = [:]
    private static let cacheLimit = 20_000
    /// SQLite statements and caches are shared; the deferred rerank and learning query from other threads.
    private let queryLock = NSRecursiveLock()
    /// Words returned per abbreviation pattern (set before opening the lexicon; the statements embed it).
    public static var abbreviationLimit = 30

    static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    public init(path: String) throws {
        // immutable=1 keeps SQLite from creating journal files next to the bundled database.
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "?#"))
        let uri = "file:" + (path.addingPercentEncoding(withAllowedCharacters: allowed) ?? path) + "?immutable=1"
        guard sqlite3_open_v2(uri, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nil) == SQLITE_OK else {
            let message = db.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            sqlite3_close(db)
            throw LexiconError.cannotOpen(message)
        }

        var all = Set<String>()
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT s FROM syllables", -1, &stmt, nil) == SQLITE_OK else {
            let message = String(cString: sqlite3_errmsg(db))
            sqlite3_close(db)
            throw LexiconError.cannotOpen(message)
        }
        while sqlite3_step(stmt) == SQLITE_ROW {
            all.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        sqlite3_finalize(stmt)
        syllables = all
        var prefixes = Set<String>()
        for s in all {
            for length in 1..<s.count {
                prefixes.insert(String(s.prefix(length)))
            }
        }
        syllablePrefixes = prefixes

        let columns = "SELECT word, wid, key, logp FROM lexicon"
        // The statements every decode needs must prepare; a lexicon file without them would
        // otherwise give no candidates at all without saying why.
        for (sql, target) in [
            ("\(columns) WHERE key = ?1 ORDER BY logp DESC LIMIT 60", \Lexicon.exactStmt),
            ("SELECT 1 FROM lexicon WHERE key >= ?1 AND key < ?2 LIMIT 1", \Lexicon.prefixStmt),
            ("\(columns) WHERE n = ?1 AND key >= ?2 AND key < ?3 ORDER BY logp DESC LIMIT 30", \Lexicon.partialStmt),
            ("\(columns) INDEXED BY idx_abbr WHERE abbr = ?1 AND n >= 2 AND n = ?2 AND key GLOB ?3 ORDER BY logp DESC LIMIT \(Lexicon.abbreviationLimit)", \Lexicon.abbreviationStmt),
        ] as [(String, ReferenceWritableKeyPath<Lexicon, OpaquePointer?>)] {
            guard sqlite3_prepare_v2(db, sql, -1, &self[keyPath: target], nil) == SQLITE_OK else {
                let message = String(cString: sqlite3_errmsg(db))
                sqlite3_close(db)
                throw LexiconError.badFile("cannot prepare: \(message)")
            }
        }
        sqlite3_prepare_v2(db, "\(columns) INDEXED BY idx_n_key WHERE n = ?1 AND key >= ?2 AND key < ?3 AND key GLOB ?4 ORDER BY logp DESC LIMIT \(Lexicon.abbreviationLimit)", -1, &leadingStmt, nil)
        sqlite3_prepare_v2(db, "SELECT word, 0, key, logp FROM english WHERE key = ?1 ORDER BY logp DESC LIMIT 8", -1, &englishStmt, nil)
    }

    deinit {
        sqlite3_finalize(exactStmt)
        sqlite3_finalize(prefixStmt)
        sqlite3_finalize(partialStmt)
        sqlite3_finalize(abbreviationStmt)
        sqlite3_finalize(leadingStmt)
        sqlite3_finalize(englishStmt)
        sqlite3_close(db)
    }

    /// Words whose pinyin is exactly `key`, best first.
    public func exact(_ key: String) -> [Entry] {
        queryLock.lock(); defer { queryLock.unlock() }
        if let cached = exactCache[key] { return cached }
        sqlite3_bind_text(exactStmt, 1, key, -1, Self.transient)
        let result = collect(exactStmt)
        store(result, for: key, in: &exactCache)
        return result
    }

    /// Whether any word's pinyin is `key` or starts with `key` followed by more syllables.
    public func hasPrefix(_ key: String) -> Bool {
        queryLock.lock(); defer { queryLock.unlock() }
        if let cached = prefixCache[key] { return cached }
        // "'" is the only character below "(" that appears in keys.
        sqlite3_bind_text(prefixStmt, 1, key, -1, Self.transient)
        sqlite3_bind_text(prefixStmt, 2, key + "(", -1, Self.transient)
        let found = sqlite3_step(prefixStmt) == SQLITE_ROW
        sqlite3_reset(prefixStmt)
        store(found, for: key, in: &prefixCache)
        return found
    }

    /// Words of exactly `syllableCount` syllables whose pinyin starts with `keyPrefix`,
    /// where the last syllable of `keyPrefix` may be incomplete. Best first.
    public func partial(syllableCount: Int, keyPrefix: String) -> [Entry] {
        queryLock.lock(); defer { queryLock.unlock() }
        let cacheKey = "\(syllableCount)|\(keyPrefix)"
        if let cached = partialCache[cacheKey] { return cached }
        sqlite3_bind_int(partialStmt, 1, Int32(syllableCount))
        sqlite3_bind_text(partialStmt, 2, keyPrefix, -1, Self.transient)
        sqlite3_bind_text(partialStmt, 3, keyPrefix + "{", -1, Self.transient)
        let result = collect(partialStmt)
        store(result, for: cacheKey, in: &partialCache)
        return result
    }

    /// Words typed with some syllables cut short to their first letters ("shu'r'f" -> 输入法,
    /// "z'g" -> 中国): one segment per syllable, complete or just the letters it starts with. Best first.
    public func abbreviated(_ segments: [(text: String, complete: Bool)]) -> [Entry] {
        queryLock.lock(); defer { queryLock.unlock() }
        let glob = segments.map { $0.complete ? $0.text : $0.text + "*" }.joined(separator: "'")
        if let cached = abbreviationCache[glob] { return cached }
        let result: [Entry]
        if segments[0].complete, let firstCut = segments.firstIndex(where: { !$0.complete }) {
            // The syllables typed in full narrow the search to a small key range.
            let keyPrefix = segments[...firstCut].map(\.text).joined(separator: "'")
            sqlite3_bind_int(leadingStmt, 1, Int32(segments.count))
            sqlite3_bind_text(leadingStmt, 2, keyPrefix, -1, Self.transient)
            sqlite3_bind_text(leadingStmt, 3, keyPrefix + "{", -1, Self.transient)
            sqlite3_bind_text(leadingStmt, 4, glob, -1, Self.transient)
            result = collect(leadingStmt)
        } else {
            sqlite3_bind_text(abbreviationStmt, 1, String(segments.compactMap(\.text.first)), -1, Self.transient)
            sqlite3_bind_int(abbreviationStmt, 2, Int32(segments.count))
            sqlite3_bind_text(abbreviationStmt, 3, glob, -1, Self.transient)
            result = collect(abbreviationStmt)
        }
        store(result, for: glob, in: &abbreviationCache)
        return result
    }


    /// English words typed as `letters` (lowercase), written as they usually are, best first.
    public func english(_ letters: String) -> [Entry] {
        queryLock.lock(); defer { queryLock.unlock() }
        guard englishStmt != nil else { return [] }
        if let cached = englishCache[letters] { return cached }
        sqlite3_bind_text(englishStmt, 1, letters, -1, Self.transient)
        let result = collect(englishStmt)
        store(result, for: letters, in: &englishCache)
        return result
    }


    private func collect(_ stmt: OpaquePointer?) -> [Entry] {
        var result: [Entry] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            result.append(Entry(
                word: String(cString: sqlite3_column_text(stmt, 0)),
                wid: UInt32(sqlite3_column_int64(stmt, 1)),
                key: String(cString: sqlite3_column_text(stmt, 2)),
                logp: sqlite3_column_double(stmt, 3)
            ))
        }
        sqlite3_reset(stmt)
        return result
    }

    private func store<V>(_ value: V, for key: String, in cache: inout [String: V]) {
        if cache.count >= Self.cacheLimit { cache.removeAll(keepingCapacity: true) }
        cache[key] = value
    }
}
