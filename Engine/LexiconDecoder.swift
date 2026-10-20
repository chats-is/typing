import Foundation

/// Turns a pinyin letter buffer into ranked candidates with two parts and a clear division of
/// labour: the lexicon says which words the keys can stand for and, with nothing before the
/// cursor, which of them is usual; the language model says which reading of the whole buffer
/// fits the text before the cursor.
///
/// A word lattice is built over the buffer (full syllables, an unfinished last syllable,
/// syllables cut to their initials, English words), the best readings under the lexicon's
/// frequencies are found, and the language model then adds what the context changes:
/// log P(reading | context) - sum of log P(word | nothing) over its words. For a lone word with no
/// context that difference is zero, so the lexicon's order stands, as the standard asks.
public final class LexiconDecoder: CandidateDecoding {
    private let lexicon: Lexicon
    private let matcher: SyllableMatcher
    private var scorer: SentenceScorer?
    /// Swaps the language model (e.g. for one that has learned on this machine).
    public func replaceScorer(_ newScorer: SentenceScorer?) {
        lock.lock(); defer { lock.unlock() }
        scorer = newScorer
    }

    /// The first candidate for `keys` with no context, with or without what was learned (for the
    /// settings window's list of learned changes).
    public func firstChoice(_ keys: String, learned: Bool) -> String {
        // The decode runs under the lock with the learning filter swapped (the decode takes its own
        // snapshot of the filter); the model's look runs after the lock is released, so typing is
        // not held up behind it.
        lock.lock()
        let saved = learnedWords
        if !learned { learnedWords = { _ in false } }
        let quick = decode(keys, precedingText: "")
        learnedWords = saved
        lock.unlock()
        return quick.refined().candidates.first?.text ?? ""
    }
    /// Sets the learning pair under the lock, so it cannot interleave with `firstChoice`, which
    /// swaps `learnedWords` for the length of a decode.
    public func setLearning(base: SentenceScorer?, words: ((String) -> Bool)?) {
        lock.lock(); defer { lock.unlock() }
        baseScorer = base
        learnedWords = words
    }
    /// The shipped model beside a tuned `scorer`: for a lone word with no context, the order stays
    /// the lexicon's (general usage) except for what this machine has learned: a word the person
    /// actually uses gains its lift (`learnedLift`), so only those words move, and only up.
    public var baseScorer: SentenceScorer?
    /// Which words the person has actually written (learning may move only those as lone words).
    public var learnedWords: ((String) -> Bool)?
    /// English words the person has written, by the letters typed for them: offered as English
    /// readings even when the English table lacks them (at the table's rarest frequency; learning
    /// decides how far they rise).
    public var typedEnglish: ((String) -> [String])?
    /// Keys (letters only) whose learned order the person deleted: typed alone they keep the
    /// general order until the next kept learning session.
    public var heldKeys: ((String) -> Bool)?
    /// Word-pair statistics over everyday text (Tools/build_bigram.py), so the lattice search
    /// itself prefers words that go together; nil falls back to word frequencies alone.
    private let bigram: BigramModel?
    /// Weight of the pair probability against the word's own frequency in the lattice search.
    public var bigramWeight = 0.6 { didSet { bigramWeight = min(max(bigramWeight, 0), 0.999); logOneMinusBigramWeight = log(1 - bigramWeight) } }
    private var logOneMinusBigramWeight = log(1 - 0.6)
    /// The same for the first word of the buffer, against the start-of-run statistics.
    public var startBigramWeight = 0.0 { didSet { logOneMinusStartWeight = log(1 - startBigramWeight) } }
    private var logOneMinusStartWeight = 0.0

    /// Weight of what the context adds against the lexicon's own order.
    public var contextWeight = 2.0
    /// Log-probability penalty for a word matched through an unfinished last syllable: complete
    /// readings come first (hu is 胡, not 会 on the way to hui), as in every input method.
    public var partialPenalty = 9.0
    /// Log-probability penalty for each syllable typed as just its initial.
    public var initialPenalty = 3.0  // fitted 2026-10-14 on sentence sets with real-text answers (english/typo up, abbr/dialogue/short flat)
    /// Log-probability penalty for reading a run of letters as an English word (English words are
    /// listed by their share of English text, which is much rarer than Chinese while typing pinyin).
    public var englishPenalty = 1.0
    /// Log-probability cost of each word in a reading: frequent single characters would otherwise
    /// always outscore the word they spell (推+个 against 退格); a reading in fewer words is a reading
    /// in words the lexicon knows.
    public var wordPenalty = 0.0  // no hand-set cost per word: the statistics and the model decide (2026-10-13)
    /// Weight of the lattice's own statistics (word frequencies and pairs) in the reranked score,
    /// beside contextWeight for the model; typing costs weigh rerankLatticeWeight + contextWeight.
    public var rerankLatticeWeight = 1.0
    /// Whether the model's rerank is returned as the result's `refine` step (the host shows the
    /// lattice's reading first) instead of being done before returning.
    public var deferRerank = true
    /// Chinese before English: among candidates covering the same keys, a Chinese word comes before
    /// any reading that contains an English word (the user's condition).
    public var chineseFirst = true
    /// Multiplier on the typing-slip costs of SyllableMatcher (substitution, missing, stray, transposed letters).
    public var slipScale = 1.0
    /// How much better a typo-corrected reading must score to replace the literal one.
    public var correctionMargin = 8.0
    /// How much better, after the model's say, a reading with a corrected slip must score than
    /// the best literal reading to come first.
    public var extraMargin = 3.0
    /// Whether slips of the finger are looked for at all.
    public var tolerant = true

    private static let maxWordSyllables = 8
    private static let maxCandidates = 300
    /// Every character of a syllable as candidates, by syllable (the buffer is that syllable alone).
    private var allCharactersCache: [String: [Candidate]] = [:]
    private func allCharacters(_ syllable: String, consumed: Int) -> [Candidate] {
        if let cached = allCharactersCache[syllable], cached.first?.consumed == consumed { return cached }
        let list = timedLexicon({ lexicon.characters(syllable) }).map {
            Candidate(text: $0.word, consumed: consumed, words: [WordRef(text: $0.word, wid: $0.wid, key: $0.key)], score: $0.logp)
        }
        if allCharactersCache.count > 512 { allCharactersCache.removeAll() }
        allCharactersCache[syllable] = list
        return list
    }
    /// States kept per lattice node (one per word).
    public var beamWidth = 12  // 24 → 12 measured identical on 1200 sentences, lattice time −35% (2026-10-14)
    /// How many whole-buffer readings the language model compares (--sentences in the CLI).
    public var sentenceCount = 16
    private static let maxSentenceSearchSteps = 1200
    private static let rerankedLeaders = 3
    private static let maxAbbreviatedSyllables = 6
    private static let maxEnglishLength = 20
    private static let retroflexInitials = Set("zcs".utf8)
    private static let apostrophe = UInt8(ascii: "'")
    private static func isCapital(_ byte: UInt8) -> Bool { byte >= UInt8(ascii: "A") && byte <= UInt8(ascii: "Z") }
    private static let vowels = Set("aeiouv".utf8)
    public static let bos: Int32 = 1

    public init(lexicon: Lexicon, scorer: SentenceScorer?, bigram: BigramModel? = nil, fuzzyRules: Set<FuzzyRule> = []) {
        self.lexicon = lexicon
        self.scorer = scorer
        self.bigram = bigram
        matcher = SyllableMatcher(syllables: lexicon.syllables, fuzzyRules: fuzzyRules)
    }

    /// Lead (log-probability) of the lexicon's best reading over the second at which the model is
    /// not consulted; 0 always consults it.
    public var skipMargin = 0.0
    public private(set) var skippedCount = 0
    /// Model invocations and time, for performance measurements.
    public private(set) var decodeCount = 0
    /// The lexicon (SQLite statements), the pair statistics' caches and the lattice caches are not
    /// thread-safe; typing (main thread) and learning (background) all come through here.
    private let lock = NSRecursiveLock()
    public private(set) var languageModelMilliseconds = 0.0
    /// Time spent building the lattice (lexicon queries included) and searching it, for profiling.
    public private(set) var latticeMilliseconds = 0.0
    public private(set) var lexiconMilliseconds = 0.0
    public private(set) var phaseMilliseconds = [0.0, 0.0, 0.0, 0.0, 0.0]  // steps, edges, viterbi, sentences, candidates

    private struct Edge {
        let start: Int
        let end: Int
        let word: WordRef
        /// The lexicon's log-probability of the word.
        let logp: Double
        /// What the language model gives the word with no context.
        let penalty: Double
        /// Buffer offsets where each syllable of the word ends.
        let cuts: [Int]
        /// Matched through an unfinished syllable or an abbreviation.
        let partial: Bool
        let english: Bool
    }

    private struct State {
        let score: Double
        /// Index of the edge that led here, -1 at the origin.
        let edge: Int
        /// Index of the predecessor among the states of the edge's start node.
        let previous: Int
    }

    /// What identifies an edge across two runs over inputs that share a prefix.
    private struct EdgeKey: Hashable {
        let start: Int, end: Int, wid: UInt32, text: String, penalty: UInt64, logp: UInt64, partial: Bool, english: Bool
        init(_ edge: Edge) {
            start = edge.start; end = edge.end; wid = edge.word.wid; text = edge.word.text
            penalty = edge.penalty.bitPattern; logp = edge.logp.bitPattern; partial = edge.partial; english = edge.english
        }
    }
    /// The last run's lattice and Viterbi states, per tolerance: when the next input only appends
    /// letters, every state at a node inside the old input is unchanged and is reused.
    private struct RunCache {
        let chars: [UInt8]
        let edges: [Edge]
        let states: [[State]]
        let origin: Int
    }
    private var runCaches: [Bool: RunCache] = [:]
    /// Whether consecutive runs over extending inputs reuse the previous Viterbi states.
    public var incremental = true

    private struct Sentence {
        var candidate: Candidate
        /// Sum of the typing costs along the path (unfinished syllables, initials, slips, English).
        var penalty = 0.0
        /// Reads some letters as an English word / some syllables by their initials / a slip.
        var english = false
        var abbreviated = false
        var slipped = false
    }
    /// Readings of other kinds kept for the language model beside the best literal ones (--extras).
    public var englishExtras = 0
    public var abbreviationExtras = 0
    public var slipExtras = 0

    public func decode(_ input: String, precedingText: String = "") -> DecodeResult {
        lock.lock()
        defer { lock.unlock() }
        let chars = Array(input.utf8)
        guard !chars.isEmpty else { return .empty }
        decodeCount += 1
        // The buffer is read literally first; a reading that needs a slip of the finger replaces
        // it only when it reads more of the buffer or is clearly the better explanation.
        let strict = run(chars, tolerant: false)
        var chosen = strict
        var sentences = strict.sentences
        // A lone word typed with nothing before it is taken as typed (the same rule that keeps the
        // model out of a lone word): a slip correction may follow it, never displace it.
        let loneLiteral = Self.usableContext(precedingText).isEmpty && strict.covered == chars.count
            && (strict.sentences.first?.candidate.words.count ?? 0) <= 1
        if tolerant && !loneLiteral {
            let loose = run(chars, tolerant: true)
            if loose.covered > strict.covered || (loose.covered == strict.covered && loose.score > strict.score + correctionMargin) {
                chosen = loose
                sentences = loose.sentences
            } else if loose.covered == strict.covered {
                let literal = Set(sentences.map(\.candidate.text))
                var slips = loose.sentences.filter { !literal.contains($0.candidate.text) }.prefix(slipExtras).map { $0 }
                for i in slips.indices { slips[i].slipped = true }
                if !slips.isEmpty { sentences = Array(sentences.prefix(max(1, sentenceCount - slips.count))) + slips }
            }
        }
        // Keys whose learned order was deleted are ranked without what was learned.
        let held = heldKeys?(input.filter(\.isLetter).lowercased()) ?? false
        if deferRerank, scorer != nil, sentences.count > 1 {
            // Show the lattice's reading at once; the model's say arrives as the second look.
            var quick = chinesePriority(chosen.result)
            let quickTexts = quick.candidates.map(\.text)
            let finalSentences = sentences
            // The second look runs off the main thread without the decoder's lock, so it works on
            // a snapshot of the model and the learning pair taken here, under the lock.
            let (model, base, learned) = (scorer, held ? nil : baseScorer, learnedWords)
            quick.refine = { [self] isCancelled in
                if isCancelled() { return nil }
                var better = self.chinesePriority(self.rerank(chosen.result, sentences: finalSentences, precedingText: precedingText, scorer: model, base: base, learned: learned))
                if better.candidates.map(\.text) == quickTexts { return nil }  // nothing to update
                better.refine = nil
                return better
            }
            return quick
        }
        return chinesePriority(rerank(chosen.result, sentences: sentences, precedingText: precedingText, scorer: scorer, base: held ? nil : baseScorer, learned: learnedWords))
    }

    /// Text before the cursor that is worth conditioning on: its last characters, when Chinese.
    /// Box drawing and prompt marks are left out: in a terminal they frame the line being typed
    /// (────, ❯) and are no part of what is being said.
    static func usableContext(_ text: String) -> String {
        let recent = text.suffix(64).unicodeScalars.filter { !isDecoration($0) }
        let tail = String(String(String.UnicodeScalarView(recent)).suffix(24))
        return tail.contains(where: { ("\u{4E00}"..."\u{9FFF}").contains($0) }) ? tail : ""
    }

    /// Box drawing characters and the prompt marks terminals draw.
    static func isDecoration(_ scalar: Unicode.Scalar) -> Bool {
        (0x2500...0x257F).contains(scalar.value) || scalar.value == 0x276F || scalar.value == 0x23FA
    }

    /// Lets the language model choose among the best whole-buffer readings and puts its picks first.
    private func rerank(_ result: DecodeResult, sentences: [Sentence], precedingText: String,
                        scorer: SentenceScorer?, base: SentenceScorer?, learned: ((String) -> Bool)?) -> DecodeResult {
        guard let scorer, sentences.count > 1 else { return result }
        let usable = Self.usableContext(precedingText)
        // A lone word with nothing before it is ordered by the lexicon (the standard: the most
        // usual word as typed); the model decides once there is context or a sentence to read.
        if usable.isEmpty, let best = sentences.first, best.candidate.words.count <= 1 {
            guard let base else { return result }
            return rerankLoneWordByLearning(result, sentences: sentences, tuned: scorer, base: base, learned: learned)
        }
        // When the lexicon's best reading is far ahead of every other, the model is not asked:
        // most keystrokes of a plain sentence, and the model call is most of the time per key.
        if skipMargin > 0, sentences.count > 1, sentences[0].candidate.score - sentences[1].candidate.score > skipMargin {
            skippedCount += 1
            return result
        }
        let prefix = usable.isEmpty ? [Self.bos] : [Self.bos] + scorer.contextTokens(usable, reserving: 1)
        let started = DispatchTime.now().uptimeNanoseconds
        var neural: [Double] = []
        var start = 0
        while start < sentences.count {
            let batch = sentences[start..<min(start + scorer.batchSize, sentences.count)]
            guard let scores = scorer.score(prefix: prefix, candidates: batch.map(\.candidate.text)) else { return result }
            neural += scores
            start += batch.count
        }
        languageModelMilliseconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
        var scored = zip(sentences, neural).map { sentence, lm -> Sentence in
            var scored = sentence
            // P(reading | keys) ∝ P(keys | reading) · P(reading): the typing costs are P(keys | reading)
            // and weigh as much as the whole estimate of P(reading) (lattice statistics + model), so
            // that an unfinished syllable or a slip costs the same whether or not the model reads.
            let statistics = sentence.candidate.score + sentence.penalty
            scored.candidate.score = rerankLatticeWeight * statistics
                + contextWeight * lm
                - (rerankLatticeWeight + contextWeight) * sentence.penalty
            return scored
        }
        // A reading that corrects a slip displaces the best literal reading only when the model
        // finds it clearly better: the model cannot see the keys, and the literal reading is what
        // was typed.
        if let literal = scored.filter({ !$0.slipped }).map(\.candidate.score).max() {
            for i in scored.indices where scored[i].slipped && scored[i].candidate.score < literal + extraMargin {
                scored[i].candidate.score = min(scored[i].candidate.score, literal - 0.001)
            }
        }
        let ranked = scored.map(\.candidate).sorted { $0.score > $1.score }
        var reranked = result
        reranked.sentences = ranked
        let leaders = Array(ranked.prefix(Self.rerankedLeaders))
        reranked.candidates = leaders + result.candidates.filter { candidate in
            !leaders.contains { $0.text == candidate.text && $0.consumed == candidate.consumed }
        }
        return reranked
    }

    /// Moves the best Chinese word in front of English-containing readings of the same length. Only a
    /// reading that is one word of the lexicon counts: single characters strung over the letters of
    /// an English word (咯啊定 for loading) are no Chinese word.
    private func chinesePriority(_ result: DecodeResult) -> DecodeResult {
        guard chineseFirst, let first = result.candidates.first else { return result }
        func isEnglish(_ c: Candidate) -> Bool { c.words.contains { $0.wid == 0 && $0.text.contains { $0.isASCII && $0.isLetter } } }
        guard isEnglish(first) else { return result }
        var candidates = result.candidates
        if let index = candidates.firstIndex(where: { $0.consumed == first.consumed && !isEnglish($0) && $0.words.count == 1 }), index > 0 {
            let chinese = candidates.remove(at: index)
            candidates.insert(chinese, at: 0)
        }
        var sentences = result.sentences
        if let s = sentences.first, isEnglish(s), let index = sentences.firstIndex(where: { !isEnglish($0) && $0.words.count == 1 }), index > 0 {
            let chinese = sentences.remove(at: index)
            sentences.insert(chinese, at: 0)
        }
        var ordered = result
        ordered.candidates = candidates
        ordered.sentences = sentences
        return ordered
    }

    /// Lone word, no context, learning on: lexicon order plus contextWeight × (log P under the
    /// tuned model − log P under the shipped model) for each reading. Untaught words keep their place.
    /// How far learning lifts each reading of a lone word: its change of log probability from the
    /// shipped to the tuned model, less the median change over all the readings, never below zero.
    /// Tuning moves nearly every lone reading a little (mostly down, as probability gathers on what
    /// is typed most); only the person's words are moved, so the shared part of the change is taken
    /// out first, or a word they typed would sink below readings they never typed merely because
    /// those are not moved at all. A word is never moved down.
    static func learnedLift(tuned: [Double], base: [Double]) -> [Double] {
        let change = zip(tuned, base).map { $0 - $1 }
        guard !change.isEmpty else { return [] }
        let median = change.sorted()[change.count / 2]
        return change.map { max(0, $0 - median) }
    }

    private func rerankLoneWordByLearning(_ result: DecodeResult, sentences: [Sentence], tuned: SentenceScorer, base: SentenceScorer, learned: ((String) -> Bool)?) -> DecodeResult {
        let texts = sentences.map(\.candidate.text)
        // Only readings the person has written can move; with none among them the models are not asked.
        let movable = texts.map { learned?($0) ?? true }
        guard movable.contains(true) else { return result }
        var tunedScores: [Double] = [], baseScores: [Double] = []
        var start = 0
        while start < texts.count {
            let batch = Array(texts[start..<min(start + tuned.batchSize, texts.count)])
            guard let a = tuned.score(prefix: [Self.bos], candidates: batch), let b = base.score(prefix: [Self.bos], candidates: batch) else { return result }
            tunedScores += a; baseScores += b
            start += batch.count
        }
        var ranked = sentences.map(\.candidate)
        let lift = Self.learnedLift(tuned: tunedScores, base: baseScores)
        for i in ranked.indices where movable[i] {
            ranked[i].score += contextWeight * lift[i]
        }
        ranked.sort { $0.score > $1.score }
        var reranked = result
        reranked.sentences = ranked
        reranked.candidates = ranked + result.candidates.filter { candidate in
            !ranked.contains { $0.text == candidate.text && $0.consumed == candidate.consumed }
        }
        return reranked
    }

    /// Log-probability of following `previous` (nil = sentence start) with `edge`.
    private func transition(from previous: WordRef?, to edge: Edge) -> Double {
        // At the start the word's own frequency leads (the order the standard asks for a lone
        // word) and the start-of-run statistics only help a sentence along; after a word the pair
        // statistics lead.
        var pair = 0.0
        let weight = previous == nil ? startBigramWeight : bigramWeight
        if let bigram, edge.word.wid != 0, (previous?.wid ?? 1) != 0 {
            pair = bigram.probability(previous?.wid ?? BigramModel.sentenceStart, edge.word.wid)
        }
        let score = pair > 0 ? log(weight * pair + (1 - weight) * exp(edge.logp)) : edge.logp + (previous == nil ? logOneMinusStartWeight : logOneMinusBigramWeight)
        return score - edge.penalty - wordPenalty
    }

    private func timedLexicon(_ query: () -> [Lexicon.Entry]) -> [Lexicon.Entry] {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { lexiconMilliseconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6 }
        return query()
    }

    private struct Lattice {
        let edges: [Edge]
        let origin: Int
        let target: Int
    }

    private func run(_ chars: [UInt8], tolerant: Bool) -> (result: DecodeResult, sentences: [Sentence], covered: Int, score: Double, lattice: Lattice) {
        let runStarted = DispatchTime.now().uptimeNanoseconds
        defer { latticeMilliseconds += Double(DispatchTime.now().uptimeNanoseconds - runStarted) / 1e6 }
        let n = chars.count

        func slice(_ from: Int, _ to: Int) -> String {
            String(decoding: chars[from..<to], as: UTF8.self)
        }
        func skipSeparators(_ position: Int) -> Int {
            var p = position
            while p < n && chars[p] == Self.apostrophe { p += 1 }
            return p
        }

        // steps[i]: the syllables that the letters starting at i can stand for. A step is one
        // syllable, or two when the letters either side of their boundary were typed in swapped order.
        typealias Step = (end: Int, syllables: [String], cuts: [Int], penalty: Double)
        var phaseStarted = DispatchTime.now().uptimeNanoseconds
        func phase(_ k: Int) { let now = DispatchTime.now().uptimeNanoseconds; phaseMilliseconds[k] += Double(now - phaseStarted) / 1e6; phaseStarted = now }
        var steps = [[Step]](repeating: [], count: n)
        for i in 0..<n where chars[i] != Self.apostrophe {
            for j in (i + 1)...min(n, i + SyllableMatcher.maxTypedLength) {
                if chars[j - 1] == Self.apostrophe { break }
                for match in matcher.matches(slice(i, j), tolerant: tolerant) {
                    // Slips of the finger (not fuzzy sounds) scale with slipScale.
                    let penalty = match.penalty >= SyllableMatcher.minimumTypoPenalty ? match.penalty * slipScale : match.penalty
                    steps[i].append((j, [match.syllable], [j], penalty))
                }
            }
            guard tolerant else { continue }
            for j in (i + 1)..<min(n, i + SyllableMatcher.maxTypedLength) {
                if chars[j - 1] == Self.apostrophe || chars[j] == Self.apostrophe { break }
                guard chars[j - 1] != chars[j] else { continue }
                let first = slice(i, j - 1) + slice(j, j + 1)
                guard lexicon.syllables.contains(first) else { continue }
                for k in (j + 1)...min(n, j + SyllableMatcher.maxTypedLength - 1) {
                    if chars[k - 1] == Self.apostrophe { break }
                    let second = slice(j - 1, j) + slice(j + 1, k)
                    if lexicon.syllables.contains(second) {
                        steps[i].append((k, [first, second], [j, k], SyllableMatcher.minimumTypoPenalty))
                    }
                }
            }
        }

        phase(0)
        var edges: [Edge] = []
        var reachable = [Bool](repeating: false, count: n + 1)
        let origin = skipSeparators(0)
        reachable[origin] = true

        func word(_ entry: Lexicon.Entry, key: String) -> WordRef {
            WordRef(text: entry.word, wid: entry.wid, key: key)
        }

        func extend(start: Int, position: Int, key: String, cuts: [Int], penalty: Double, corrected: Bool) {
            let depth = cuts.count
            var exactAtEnd = Set<String>()
            for step in steps[position] where depth + step.syllables.count <= Self.maxWordSyllables {
                let isCorrection = step.penalty >= SyllableMatcher.minimumTypoPenalty
                if isCorrection && corrected { continue }
                let syllables = step.syllables.joined(separator: "'")
                let newKey = key.isEmpty ? syllables : key + "'" + syllables
                guard lexicon.hasPrefix(newKey) else { continue }
                let end = skipSeparators(step.end)
                let newCuts = cuts + step.cuts
                let newPenalty = penalty + step.penalty

                var known = Set<String>()
                for entry in timedLexicon({ lexicon.exact(newKey) }) {
                    known.insert(entry.word)
                    edges.append(Edge(start: start, end: end, word: word(entry, key: newKey), logp: entry.logp,
                                      penalty: newPenalty, cuts: newCuts, partial: false, english: false))
                }
                if !known.isEmpty { reachable[end] = true }
                if end == n { exactAtEnd.formUnion(known) }
                if end < n && newCuts.count < Self.maxWordSyllables {
                    extend(start: start, position: end, key: newKey, cuts: newCuts, penalty: newPenalty, corrected: corrected || isCorrection)
                }
            }

            // The rest of the buffer may be an unfinished syllable ("zhongg" -> zhong'g...).
            let rest = chars[position..<n]
            if rest.count < SyllableMatcher.maxTypedLength, !rest.contains(Self.apostrophe), position < n {
                let tail = slice(position, n)
                if lexicon.syllablePrefixes.contains(tail) {
                    let keyPrefix = key.isEmpty ? tail : key + "'" + tail
                    for entry in timedLexicon({ lexicon.partial(syllableCount: depth + 1, keyPrefix: keyPrefix) }) where !exactAtEnd.contains(entry.word) {
                        reachable[n] = true
                        edges.append(Edge(start: start, end: n, word: word(entry, key: entry.key), logp: entry.logp,
                                          penalty: penalty + partialPenalty, cuts: cuts + [n], partial: true, english: false))
                    }
                }
            }
        }

        // Words with some syllables cut short to their initial: "shurf" -> shu'r'f -> 输入法, "zg" -> 中国.
        // A letter may be a bare initial even where a full syllable starts ("mlnao": n + ao, not nao).
        func abbreviate(start: Int, position: Int, segments: [(text: String, complete: Bool)], cuts: [Int]) {
            guard segments.count < Self.maxAbbreviatedSyllables, position < n else { return }
            let initials = segments.filter { !$0.complete }.count

            func add(_ text: String, complete: Bool, to j: Int) {
                let end = skipSeparators(j)
                let newSegments = segments + [(text, complete)]
                let newCuts = cuts + [j]
                let newInitials = initials + (complete ? 0 : 1)
                if newInitials == 0 {
                    guard lexicon.hasPrefix(newSegments.map(\.text).joined(separator: "'")) else { return }
                } else if newSegments.count > 1 {
                    // Words with some syllables cut to their initial. A single character by its initial
                    // alone was tried everywhere and at the buffer end only: both lower the word tests.
                    for entry in timedLexicon({ lexicon.abbreviated(newSegments) }) {
                        reachable[end] = true
                        edges.append(Edge(start: start, end: end, word: word(entry, key: entry.key), logp: entry.logp,
                                          penalty: Double(newInitials) * initialPenalty, cuts: newCuts, partial: true, english: false))
                    }
                }
                abbreviate(start: start, position: end, segments: newSegments, cuts: newCuts)
            }

            for step in steps[position] where step.penalty == 0 { add(step.syllables[0], complete: true, to: step.end) }
            // Any first letter may stand for its syllable, a vowel too (aw -> 爱我, oz -> 欧洲).
            do {
                add(slice(position, position + 1), complete: false, to: position + 1)
                if position + 1 < n, chars[position + 1] == UInt8(ascii: "h"), Self.retroflexInitials.contains(chars[position]) {
                    add(slice(position, position + 2), complete: false, to: position + 2)
                }
            }
        }

        // A run of letters may be an English word.
        func english(start: Int) {
            // A single letter is the start of a syllable, never an English word.
            guard start + 2 <= n else { return }
            for j in (start + 2)...min(n, start + Self.maxEnglishLength) {
                if chars[j - 1] == Self.apostrophe { break }
                // One or two letters typed on their own are a syllable's start or an abbreviation far
                // more often than an English word: such an English reading pays what the abbreviation
                // pays (initialPenalty per letter), and the frequencies decide. Inside a longer buffer
                // English keeps its own cost.
                let letters = slice(start, j)
                let table = timedLexicon({ lexicon.english(letters) })
                for entry in table {
                    reachable[j] = true
                    edges.append(Edge(start: start, end: j, word: WordRef(text: entry.word, wid: 0, key: entry.key), logp: entry.logp,
                                      penalty: englishPenalty, cuts: [j], partial: false, english: true))
                }
                for word in typedEnglish?(letters) ?? [] where !table.contains(where: { $0.word == word }) {
                    reachable[j] = true
                    edges.append(Edge(start: start, end: j, word: WordRef(text: word, wid: 0, key: letters), logp: lexicon.englishFloor,
                                      penalty: englishPenalty, cuts: [j], partial: false, english: true))
                }
            }
        }

        // Capital letters typed into the buffer are kept as they are: one edge per run of capitals.
        func literal(start: Int) -> Bool {
            guard Self.isCapital(chars[start]) else { return false }
            var end = start
            while end < n && Self.isCapital(chars[end]) { end += 1 }
            let text = String(decoding: chars[start..<end], as: UTF8.self)
            reachable[end] = true
            edges.append(Edge(start: start, end: end, word: WordRef(text: text, wid: 0, key: text.lowercased()), logp: 0,
                              penalty: 0, cuts: [end], partial: false, english: true))
            return true
        }

        for start in 0..<n where reachable[start] {
            if literal(start: start) { continue }
            extend(start: start, position: start, key: "", cuts: [], penalty: 0, corrected: false)
            abbreviate(start: start, position: start, segments: [], cuts: [])
            english(start: start)
        }
        guard let target = (1...n).last(where: { reachable[$0] }) else { return (.empty, [], 0, -.infinity, Lattice(edges: [], origin: 0, target: 0)) }

        phase(1)
        // Viterbi over the lattice under the lexicon's frequencies; each node keeps its best few states.
        var edgesByStart = [[Int]](repeating: [], count: n + 1)
        for (index, edge) in edges.enumerated() { edgesByStart[edge.start].append(index) }
        var states = [[State]](repeating: [], count: n + 1)
        states[origin] = [State(score: 0, edge: -1, previous: -1)]
        // Nodes up to reusedUpTo carry the previous run's (already pruned) states; only edges that
        // leave the old input still have to be followed from them.
        var reusedUpTo = -1
        if incremental, let cache = runCaches[tolerant], cache.origin == origin, cache.chars.count < n,
           chars.starts(with: cache.chars) {
            let old = cache.chars.count
            var newIndex: [EdgeKey: Int] = [:]
            for (index, edge) in edges.enumerated() where edge.end <= old { newIndex[EdgeKey(edge)] = index }
            var remapped = [[State]](repeating: [], count: n + 1)
            var complete = true
            for node in 0...old where complete {
                remapped[node] = cache.states[node].compactMap { state in
                    if state.edge < 0 { return state }
                    guard let index = newIndex[EdgeKey(cache.edges[state.edge])] else { complete = false; return nil }
                    return State(score: state.score, edge: index, previous: state.previous)
                }
            }
            if complete {
                states = remapped
                reusedUpTo = old
            }
        }

        func prune(_ node: Int) {
            // One state per word (by lexicon id; English words, which share id 0, by text).
            var bestByWid: [UInt32: State] = [:]
            var bestByText: [String: State] = [:]
            for state in states[node] {
                if state.edge < 0 { bestByWid[UInt32.max] = state; continue }
                let word = edges[state.edge].word
                if word.wid != 0 {
                    if let existing = bestByWid[word.wid], existing.score >= state.score { continue }
                    bestByWid[word.wid] = state
                } else {
                    if let existing = bestByText[word.text], existing.score >= state.score { continue }
                    bestByText[word.text] = state
                }
            }
            var kept = Array(bestByWid.values) + Array(bestByText.values)
            kept.sort { $0.score > $1.score }
            states[node] = Array(kept.prefix(beamWidth))
        }

        for node in origin..<n where !states[node].isEmpty {
            if node > reusedUpTo { prune(node) }
            let previousWords = states[node].map { $0.edge < 0 ? nil : edges[$0.edge].word }
            for index in edgesByStart[node] {
                let edge = edges[index]
                if node <= reusedUpTo && edge.end <= reusedUpTo { continue }
                var best = -Double.infinity
                var bestPrevious = -1
                for (stateIndex, state) in states[node].enumerated() {
                    let score = state.score + transition(from: previousWords[stateIndex], to: edge)
                    if score > best {
                        best = score
                        bestPrevious = stateIndex
                    }
                }
                states[edge.end].append(State(score: best, edge: index, previous: bestPrevious))
            }
        }
        if target == n { prune(target) }
        if incremental && target == n {
            runCaches[tolerant] = RunCache(chars: chars, edges: edges, states: states, origin: origin)
        }
        phase(2)
        let sentences = bestSentences(edges: edges, states: states, target: target)
        phase(3)

        var path: [Edge] = []
        var node = target
        var current = states[target].max { $0.score < $1.score }
        let score = current?.score ?? -.infinity
        while let state = current, state.edge >= 0 {
            let edge = edges[state.edge]
            path.append(edge)
            node = edge.start
            current = states[node][state.previous]
        }
        path.reverse()

        var candidates: [Candidate] = []
        var seen = Set<String>()
        func add(_ text: String, _ consumed: Int, _ words: [WordRef], _ score: Double) {
            guard candidates.count < Self.maxCandidates, seen.insert("\(consumed)|\(text)").inserted else { return }
            candidates.append(Candidate(text: text, consumed: consumed, words: words, score: score))
        }
        if path.count > 1 {
            add(path.map(\.word.text).joined(), target, path.map(\.word), score)
        }
        // The words that can start the buffer: longest first, complete readings before partial
        // ones, then by the lexicon's order.
        let leading = edgesByStart[origin]
            .map { (edge: edges[$0], score: transition(from: nil, to: edges[$0])) }
            .sorted {
                if $0.edge.end != $1.edge.end { return $0.edge.end > $1.edge.end }
                return $0.score > $1.score
            }
        for (edge, score) in leading {
            add(edge.word.text, edge.end, [edge.word], score)
        }
        // When the buffer is one syllable, every other character read so follows, past the cap above:
        // all characters must be reachable by turning the pages (GB/T 19246-2003 4.2). A rare one is
        // typed by its syllable alone; longer buffers skip this, as it would slow every keystroke.
        var syllablesListed = Set<String>()
        var listed: Set<String>?
        for (edge, _) in leading where edge.end == n && edge.word.wid != 0 && !edge.partial && !edge.english && edge.word.text.count == 1 {
            guard syllablesListed.insert(edge.word.key).inserted else { continue }
            if listed == nil { listed = Set(candidates.lazy.filter { $0.consumed == n }.map(\.text)) }
            for candidate in allCharacters(edge.word.key, consumed: n) where listed!.insert(candidate.text).inserted {
                candidates.append(candidate)
            }
        }

        var cuts = Set(path.flatMap(\.cuts))
        cuts.remove(n)
        // Syllables found by the decoder are shown apart; a "'" the person typed shows as typed (xi'an).
        var segmented = ""
        for (i, c) in chars.enumerated() {
            if cuts.contains(i) && c != Self.apostrophe && segmented.last != " " && segmented.last != "'" { segmented += " " }
            segmented += String(UnicodeScalar(c))
        }
        phase(4)
        let result = DecodeResult(candidates: candidates, sentences: sentences.map(\.candidate), segmented: segmented)
        return (result, sentences, target, score, Lattice(edges: edges, origin: origin, target: target))
    }

    /// The best readings of the buffer up to `target`, in exact score order: an A* search backwards
    /// from the end, guided by each state's best score from the start.
    private func bestSentences(edges: [Edge], states: [[State]], target: Int) -> [Sentence] {
        struct Hypothesis {
            let priority: Double
            let suffix: Double
            let node: Int
            let state: Int
            let tail: [Int]
        }
        var heap: [Hypothesis] = []
        func push(_ hypothesis: Hypothesis) {
            heap.append(hypothesis)
            var child = heap.count - 1
            while child > 0, heap[(child - 1) / 2].priority < heap[child].priority {
                heap.swapAt(child, (child - 1) / 2)
                child = (child - 1) / 2
            }
        }
        func pop() -> Hypothesis? {
            guard !heap.isEmpty else { return nil }
            heap.swapAt(0, heap.count - 1)
            let top = heap.removeLast()
            var parent = 0
            while true {
                var largest = parent
                for child in [2 * parent + 1, 2 * parent + 2] where child < heap.count && heap[child].priority > heap[largest].priority {
                    largest = child
                }
                if largest == parent { break }
                heap.swapAt(parent, largest)
                parent = largest
            }
            return top
        }

        for (index, state) in states[target].enumerated() {
            push(Hypothesis(priority: state.score, suffix: 0, node: target, state: index, tail: []))
        }
        var sentences: [Sentence] = []
        var extras: [Sentence] = []
        var english = 0, abbreviated = 0
        var seen = Set<String>()
        var steps = 0
        while steps < Self.maxSentenceSearchSteps, let hypothesis = pop() {
            steps += 1
            let state = states[hypothesis.node][hypothesis.state]
            guard state.edge >= 0 else {
                let path = hypothesis.tail.reversed().map { edges[$0] }
                let text = path.map(\.word.text).joined()
                if seen.insert(text).inserted {
                    var sentence = Sentence(
                        candidate: Candidate(text: text, consumed: target, words: path.map(\.word), score: hypothesis.priority)
                    )
                    sentence.english = path.contains { $0.english }
                    sentence.abbreviated = path.contains { $0.partial && !$0.english }
                    sentence.penalty = path.reduce(0) { $0 + $1.penalty }
                    sentence.candidate.penalty = sentence.penalty
                    if sentences.count < sentenceCount {
                        sentences.append(sentence)
                    } else if sentence.english && english < englishExtras {
                        extras.append(sentence)
                        english += 1
                    } else if sentence.abbreviated && !sentence.english && abbreviated < abbreviationExtras {
                        extras.append(sentence)
                        abbreviated += 1
                    }
                    if sentences.count == sentenceCount && english >= englishExtras && abbreviated >= abbreviationExtras { break }
                }
                continue
            }
            let edge = edges[state.edge]
            for (index, previous) in states[edge.start].enumerated() {
                let previousWord = previous.edge < 0 ? nil : edges[previous.edge].word
                let suffix = hypothesis.suffix + transition(from: previousWord, to: edge)
                push(Hypothesis(priority: previous.score + suffix, suffix: suffix, node: edge.start, state: index, tail: hypothesis.tail + [state.edge]))
            }
        }
        // Readings with an English word or an abbreviation take the places of the weakest literal
        // readings, so the model gets to see them (PC行业 against 平常也) in the one batch.
        if !extras.isEmpty {
            sentences = Array(sentences.prefix(max(1, sentenceCount - extras.count))) + extras
        }
        return sentences
    }
}
