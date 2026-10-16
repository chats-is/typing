import CoreML
import Engine
import Foundation

// Usage: enginecli --model LanguageModel.mlmodelc --vocab vocab.json --lexicon lexicon.sqlite [--bigram bigram.bin]
//        [--graph-scorer weights | --tuned tuned] [--eval eval.tsv] [pinyin...]
// Prints the segmentation, top candidates and decode time for each pinyin string, or with
// --eval scores the first candidate against a "pin yin<TAB>sentence" file. "pinyin=2,1" types the pinyin and
// picks candidate 2, then candidate 1 of what is left, like the input method does.
// --keystroke-eval types each sentence of an eval file and counts the keys a user would need to get it right.
// --context gives the text in front of the cursor for the pinyin arguments.
// An --eval file may carry a third column with the text before each sentence; --no-context ignores it.
// --typing makes --eval type each sentence letter by letter and time every keystroke; --limit caps the sentences.
// --pace MS types --eval through the input session at that many ms per key; --stress SECONDS runs the engine
// from several threads at once; --learn-demo NAME runs a learning session on Data/personal/NAME.
func timed<T>(_ body: () -> T) -> (T, Double) {
    let started = DispatchTime.now().uptimeNanoseconds
    let value = body()
    return (value, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
}

var evalPath: String?
var modelPath: String?
var vocabularyPath: String?
var computeUnits = MLComputeUnits.cpuAndNeuralEngine
var keystrokeEvalPath: String?
var showFailures = false
var noSlips = false
var lexiconPath: String?
var bigramPath: String?
var bigramWeight: Double?
var startBigramWeight: Double?
var sentenceCount: Int?
var extraMargin: Double?
var skipMargin: Double?
var extras: [Int]?
var noRerank = false
var contextWeight: Double?
var rerankLattice: Double?
var slipScale: Double?
var correctionMargin: Double?
var noDefer = false
var latticeBeam: Int?
var noIncremental = false
var dumpLatticePath: String?
var graphWeightsPath: String?
var tunedWeightsPath: String?
var learnDemo: String?
var abbreviationLimit: Int?
var stressSeconds: Double?
var paceMilliseconds: Double?
var compareBackends = false
var englishPenalty: Double?
var initialPenalty: Double?
var partialPenalty: Double?
var wordPenalty: Double?
var simulateTyping = false
var precedingText = ""
var ignoreContext = false
var limit = Int.max
var positional: [String] = []
var arguments = CommandLine.arguments.dropFirst()[...]
while let argument = arguments.popFirst() {
    switch argument {
    case "--model": modelPath = arguments.popFirst()
    case "--vocab": vocabularyPath = arguments.popFirst()
    case "--cpu": computeUnits = .cpuOnly
    case "--eval": evalPath = arguments.popFirst()
    case "--keystroke-eval": keystrokeEvalPath = arguments.popFirst()
    case "--show-failures": showFailures = true
    case "--lexicon": lexiconPath = arguments.popFirst()
    case "--bigram": bigramPath = arguments.popFirst()
    case "--bigram-weight": bigramWeight = arguments.popFirst().flatMap(Double.init)
    case "--start-weight": startBigramWeight = arguments.popFirst().flatMap(Double.init)
    case "--sentences": sentenceCount = arguments.popFirst().flatMap(Int.init)
    case "--extra-margin": extraMargin = arguments.popFirst().flatMap(Double.init)
    case "--skip-margin": skipMargin = arguments.popFirst().flatMap(Double.init)
    case "--extras":  // english,abbr,slip counts
        if let parts = arguments.popFirst()?.split(separator: ",").map({ Int($0) ?? 0 }), parts.count == 3 { extras = parts }
    case "--no-rerank": noRerank = true
    case "--context-weight": contextWeight = arguments.popFirst().flatMap(Double.init)
    case "--rerank-lattice": rerankLattice = arguments.popFirst().flatMap(Double.init)
    case "--slip-scale": slipScale = arguments.popFirst().flatMap(Double.init)
    case "--correction-margin": correctionMargin = arguments.popFirst().flatMap(Double.init)
    case "--no-defer": noDefer = true
    case "--beam": latticeBeam = arguments.popFirst().flatMap { Int($0) }
    case "--no-incremental": noIncremental = true
    case "--dump-lattice": dumpLatticePath = arguments.popFirst()
    case "--graph-scorer": graphWeightsPath = arguments.popFirst()
    case "--tuned": tunedWeightsPath = arguments.popFirst()
    case "--learn-demo": learnDemo = arguments.popFirst()  // name under Data/personal, e.g. forum2
    case "--abbr-limit": abbreviationLimit = arguments.popFirst().flatMap { Int($0) }
    case "--stress": stressSeconds = arguments.popFirst().flatMap(Double.init)
    case "--compare-backends": compareBackends = true  // with --tuned: tuned Core ML against the tuned graph runtime
    case "--pace": paceMilliseconds = arguments.popFirst().flatMap(Double.init)  // type --eval sentences through InputSession at this many ms per key
    case "--english-penalty": englishPenalty = arguments.popFirst().flatMap(Double.init)
    case "--initial-penalty": initialPenalty = arguments.popFirst().flatMap(Double.init)
    case "--partial-penalty": partialPenalty = arguments.popFirst().flatMap(Double.init)
    case "--word-penalty": wordPenalty = arguments.popFirst().flatMap(Double.init)
    case "--no-slips": noSlips = true
    case "--typing": simulateTyping = true
    case "--context": precedingText = arguments.popFirst() ?? ""
    case "--no-context": ignoreContext = true
    case "--limit": limit = arguments.popFirst().flatMap(Int.init) ?? limit
    default: positional.append(argument)
    }
}
var scorer: NeuralScorer?
if let modelPath, let vocabularyPath {
    scorer = try NeuralScorer(
        modelURL: URL(fileURLWithPath: modelPath), vocabularyURL: URL(fileURLWithPath: vocabularyPath), computeUnits: computeUnits
    )
}
guard let scorer, let lexiconPath else {
    FileHandle.standardError.write(Data("--model, --vocab and --lexicon are required\n".utf8))
    exit(2)
}
let engine: CandidateDecoding
var lexiconEngine: LexiconDecoder?
var engineSentenceScorer: SentenceScorer?  // what the lexicon engine ranks with (graph or Core ML)
if let abbreviationLimit { Lexicon.abbreviationLimit = abbreviationLimit }
do {
    // The lexicon engine: the lexicon says what the keys can be, the language model ranks by context.
    let bigram = try bigramPath.map { try BigramModel(path: $0) }
    var sentenceScorer: SentenceScorer? = scorer
    if let graphWeightsPath {
        let started = Date()
        sentenceScorer = try GraphScorer(weightsPath: graphWeightsPath, tunedPath: tunedWeightsPath, vocabularyURL: URL(fileURLWithPath: vocabularyPath ?? "Resources/vocab.json"))
        FileHandle.standardError.write("graph scorer ready in \(String(format: "%.1f", Date().timeIntervalSince(started))) s\n".data(using: .utf8)!)
    }
    if graphWeightsPath == nil, let tunedWeightsPath, let modelPath {
        // --tuned alone: the learned tensors written into a copy of the Core ML model (Neural Engine).
        let started = Date()
        let patched = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("typing-tuned-\(getpid()).mlmodelc")
        try TunedModel.build(shipped: URL(fileURLWithPath: modelPath), tuned: try GraphScorer.Weights(path: tunedWeightsPath), destination: patched)
        sentenceScorer = try NeuralScorer(modelURL: patched, vocabularyURL: URL(fileURLWithPath: vocabularyPath ?? "Resources/vocab.json"))
        FileHandle.standardError.write("tuned Core ML model ready in \(String(format: "%.1f", Date().timeIntervalSince(started))) s\n".data(using: .utf8)!)
        if compareBackends {
            let graph = try GraphScorer(weightsPath: URL(fileURLWithPath: modelPath).deletingLastPathComponent().appendingPathComponent("weights").path, tunedPath: tunedWeightsPath, vocabularyURL: URL(fileURLWithPath: vocabularyPath ?? "Resources/vocab.json"))
            let coreML = sentenceScorer!
            var worst = 0.0
            for (context, candidates) in [("今天天气", ["很好", "不错", "怎么样", "一般", "太热了"]), ("我觉得这个", ["问题", "方案", "人", "东西", "办法"]), ("", ["你好", "我们", "明天见", "谢谢", "不知道"])] {
                let prefix: [Int32] = [1] + coreML.contextTokens(context, reserving: 1)
                guard let a = coreML.score(prefix: prefix, candidates: candidates), let b = graph.score(prefix: prefix, candidates: candidates) else { continue }
                for (x, y) in zip(a, b) { worst = max(worst, abs(x - y)) }
                print("  \(context.isEmpty ? "<start>" : context): coreML " + a.map { String(format: "%.3f", $0) }.joined(separator: " ") + "  graph " + b.map { String(format: "%.3f", $0) }.joined(separator: " "))
            }
            print(String(format: "tuned Core ML vs tuned graph: max |log P difference| %.4f", worst))
        }
    }
    let decoder = LexiconDecoder(lexicon: try Lexicon(path: lexiconPath), scorer: noRerank ? nil : sentenceScorer, bigram: bigram)
    engineSentenceScorer = sentenceScorer
    if tunedWeightsPath != nil, !noRerank { decoder.baseScorer = scorer }  // learning on: shipped model beside the tuned one
    if let bigramWeight { decoder.bigramWeight = bigramWeight }
    if let startBigramWeight { decoder.startBigramWeight = startBigramWeight }
    if let sentenceCount { decoder.sentenceCount = sentenceCount }
    if let extraMargin { decoder.extraMargin = extraMargin }
    if let skipMargin { decoder.skipMargin = skipMargin }
    if let extras { decoder.englishExtras = extras[0]; decoder.abbreviationExtras = extras[1]; decoder.slipExtras = extras[2] }
    if let contextWeight { decoder.contextWeight = contextWeight }
    if let rerankLattice { decoder.rerankLatticeWeight = rerankLattice }
    if let slipScale { decoder.slipScale = slipScale }
    if let correctionMargin { decoder.correctionMargin = correctionMargin }
    if noDefer { decoder.deferRerank = false }
    if let latticeBeam { decoder.beamWidth = latticeBeam }
    if noIncremental { decoder.incremental = false }
    if let englishPenalty { decoder.englishPenalty = englishPenalty }
    if let initialPenalty { decoder.initialPenalty = initialPenalty }
    if let partialPenalty { decoder.partialPenalty = partialPenalty }
    if let wordPenalty { decoder.wordPenalty = wordPenalty }
    decoder.tolerant = !noSlips
    engine = decoder
    lexiconEngine = decoder
}

// "-" is accepted and ignored where older scripts passed a lexicon path.
for input in positional where input != "-" {
    if let separator = input.firstIndex(of: "=") {
        let session = InputSession(decoder: engine)
        input[..<separator].forEach(session.insert)
        var committed: String?
        for pick in input[input.index(after: separator)...].split(separator: ",") {
            // A number picks by position, anything else the candidate with that text.
            let index = Int(pick).map { $0 - 1 } ?? session.candidates.firstIndex { $0.text == pick } ?? 0
            committed = session.select(index)
        }
        print("\(input) -> \(committed ?? "(not finished: \(session.preedit))")")
        continue
    }
    let (result, milliseconds) = timed { engine.decode(input, precedingText: precedingText).refined() }
    print("\(input) -> [\(result.segmented)]  \(String(format: "%.2f", milliseconds))ms  \(result.candidates.count) candidates")
    let top = result.candidates.prefix(9).enumerated().map { "\($0.offset + 1).\($0.element.text)(\($0.element.consumed))" }
    print("  " + top.joined(separator: " "))
    if showFailures {
        print("  sentences: " + result.sentences.map { "\($0.text) \(String(format: "%.1f", $0.score))" }.joined(separator: " | "))
    }
}

if let paceMilliseconds, let evalPath, let lexiconEngine {
    // Real pacing through the session the input method uses: keys arrive on the main run loop
    // every `pace` ms while the deferred looks run on their own queues, as in the app. Reports how
    // long each key blocks the main thread and how late the model's reorder lands.
    let session = InputSession(decoder: lexiconEngine)
    var keyBlock: [Double] = [], refineLag: [Double] = []
    var updates = 0
    var keyTime = DispatchTime.now()
    var sawRefine = false
    session.onUpdate = {
        let lag = Double(DispatchTime.now().uptimeNanoseconds - keyTime.uptimeNanoseconds) / 1e6
        updates += 1
        if !sawRefine { refineLag.append(lag); sawRefine = true }
    }
    var typedKeys = 0
    for line in try String(contentsOfFile: evalPath, encoding: .utf8).split(separator: "\n").prefix(limit) {
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count >= 2 else { continue }
        session.precedingText = fields.count > 2 ? fields[2].replacingOccurrences(of: "\\n", with: "\n") : ""
        for letter in fields[0].replacingOccurrences(of: " ", with: "") {
            keyTime = DispatchTime.now()
            sawRefine = false
            let started = DispatchTime.now()
            session.insert(letter)
            keyBlock.append(Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1e6)
            typedKeys += 1
            RunLoop.main.run(until: Date(timeIntervalSinceNow: paceMilliseconds / 1000))
        }
        _ = session.select(0)
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.3))
    }
    func stats(_ v: [Double]) -> String {
        guard !v.isEmpty else { return "-" }
        let s = v.sorted()
        return String(format: "mean %.1f  p95 %.1f  max %.1f", v.reduce(0, +) / Double(v.count), s[min(s.count - 1, Int(Double(s.count) * 0.95))], s.last!)
    }
    print("pace \(Int(paceMilliseconds)) ms, \(typedKeys) keys: main thread blocked per key ms: \(stats(keyBlock))")
    print("model reorder landed after ms: \(stats(refineLag)) (on \(refineLag.count) keys, \(updates) updates)")
    exit(0)
}

// --stress SECONDS: typing, deferred reranks, short decodes and a settings-style reader all at once
// from several threads, as in the input method, to catch thread-safety faults.
if let stressSeconds, let lexiconEngine {
    let lines = (try? String(contentsOfFile: "Data/eval_word_mixed.tsv", encoding: .utf8))?.split(separator: "\n").map { $0.split(separator: "\t", omittingEmptySubsequences: false) } ?? []
    let deadline = Date(timeIntervalSinceNow: stressSeconds)
    let group = DispatchGroup()
    var counts = [0, 0, 0, 0]
    let countLock = NSLock()
    func bump(_ k: Int) { countLock.lock(); counts[k] += 1; countLock.unlock() }
    for worker in 0..<4 {
        group.enter()
        DispatchQueue.global(qos: worker == 0 ? .userInteractive : .utility).async {
            var i = worker
            while Date() < deadline {
                let f = lines[i % lines.count]; i += 7
                let keys = String(f[0]), context = f.count > 2 ? String(f[2]) : ""
                switch worker {
                case 0:  // typing: letter by letter, deferred rerank on another queue
                    var typed = ""
                    for c in keys { typed.append(c); let r = lexiconEngine.decode(typed, precedingText: context); if let refine = r.refine { DispatchQueue.global().async { _ = refine({ false }) } } }
                    bump(0)
                case 1: _ = lexiconEngine.decode(String(keys.prefix(3)), precedingText: context); bump(1)
                case 2: _ = lexiconEngine.decode(keys, precedingText: context).refined(); bump(2)
                default: _ = lexiconEngine.firstChoice(keys, learned: false); bump(3)
                }
            }
            group.leave()
        }
    }
    group.wait()
    print("stress \(stressSeconds) s without a fault: typed \(counts[0]) words, \(counts[1]) short decodes, \(counts[2]) background decodes, \(counts[3]) settings lookups")
    exit(0)
}

// --learn-demo NAME: an end-to-end learning session on Data/personal/NAME.{train.txt,eval.tsv} with a
// fresh store in a temporary directory, to check the on-device learner against the PyTorch numbers.
if let learnDemo, let lexiconEngine, let graphWeightsPath {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("typing-learn-\(learnDemo)-\(Int(Date().timeIntervalSince1970))")
    let store = LearningStore(directory: directory)
    let train = try String(contentsOfFile: "Data/personal/\(learnDemo).train.txt", encoding: .utf8)
    for line in train.split(separator: "\n") where line.count >= 2 {
        store.record(keys: "", text: String(line), context: "", undoWindow: 0)
    }
    for line in try String(contentsOfFile: "Data/personal/\(learnDemo).eval.tsv", encoding: .utf8).split(separator: "\n").prefix(100) {
        let f = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard f.count >= 3 else { continue }
        store.record(keys: f[0].replacingOccurrences(of: " ", with: ""), text: String(f[1]), context: f[2].replacingOccurrences(of: "\\n", with: "\n"), undoWindow: 0)
    }
    store.flush()
    print("store: \(store.count) entries in \(directory.path)")
    let general = URL(fileURLWithPath: "Data/personal/dialogue.lattice.tsv")
    let learner = try Learner(store: store, weightsPath: graphWeightsPath, vocabularyURL: URL(fileURLWithPath: vocabularyPath ?? "Resources/vocab.json"), generalLatticesURL: general) { keys, context in
        let quick = lexiconEngine.decode(keys, precedingText: context)  // lattice order; the model's look is deferred and not taken
        return quick.sentences.map { (text: $0.text, score: $0.score, penalty: $0.penalty) }
    }
    learner.generalWordsURL = URL(fileURLWithPath: "Resources/general_words.tsv")
    learner.minimumNewEntries = 10
    // As in the app: the engine's own scorer is "before" when it is the GPU runtime.
    learner.engineScorer = { engineSentenceScorer as? GraphScorer }
    // As when the person starts typing during a session: paused for 3 s, then resumed.
    DispatchQueue.global().async {
        Thread.sleep(forTimeInterval: 5)
        learner.paused = true; print("paused (typing)")
        Thread.sleep(forTimeInterval: 3)
        learner.paused = false; print("resumed (idle)")
    }
    let outcome = learner.runSession { print($0) }
    if outcome.accepted {
        // How lone words come out now, with learning gated to what the person wrote.
        let tuned = try GraphScorer(weightsPath: graphWeightsPath, tunedPath: store.tunedBase, vocabularyURL: URL(fileURLWithPath: vocabularyPath ?? "Resources/vocab.json"))
        lexiconEngine.replaceScorer(tuned)
        lexiconEngine.baseScorer = scorer
        lexiconEngine.learnedWords = { store.hasLearned($0) }
        for probe in ["cs", "ces", "chengshi", "sj", "nh", "yx"] {
            let result = lexiconEngine.decode(probe, precedingText: "").refined()
            print("  \(probe) -> " + result.candidates.prefix(5).map(\.text).joined(separator: " "))
        }
    }
    print(String(format: "session: %@ (%@) steps %d  personal %.2f -> %.2f (n=%d)  general %.2f -> %.2f (n=%d)  words %.2f -> %.2f (n=%d)  %.0f s", outcome.accepted ? "KEPT" : "DISCARDED", outcome.reason, outcome.steps, outcome.personalBefore, outcome.personalAfter, outcome.personalCount, outcome.generalBefore, outcome.generalAfter, outcome.generalCount, outcome.wordsBefore, outcome.wordsAfter, outcome.wordsCount, outcome.seconds))
    exit(0)
}

if let evalPath {
    var sentences = 0, exactSentences = 0, characters = 0, correctCharacters = 0, withinBest = 0, onFirstPage = 0, inTopTwo = 0
    var times: [Double] = []
    var refineTimes: [Double] = []
    var timesByLength: [Int: [Double]] = [:]
    var dumpLines: [String] = []
    var byPattern: [String: (exact: Int, page: Int, total: Int)] = [:]
    for line in try String(contentsOfFile: evalPath, encoding: .utf8).split(separator: "\n") {
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count >= 2, sentences < limit else { continue }
        // Line breaks in the context column are written as a literal \n.
        let before = fields.count > 2 && !ignoreContext ? fields[2].replacingOccurrences(of: "\\n", with: "\n") : ""
        let input = fields[0].replacingOccurrences(of: " ", with: "")
        let expected = Array(fields[1])
        var result = DecodeResult.empty
        if simulateTyping {
            var typed = ""
            for letter in input {
                typed.append(letter)
                let (quick, quickMilliseconds) = timed { engine.decode(typed, precedingText: before) }
                // The second look (the model's rerank) is paid per key.
                let (refined, refineMilliseconds) = timed { quick.refine?({ false }) ?? quick }
                times.append(quickMilliseconds)
                refineTimes.append(refineMilliseconds)
                timesByLength[min(typed.count / 6, 4), default: []].append(quickMilliseconds + refineMilliseconds)
                result = refined
            }
        } else {
            let (whole, milliseconds) = timed { engine.decode(input, precedingText: before).refined() }
            times.append(milliseconds)
            result = whole
        }
        let first = result.candidates.first
        let actual = first?.consumed == input.count ? Array(first!.text) : []

        sentences += 1
        characters += expected.count
        if actual == expected { exactSentences += 1 }
        if result.sentences.contains(where: { $0.text == String(expected) }) { withinBest += 1 }
        let onPage = result.candidates.prefix(9).contains { $0.consumed == input.count && $0.text == String(expected) }
        if onPage { onFirstPage += 1 }
        if result.candidates.prefix(2).contains(where: { $0.consumed == input.count && $0.text == String(expected) }) { inTopTwo += 1 }
        // An optional fourth column names the input pattern; accuracy is also reported per pattern.
        if fields.count > 3 {
            let pattern = String(fields[3])
            byPattern[pattern, default: (0, 0, 0)].total += 1
            if actual == expected { byPattern[pattern]!.exact += 1 }
            if onPage { byPattern[pattern]!.page += 1 }
        }
        if dumpLatticePath != nil {
            // input, expected, context, then every lattice reading as text|score|penalty.
            let readings = result.sentences.map { "\($0.text)|\(String(format: "%.4f", $0.score))|\(String(format: "%.4f", $0.penalty))" }.joined(separator: " ;; ")
            dumpLines.append("\(input)\t\(String(expected))\t\(before.replacingOccurrences(of: "\n", with: "\\n"))\t\(readings)")
        }
        if showFailures && actual != expected {
            let rank = result.sentences.firstIndex { $0.text == String(expected) }.map { "rank \($0 + 1)" } ?? "not among best"
            print("\(input)\t\(String(expected))\t\(String(actual))\t\(rank)")
        }
        if actual.count == expected.count {
            correctCharacters += zip(actual, expected).filter { $0 == $1 }.count
        }
    }
    if !times.isEmpty {
        print(String(format: "model calls per decode: %.2f", Double(scorer.callCount) / Double(times.count)))
        if let engine = lexiconEngine {
            print(String(format: "per decode: language model %.1f ms, lattice %.1f ms (lexicon queries %.1f ms); model skipped on %.0f%% of decodes", engine.languageModelMilliseconds / Double(times.count), engine.latticeMilliseconds / Double(times.count), engine.lexiconMilliseconds / Double(times.count), 100 * Double(engine.skippedCount) / Double(max(1, engine.decodeCount))))
            print(String(format: "lattice phases per decode: steps %.1f, edges %.1f, viterbi %.1f, sentences %.1f, candidates %.1f ms", engine.phaseMilliseconds[0] / Double(times.count), engine.phaseMilliseconds[1] / Double(times.count), engine.phaseMilliseconds[2] / Double(times.count), engine.phaseMilliseconds[3] / Double(times.count), engine.phaseMilliseconds[4] / Double(times.count)))
        }
    }
    times.sort()
    func percent(_ part: Int, _ whole: Int) -> String { String(format: "%.2f%%", 100 * Double(part) / Double(max(whole, 1))) }
    print("sentences: \(sentences)  exact: \(percent(exactSentences, sentences))  characters: \(percent(correctCharacters, characters))"
        + "  among best \(lexiconEngine?.sentenceCount ?? 0): \(percent(withinBest, sentences))  first page: \(percent(onFirstPage, sentences))  top 2: \(percent(inTopTwo, sentences))")
    for (pattern, counts) in byPattern.sorted(by: { $0.key < $1.key }) {
        print("  \(pattern): \(percent(counts.exact, counts.total)) of \(counts.total), first page \(percent(counts.page, counts.total))")
    }
    if !times.isEmpty {
        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: "decode ms: mean %.2f  p95 %.2f  max %.2f", mean, times[times.count * 95 / 100], times[times.count - 1]))
        if let dumpLatticePath {
        try? dumpLines.joined(separator: "\n").appending("\n").write(toFile: dumpLatticePath, atomically: true, encoding: .utf8)
        print("wrote \(dumpLines.count) lattices to \(dumpLatticePath)")
    }
    if !timesByLength.isEmpty {
            let buckets = timesByLength.keys.sorted().map { k -> String in
                let v = timesByLength[k]!.sorted()
                return String(format: "%@ letters: mean %.1f p95 %.1f (n=%d)", k < 4 ? "\(k * 6 + 1)-\(k * 6 + 6)" : "25+", v.reduce(0, +) / Double(v.count), v[Int(Double(v.count - 1) * 0.95)], v.count)
            }
            print("per key by buffer length (quick + deferred): " + buckets.joined(separator: "; "))
        }
        if !refineTimes.isEmpty {
            print(String(format: "deferred slip search ms: mean %.2f  max %.2f", refineTimes.reduce(0, +) / Double(refineTimes.count), refineTimes.max()!))
        }
    }
}

if let keystrokeEvalPath {
    // A user types the whole pinyin, then repeatedly picks the longest candidate on the first two
    // pages that is a correct prefix of what is left (space or a digit, plus one key per page
    // turned). A character no candidate offers is counted as given up: typed by other means at
    // an estimated 4 keys.
    let pageSize = 9, pagesLooked = 2, giveUpCost = 4
    var sentences = 0, characters = 0, keys = 0, letters = 0, failures = 0, firstPickWhole = 0
    for line in try String(contentsOfFile: keystrokeEvalPath, encoding: .utf8).split(separator: "\n") {
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count >= 2, sentences < limit else { continue }
        let syllables = fields[0].split(separator: " ").map(String.init)
        let expected = Array(fields[1])
        guard syllables.count == expected.count else { continue }
        let before = fields.count > 2 && !ignoreContext ? fields[2].replacingOccurrences(of: "\\n", with: "\n") : ""
        let session = InputSession(decoder: engine)
        session.precedingText = before
        let typed = syllables.joined()
        for letter in typed { session.insert(letter) }
        sentences += 1
        characters += expected.count
        letters += typed.count
        keys += typed.count

        var done = 0  // characters of `expected` committed so far
        var first = true
        while done < expected.count {
            let candidates = session.candidates
            // Letters consumed by the next k characters.
            func lettersFor(_ k: Int) -> Int { syllables[done..<(done + k)].joined().count }
            var choice: (index: Int, length: Int)?
            for (index, candidate) in candidates.prefix(pageSize * pagesLooked).enumerated() {
                let length = candidate.text.count
                guard length > (choice?.length ?? 0), done + length <= expected.count,
                      Array(candidate.text) == Array(expected[done..<(done + length)]),
                      candidate.consumed == lettersFor(length)
                else { continue }
                choice = (index, length)
            }
            guard let choice else {
                failures += 1
                if showFailures && failures <= 12 {
                    let offered = candidates.prefix(6).map { "\($0.text)(\($0.consumed))" }.joined(separator: " ")
                    print("  stuck at \(String(expected[done...])) [\(syllables[done...].joined(separator: " "))]: \(offered)")
                }
                keys += giveUpCost * (expected.count - done)
                break
            }
            if first && choice.length == expected.count { firstPickWhole += 1 }
            first = false
            keys += 1 + choice.index / pageSize
            done += choice.length
            _ = session.select(choice.index)
        }
    }
    print("typed \(sentences) sentences, \(characters) characters: \(String(format: "%.2f", Double(keys) / Double(max(characters, 1)))) keys per character"
        + " (\(String(format: "%.2f", Double(letters) / Double(max(characters, 1)))) of them pinyin letters);"
        + " whole sentence picked at once \(String(format: "%.1f%%", 100 * Double(firstPickWhole) / Double(max(sentences, 1))));"
        + " gave up on \(failures) sentences")
}
