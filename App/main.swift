import Cocoa
import Engine
import InputMethodKit

enum EngineHost {
    /// The language model: which character comes next. nil if it is not bundled.
    static let scorer: NeuralScorer? = loadScorer()

    /// The model actually ranking: Core ML as shipped, or the GPU runtime with what this machine
    /// has learned once learning is on and a tuned file exists.
    static var sentenceScorer: SentenceScorer? = Learning.initialScorer() ?? scorer
    static var lexiconDecoder: LexiconDecoder? { decoder as? LexiconDecoder }

    /// The engine shared by every input controller: the lexicon says what the keys can be, the
    /// language model ranks by context.
    static let decoder: CandidateDecoding? = loadDecoder()

    private static func loadDecoder() -> CandidateDecoding? {
        if let lexiconURL = Bundle.main.url(forResource: "lexicon", withExtension: "sqlite") {
            do {
                let bigram = Bundle.main.url(forResource: "bigram", withExtension: "bin").flatMap { try? BigramModel(path: $0.path) }
                let decoder = LexiconDecoder(lexicon: try Lexicon(path: lexiconURL.path), scorer: sentenceScorer, bigram: bigram)
                if let sentenceScorer, sentenceScorer !== scorer {
                    decoder.setLearning(base: scorer, words: Learning.learnedWordFilter())
                }
                return decoder
            } catch {
                NSLog("Typing: cannot open lexicon: \(error)")
            }
        }
        return nil
    }

    private static func loadScorer() -> NeuralScorer? {
        guard let model = Bundle.main.url(forResource: "LanguageModel", withExtension: "mlmodelc"),
              let vocabulary = Bundle.main.url(forResource: "vocab", withExtension: "json")
        else { return nil }
        do {
            return try NeuralScorer(modelURL: model, vocabularyURL: vocabulary)
        } catch {
            NSLog("Typing: cannot load language model: \(error)")
            return nil
        }
    }
}

guard let connectionName = Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String else {
    fatalError("InputMethodConnectionName is missing from Info.plist")
}
// MPSGraph leaves a folder per compilation in the temporary directory; remove those of processes that
// have ended (they filled gigabytes before the graphs were compiled GPU-only).
DispatchQueue.global(qos: .background).async {
    let folder = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("com.apple.MetalPerformanceShadersGraph")
    for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] where name.hasPrefix("mpsgraph-") {
        // Only folders whose process is gone (kill with signal 0 just asks whether it exists).
        guard let pid = Int32(name.split(separator: "-")[1]), kill(pid, 0) != 0, errno == ESRCH else { continue }
        try? FileManager.default.removeItem(at: folder.appendingPathComponent(name))
    }
}
let server = IMKServer(name: connectionName, bundleIdentifier: Bundle.main.bundleIdentifier)
NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main) { _ in
    TypingInputController.hidePanel()
}
// Warm up once at launch (model load, first Core ML call, lexicon caches) so the first keystroke
// is as quick as the rest; the first decode otherwise takes over 100 ms.
_ = EngineHost.decoder?.decode("nihao", precedingText: "").refined()
NSApplication.shared.run()
