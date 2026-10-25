import CoreML
import Foundation

/// The character-level language model exported by Training/export_coreml.py, run through Core ML.
/// Scores how well candidate texts continue what was typed before.
public final class NeuralScorer {
    public enum ScorerError: Error {
        case badVocabulary
    }

    /// Rows per model call and tokens per row; fixed when the model is exported. The batch size is
    /// read from the model; `defaultBatchSize` is what the standard export uses.
    public static let defaultBatchSize = 16
    /// Tokens per row, read from the model (64 for the original export, 32 for the short export).
    public static let defaultRowLength = 64
    public let rowLength: Int
    public let batchSize: Int

    private static let pad: Int32 = 0
    private static let unknown: Int32 = 2

    private let model: MLModel
    private let tokens: [Character]
    private let ids: [Character: Int32]
    private let newline: Int32

    /// `modelURL` is a compiled .mlmodelc; `vocabularyURL` the vocab.json it was trained with.
    public init(modelURL: URL, vocabularyURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        batchSize = model.modelDescription.inputDescriptionsByName["ids"]?.multiArrayConstraint?.shape.first?.intValue ?? Self.defaultBatchSize
        rowLength = model.modelDescription.inputDescriptionsByName["ids"]?.multiArrayConstraint?.shape.last?.intValue ?? Self.defaultRowLength

        let vocabulary = try JSONDecoder().decode([String].self, from: Data(contentsOf: vocabularyURL))
        // Special tokens ("<pad>" ...) are longer than one character and never match typed text.
        tokens = vocabulary.map { $0.count == 1 ? $0.first! : "\u{0}" }
        var ids: [Character: Int32] = [:]
        for (index, token) in vocabulary.enumerated() where token.count == 1 {
            ids[token.first!] = Int32(index)
        }
        guard let newline = ids["\n"] else { throw ScorerError.badVocabulary }
        self.ids = ids
        self.newline = newline
    }

    /// How many tokens the model has.
    public var vocabularySize: Int { tokens.count }

    /// The model's token for a character, if it has one.
    public func token(for character: Character) -> Int32? {
        ids[character]
    }

    public func character(for token: Int32) -> Character {
        tokens.indices.contains(Int(token)) ? tokens[Int(token)] : "\u{FFFD}"
    }

    public func encode(_ text: String) -> [Int32] {
        // A cased vocabulary keeps capitals (the model learnt how English is written); a letter
        // the model has no token for falls back to its small form.
        text.map { ids[$0] ?? $0.lowercased().first.flatMap { ids[$0] } ?? Self.unknown }
    }

    /// The context tokens that fit in a row next to `reserved` more tokens. Never empty:
    /// with nothing before the cursor the text is taken to start a new paragraph.
    public func contextTokens(_ context: String, reserving reserved: Int) -> [Int32] {
        let encoded = encode(context)
        let room = max(1, rowLength - reserved)
        return encoded.isEmpty ? [newline] : Array(encoded.suffix(room))
    }

    /// Model invocations so far, for performance measurements.
    public private(set) var callCount = 0
    /// The deferred rerank and learning may call the model from different threads.
    private let runLock = NSLock()

    private func run(rows: [[Int32]], targets: [[Int32]], lastIndexes: [Int32]) -> (targetLogp: MLMultiArray, nextLogp: MLMultiArray)? {
        runLock.lock()
        defer { runLock.unlock() }
        callCount += 1
        let shape = [NSNumber(value: batchSize), NSNumber(value: rowLength)]
        guard let idArray = try? MLMultiArray(shape: shape, dataType: .int32),
              let targetArray = try? MLMultiArray(shape: shape, dataType: .int32),
              let lastArray = try? MLMultiArray(shape: [NSNumber(value: batchSize)], dataType: .int32)
        else { return nil }
        idArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            buffer.initialize(repeating: Self.pad)
            for (r, row) in rows.enumerated() {
                for (t, token) in row.enumerated() { buffer[r * rowLength + t] = token }
            }
        }
        targetArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            buffer.initialize(repeating: Self.pad)
            for (r, row) in targets.enumerated() {
                for (t, token) in row.enumerated() { buffer[r * rowLength + t] = token }
            }
        }
        lastArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            buffer.initialize(repeating: 0)
            for (r, index) in lastIndexes.enumerated() { buffer[r] = index }
        }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["ids": idArray, "targets": targetArray, "last_index": lastArray]),
              let output = try? model.prediction(from: input),
              let targetLogp = output.featureValue(for: "target_logp")?.multiArrayValue,
              let nextLogp = output.featureValue(for: "next_logp")?.multiArrayValue
        else { return nil }
        return (targetLogp, nextLogp)
    }

    /// log P(candidate | prefix) for up to `batchSize` candidates after a row prefix of tokens
    /// (a start token and the text before the cursor), or nil if the model cannot be run.
    public func score(prefix: [Int32], candidates: [String]) -> [Double]? {
        let encoded = candidates.prefix(batchSize).map(encode)
        guard let longest = encoded.map(\.count).max(), longest > 0, longest < rowLength, !prefix.isEmpty else { return nil }
        let head = Array(prefix.suffix(rowLength - longest))
        let rows = encoded.map { head + $0 }
        let targets = rows.map { Array($0.dropFirst()) }
        guard let output = run(rows: rows, targets: targets, lastIndexes: []) else { return nil }
        return encoded.enumerated().map { r, candidate in
            var total = 0.0
            for t in (head.count - 1)..<(head.count - 1 + candidate.count) {
                total += output.targetLogp[[NSNumber(value: r), NSNumber(value: t)]].doubleValue
            }
            return total
        }
    }
}
