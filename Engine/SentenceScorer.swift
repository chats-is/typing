import Foundation
import MetalPerformanceShadersGraph

/// What the lattice decoder needs from a language model: token tables and log-probabilities of
/// candidate readings after a prefix. Two runtimes
/// provide it: `NeuralScorer` (Core ML, the shipped weights) and `GraphScorer` (MPSGraph, the
/// same weights plus whatever the top blocks have learned on this machine).
public protocol SentenceScorer: AnyObject {
    var rowLength: Int { get }
    var batchSize: Int { get }
    var vocabularySize: Int { get }
    func token(for character: Character) -> Int32?
    func character(for token: Int32) -> Character
    func encode(_ text: String) -> [Int32]
    /// The last tokens of `context` that fit a row beside `reserved` more tokens.
    func contextTokens(_ context: String, reserving reserved: Int) -> [Int32]
    /// log P(candidate | prefix) for up to `batchSize` candidates after a row prefix of tokens.
    func score(prefix: [Int32], candidates: [String]) -> [Double]?
}

extension NeuralScorer: SentenceScorer {}

/// Serialises every MPSGraph compilation and run in the process: compiling two graphs at once
/// (the learner's and the input method's) crashes inside MPSGraph (seen in crash reports).
public enum GraphLock {
    public static let shared = NSRecursiveLock()

    /// GPU only: the default optimisation level also plans Neural Engine regions, which wrote
    /// gigabytes of compile caches to the temporary folder and crashed in that planning.
    public static var gpuOnly: MPSGraphCompilationDescriptor {
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level0
        return descriptor
    }
}
