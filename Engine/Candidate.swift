import Foundation

/// A word as used in a conversion (kept for hosts that track the last committed word).
public struct WordRef: Equatable {
    public let text: String
    public let wid: UInt32
    /// Canonical pinyin, syllables joined by "'".
    public let key: String
}

/// One conversion choice for the front of the pinyin buffer.
public struct Candidate: Equatable {
    public let text: String
    /// How many characters of the buffer this candidate converts.
    public let consumed: Int
    public let words: [WordRef]
    /// Log-probability under the models.
    public var score = 0.0
    /// Typing costs along the reading (unfinished syllables, initials, slips, English), already
    /// included in `score`; kept apart so a reranker can weigh them.
    public var penalty = 0.0
    public init(text: String, consumed: Int, words: [WordRef], score: Double = 0) {
        self.text = text
        self.consumed = consumed
        self.words = words
        self.score = score
    }
}

public struct DecodeResult {
    public var candidates: [Candidate]
    /// The best few readings of the whole buffer, best first; what a reranker chooses between.
    public var sentences: [Candidate]
    /// The buffer split into syllables along the best path, for display.
    public let segmented: String
    /// A slower second look (the language model's rerank) that may produce a better result; nil when
    /// nothing more is worth trying. The host runs it after showing this result. The closure is
    /// given a check that says whether the host still wants the result (a newer key cancels it);
    /// long looks consult it between steps.
    public var refine: ((_ isCancelled: @escaping () -> Bool) -> DecodeResult?)?

    public static let empty = DecodeResult(candidates: [], sentences: [], segmented: "")

    public init(candidates: [Candidate], sentences: [Candidate], segmented: String, refine: ((_ isCancelled: @escaping () -> Bool) -> DecodeResult?)? = nil) {
        self.candidates = candidates
        self.sentences = sentences
        self.segmented = segmented
        self.refine = refine
    }

    /// This result with every further look applied, for callers that can wait.
    public func refined() -> DecodeResult {
        var result = self
        while let refine = result.refine, let better = refine({ false }) { result = better }
        result.refine = nil
        return result
    }
}

/// Something that turns a pinyin letter buffer into ranked candidates.
public protocol CandidateDecoding: AnyObject {
    /// `input` holds pinyin letters and "'" separators; `precedingText` is the text in front of the
    /// cursor, as far as it is known.
    func decode(_ input: String, precedingText: String) -> DecodeResult
}
