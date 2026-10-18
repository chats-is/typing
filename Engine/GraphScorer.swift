import Foundation
import Metal
import MetalPerformanceShadersGraph

/// The language model run on the GPU with MPSGraph from the plain weight file
/// (Training/export_weights.py), optionally with tuned tensors learned on this machine replacing
/// some of them. Same rows, same numbers as `NeuralScorer`; this is the runtime used once the
/// model has learned, because Core ML cannot take new weights at run time.
public final class GraphScorer: SentenceScorer {
    public enum GraphError: Error { case noDevice, badHeader, missingTensor(String) }

    public let rowLength: Int
    public let batchSize: Int
    public var vocabularySize: Int { tokens.count }

    private let tokens: [Character]
    private let ids: [Character: Int32]
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice
    private let targetExecutable: MPSGraphExecutable
    private var lock: NSRecursiveLock { GraphLock.shared }

    struct TensorInfo: Decodable { let name: String; let shape: [Int]; let offset: Int; let count: Int }
    struct Header: Decodable { let config: [String: Int]; let tensors: [TensorInfo] }
    /// The shipped model's header (Resources/weights.json): only the architecture, its tensors being
    /// read out of the compiled Core ML model.
    struct ModelHeader: Decodable { let config: [String: Int] }

    /// Stored tensors: fp16 in the shipped file, fp32 in a tuned file; both read into fp16 Data.
    public struct Weights {
        public var config: [String: Int]
        public var tensors: [String: (shape: [Int], half: Data)]

        public init(path: String) throws {
            let header = try JSONDecoder().decode(Header.self, from: Data(contentsOf: URL(fileURLWithPath: path + ".json")))
            let blob = try Data(contentsOf: URL(fileURLWithPath: path + ".bin"))
            let isHalf = header.config["dtype_bits"] ?? 16 == 16
            config = header.config
            tensors = [:]
            for info in header.tensors {
                let width = isHalf ? 2 : 4
                let end = (info.offset + info.count) * width
                guard info.offset >= 0, info.count >= 0, end <= blob.count, info.shape.reduce(1, *) == info.count else {
                    throw NSError(domain: "GraphScorer", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(path).bin: \(info.name) does not fit the file"])
                }
                let slice = blob.subdata(in: info.offset * width ..< end)
                tensors[info.name] = (info.shape, isHalf ? slice : Self.toHalf(slice))
            }
        }

        static func toHalf(_ floats: Data) -> Data {
            let count = floats.count / 4
            var out = [Float16](repeating: 0, count: count)
            floats.withUnsafeBytes { raw in
                let f = raw.bindMemory(to: Float.self)
                for i in 0..<count { out[i] = Float16(f[i]) }
            }
            return Data(bytes: &out, count: count * 2)
        }

        /// Whether a tuned file belongs to this model (same architecture; a tuned file from an
        /// earlier model would silently replace the wrong blocks).
        public func matches(tuned: Weights) -> Bool {
            for key in ["n_layer", "d_model", "n_head", "vocab_size"] where config[key] != tuned.config[key] { return false }
            return true
        }

        /// The shipped weights as the input method carries them: the fp16 constants of the compiled
        /// Core ML model (model.mil names each tensor and its place in weights/weight.bin), the full
        /// position table (the 32-row model keeps only its first rows) and the architecture from
        /// the header file. One copy of the model in the bundle serves both runtimes and learning.
        public init(compiledModel: URL, positions: URL, header: URL) throws {
            let head = try JSONDecoder().decode(ModelHeader.self, from: Data(contentsOf: header))
            config = head.config
            let blob = try Data(contentsOf: compiledModel.appendingPathComponent("weights/weight.bin"), options: .mappedIfSafe)
            tensors = [:]
            for (milName, entry) in try TunedModel.constants(in: compiledModel) {
                guard milName.hasPrefix("model_"), milName.hasSuffix("_to_fp16") else { continue }
                var name = String(milName.dropFirst("model_".count).dropLast("_to_fp16".count))
                if name.hasPrefix("blocks_") {
                    let rest = name.dropFirst("blocks_".count)
                    guard let underscore = rest.firstIndex(of: "_") else { continue }
                    name = "blocks." + rest[..<underscore] + "." + rest[rest.index(after: underscore)...]
                }
                for suffix in ["_weight", "_bias"] where name.hasSuffix(suffix) { name = String(name.dropLast(suffix.count)) + "." + suffix.dropFirst() }
                if name == "positions.weight" || name == "causal" { continue }
                guard entry.offset + 24 <= blob.count else { throw NSError(domain: "GraphScorer", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(milName): entry outside weight.bin"]) }
                let size = Int(blob.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: entry.offset + 8, as: UInt64.self) })
                let start = Int(blob.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: entry.offset + 16, as: UInt64.self) })
                guard start + size <= blob.count, size == entry.shape.reduce(1, *) * 2 else {
                    throw NSError(domain: "GraphScorer", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(milName): \(size) bytes for shape \(entry.shape)"])
                }
                tensors[name] = (entry.shape, blob.subdata(in: start ..< start + size))
            }
            let table = try Data(contentsOf: positions)
            guard let d = config["d_model"], let rows = config["block_size"], table.count == rows * d * 2 else {
                throw NSError(domain: "GraphScorer", code: 2, userInfo: [NSLocalizedDescriptionKey: "positions table does not match the model"])
            }
            tensors["positions.weight"] = ([rows, d], table)
            guard tensors["tokens.weight"] != nil, tensors["ln.weight"] != nil, tensors["blocks.0.qkv.weight"] != nil else {
                throw NSError(domain: "GraphScorer", code: 2, userInfo: [NSLocalizedDescriptionKey: "compiled model has no language-model constants"])
            }
        }

        /// Replaces tensors with those of a tuned file (same model, same names and shapes).
        public mutating func apply(tuned: Weights) {
            guard matches(tuned: tuned) else { return }
            for (name, tensor) in tuned.tensors where tensors[name]?.shape == tensor.shape {
                tensors[name] = tensor
            }
        }
    }

    public init(weights: Weights, vocabularyURL: URL, rowLength: Int = 32, batchSize: Int = NeuralScorer.defaultBatchSize) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw GraphError.noDevice }
        self.device = device
        self.queue = queue
        graphDevice = MPSGraphDevice(mtlDevice: device)
        self.rowLength = rowLength
        self.batchSize = batchSize
        let vocabulary = try JSONDecoder().decode([String].self, from: Data(contentsOf: vocabularyURL))
        tokens = vocabulary.map { $0.count == 1 ? Character($0) : "\u{FFFD}" }
        var ids: [Character: Int32] = [:]
        for (index, token) in vocabulary.enumerated() where token.count == 1 { ids[Character(token)] = Int32(index) }
        self.ids = ids

        guard let d = weights.config["d_model"], let heads = weights.config["n_head"], let layers = weights.config["n_layer"] else { throw GraphError.badHeader }
        let headDim = d / heads
        let graph = MPSGraph()
        func shape(_ s: [Int]) -> [NSNumber] { s.map { NSNumber(value: $0) } }
        func constant(_ name: String, rowsLimit: Int? = nil) throws -> MPSGraphTensor {
            guard let tensor = weights.tensors[name] else { throw GraphError.missingTensor(name) }
            var s = tensor.shape
            var data = tensor.half
            if let rowsLimit, s.count == 2, s[0] > rowsLimit {
                data = data.subdata(in: 0 ..< rowsLimit * s[1] * 2)
                s[0] = rowsLimit
            }
            return graph.constant(data, shape: shape(s), dataType: .float16)
        }
        func matrix(_ name: String) throws -> MPSGraphTensor {
            graph.transposeTensor(try constant(name), dimension: 0, withDimension: 1, name: nil)
        }
        func layerNorm(_ x: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            let mean = graph.mean(of: x, axes: [2], name: nil)
            let variance = graph.variance(of: x, mean: mean, axes: [2], name: nil)
            let gamma = graph.reshape(try constant(prefix + ".weight"), shape: [1, 1, NSNumber(value: d)], name: nil)
            let beta = graph.reshape(try constant(prefix + ".bias"), shape: [1, 1, NSNumber(value: d)], name: nil)
            return graph.normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        }
        func linear(_ x: MPSGraphTensor, _ prefix: String, out: Int) throws -> MPSGraphTensor {
            let y = graph.matrixMultiplication(primary: x, secondary: try matrix(prefix + ".weight"), name: nil)
            return graph.addition(y, graph.reshape(try constant(prefix + ".bias"), shape: [1, 1, NSNumber(value: out)], name: nil), name: nil)
        }
        func scalar(_ v: Double) -> MPSGraphTensor { graph.constant(v, dataType: .float16) }

        var maskValues = [Float16](repeating: 0, count: rowLength * rowLength)
        for i in 0..<rowLength { for j in (i + 1)..<rowLength where j < rowLength { maskValues[i * rowLength + j] = -1e4 } }
        let mask = graph.constant(Data(bytes: &maskValues, count: maskValues.count * 2), shape: shape([rowLength, rowLength]), dataType: .float16)

        let idsInput = graph.placeholder(shape: shape([batchSize, rowLength]), dataType: .int32, name: "ids")
        let tokenTable = try constant("tokens.weight")
        var x = graph.gather(withUpdatesTensor: tokenTable, indicesTensor: idsInput, axis: 0, batchDimensions: 0, name: nil)
        x = graph.addition(x, try constant("positions.weight", rowsLimit: rowLength), name: nil)
        for layer in 0..<layers {
            let p = "blocks.\(layer)"
            let qkv = try linear(try layerNorm(x, p + ".ln1"), p + ".qkv", out: 3 * d)
            let parts = graph.split(qkv, splitSizes: [NSNumber(value: d), NSNumber(value: d), NSNumber(value: d)], axis: 2, name: nil)
            func headsFirst(_ t: MPSGraphTensor) -> MPSGraphTensor {
                graph.transposeTensor(graph.reshape(t, shape: shape([batchSize, rowLength, heads, headDim]), name: nil), dimension: 1, withDimension: 2, name: nil)
            }
            let q = headsFirst(parts[0]), k = headsFirst(parts[1]), v = headsFirst(parts[2])
            var scores = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
            scores = graph.addition(graph.multiplication(scores, scalar(1.0 / sqrt(Double(headDim))), name: nil), mask, name: nil)
            let attended = graph.matrixMultiplication(primary: graph.softMax(with: scores, axis: 3, name: nil), secondary: v, name: nil)
            let merged = graph.reshape(graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil), shape: shape([batchSize, rowLength, d]), name: nil)
            x = graph.addition(x, try linear(merged, p + ".proj", out: d), name: nil)
            let hidden = try linear(try layerNorm(x, p + ".ln2"), p + ".fc", out: 4 * d)
            let erf = graph.erf(with: graph.multiplication(hidden, scalar(1.0 / sqrt(2.0)), name: nil), name: nil)
            let gelu = graph.multiplication(graph.multiplication(hidden, scalar(0.5), name: nil), graph.addition(erf, scalar(1.0), name: nil), name: nil)
            x = graph.addition(x, try linear(gelu, p + ".out", out: d), name: nil)
        }
        x = try layerNorm(x, "ln")
        let logits = graph.cast(graph.matrixMultiplication(primary: x, secondary: graph.transposeTensor(tokenTable, dimension: 0, withDimension: 1, name: nil), name: nil), to: .float32, name: nil)
        let top = graph.reductionMaximum(with: logits, axis: 2, name: nil)
        let shifted = graph.subtraction(logits, top, name: nil)
        let logSum = graph.logarithm(with: graph.reductionSum(with: graph.exponent(with: shifted, name: nil), axis: 2, name: nil), name: nil)
        let logp = graph.subtraction(shifted, logSum, name: nil)  // [B, T, V]
        // target_logp[b, t] = log P(ids[b, t+1] | ids[b, ..t]) for t < T-1.
        let nextIds = graph.sliceTensor(idsInput, dimension: 1, start: 1, length: rowLength - 1, name: nil)
        let gathered = graph.gatherAlongAxis(2, updates: graph.sliceTensor(logp, dimension: 1, start: 0, length: rowLength - 1, name: nil),
                                             indices: graph.expandDims(nextIds, axis: 2, name: nil), name: nil)
        let targetOut = graph.reshape(gathered, shape: shape([batchSize, rowLength - 1]), name: nil)

        GraphLock.shared.lock()
        defer { GraphLock.shared.unlock() }
        let idsType = MPSGraphShapedType(shape: shape([batchSize, rowLength]), dataType: .int32)
        targetExecutable = graph.compile(with: graphDevice, feeds: [idsInput: idsType], targetTensors: [targetOut], targetOperations: nil, compilationDescriptor: GraphLock.gpuOnly)
        // First runs build the GPU pipelines (hundreds of ms): do it now, not on the first keystroke.
        _ = targetLogp(rows: [[1, 2]])
    }

    public convenience init(weightsPath: String, tunedPath: String? = nil, vocabularyURL: URL, rowLength: Int = 32) throws {
        try self.init(source: .files(base: weightsPath), tunedPath: tunedPath, vocabularyURL: vocabularyURL, rowLength: rowLength)
    }

    public convenience init(source: WeightsSource, tunedPath: String? = nil, vocabularyURL: URL, rowLength: Int = 32) throws {
        var weights = try source.load()
        if let tunedPath, FileManager.default.fileExists(atPath: tunedPath + ".json") {
            weights.apply(tuned: try Weights(path: tunedPath))
        }
        try self.init(weights: weights, vocabularyURL: vocabularyURL, rowLength: rowLength)
    }

    /// Where the shipped weights come from: a weights.json/.bin pair (development), or the compiled
    /// Core ML model plus the position table and header the input method bundles.
    public enum WeightsSource {
        case files(base: String)
        case compiledModel(URL, positions: URL, header: URL)
        public func load() throws -> Weights {
            switch self {
            case .files(let base): return try Weights(path: base)
            case .compiledModel(let model, let positions, let header): return try Weights(compiledModel: model, positions: positions, header: header)
            }
        }
    }

    // MARK: - Tokens

    public func token(for character: Character) -> Int32? { ids[character] }
    public func character(for token: Int32) -> Character { tokens.indices.contains(Int(token)) ? tokens[Int(token)] : "\u{FFFD}" }
    public func encode(_ text: String) -> [Int32] {
        text.map { ids[$0] ?? $0.lowercased().first.flatMap { ids[$0] } ?? 2 }
    }
    public func contextTokens(_ context: String, reserving reserved: Int) -> [Int32] {
        let room = max(1, rowLength - reserved)
        return Array(encode(context).suffix(room))
    }

    // MARK: - Running

    private func tensorData(_ values: [Int32], _ shape: [Int]) -> MPSGraphTensorData {
        var v = values
        return MPSGraphTensorData(device: graphDevice, data: Data(bytes: &v, count: v.count * 4), shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    }

    /// target_logp rows for up to batchSize rows (each ≤ rowLength tokens), padded.
    private func targetLogp(rows: [[Int32]]) -> [[Float]]? {
        guard !rows.isEmpty, rows.count <= batchSize, rows.allSatisfy({ !$0.isEmpty && $0.count <= rowLength }) else { return nil }
        var flat = [Int32](repeating: 0, count: batchSize * rowLength)
        for (r, row) in rows.enumerated() { for (t, v) in row.enumerated() { flat[r * rowLength + t] = v } }
        lock.lock(); defer { lock.unlock() }
        let results = targetExecutable.run(with: queue, inputs: [tensorData(flat, [batchSize, rowLength])], results: nil, executionDescriptor: nil)
        var values = [Float](repeating: 0, count: batchSize * (rowLength - 1))
        results[0].mpsndarray().readBytes(&values, strideBytes: nil)
        return rows.indices.map { r in Array(values[r * (rowLength - 1) ..< (r + 1) * (rowLength - 1)]) }
    }

    public func score(prefix: [Int32], candidates: [String]) -> [Double]? {
        let encoded = candidates.prefix(batchSize).map(encode)
        guard let longest = encoded.map(\.count).max(), longest > 0, longest < rowLength, !prefix.isEmpty else { return nil }
        let head = Array(prefix.suffix(rowLength - longest))
        guard let output = targetLogp(rows: encoded.map { head + $0 }) else { return nil }
        return encoded.enumerated().map { r, candidate in
            (0..<candidate.count).reduce(0.0) { $0 + Double(output[r][head.count - 1 + $1]) }
        }
    }

}
