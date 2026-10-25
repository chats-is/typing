import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Learns from the learning records on this machine: the language model's top blocks are tuned with
/// MPSGraph (automatic differentiation + Adam on the GPU) while everything below stays as shipped.
/// Verified against PyTorch (Training/personalize_eval.py): identical perplexity after the same
/// steps. The result is a set of tensors that `GraphScorer` applies over the shipped weights.
public final class GraphTrainer {
    public enum TrainerError: Error { case noDevice, badHeader, missingTensor(String), noGradient(String) }

    public let rows: Int
    public let batch: Int
    public let trainableNames: [String]
    private let graph = MPSGraph()
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice
    private let ids: MPSGraphTensor, targets: MPSGraphTensor
    private let learningRate: MPSGraphTensor, beta1Power: MPSGraphTensor, beta2Power: MPSGraphTensor
    private let loss: MPSGraphTensor
    private let updates: [MPSGraphOperation]
    private var variables: [(name: String, tensor: MPSGraphTensor, shape: [Int])] = []
    /// The frozen weights, fed from GPU buffers (as in GraphScorer: constants are written to disk).
    private let frozen: [MPSGraphTensor: MPSGraphTensorData]
    private var stepCount = 0
    private var stepExecutable: MPSGraphExecutable?
    private var readExecutable: MPSGraphExecutable?
    private var reads: [MPSGraphTensor] = []

    /// `weights`: the shipped tensors (fp16) with any earlier tuned tensors already applied, so a
    /// new session continues from them. `top`: how many blocks from the top are tuned.
    public init(weights: GraphScorer.Weights, top: Int = 2, rows: Int = 64, batch: Int = 16) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw TrainerError.noDevice }
        self.device = device; self.queue = queue; graphDevice = MPSGraphDevice(mtlDevice: device)
        self.rows = rows; self.batch = batch
        guard let d = weights.config["d_model"], let heads = weights.config["n_head"], let layers = weights.config["n_layer"], let vocabSize = weights.config["vocab_size"] else { throw TrainerError.badHeader }
        let headDim = d / heads
        let graph = self.graph
        func shape(_ s: [Int]) -> [NSNumber] { s.map { NSNumber(value: $0) } }
        func floats(_ name: String) throws -> (Data, [Int]) {
            guard let tensor = weights.tensors[name] else { throw TrainerError.missingTensor(name) }
            let count = tensor.half.count / 2
            var out = [Float](repeating: 0, count: count)
            tensor.half.withUnsafeBytes { raw in
                let h = raw.bindMemory(to: UInt16.self)
                for i in 0..<count { out[i] = Float(Float16(bitPattern: h[i])) }
            }
            return (Data(bytes: &out, count: count * 4), tensor.shape)
        }
        var variables: [(name: String, tensor: MPSGraphTensor, shape: [Int])] = []
        var frozen: [MPSGraphTensor: MPSGraphTensorData] = [:]
        func fed(_ data: Data, _ s: [Int], _ name: String) throws -> MPSGraphTensor {
            let input = graph.placeholder(shape: shape(s), dataType: .float32, name: name)
            guard let buffer = data.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else { throw TrainerError.noDevice }
            frozen[input] = MPSGraphTensorData(buffer, shape: shape(s), dataType: .float32)
            return input
        }
        func parameter(_ name: String, trainable: Bool) throws -> MPSGraphTensor {
            let (data, s) = try floats(name)
            if trainable {
                let v = graph.variable(with: data, shape: shape(s), dataType: .float32, name: name)
                variables.append((name, v, s))
                return graph.read(v, name: nil)
            }
            return try fed(data, s, name)
        }
        func matrix(_ name: String, trainable: Bool) throws -> MPSGraphTensor {
            graph.transposeTensor(try parameter(name, trainable: trainable), dimension: 0, withDimension: 1, name: nil)
        }
        func layerNorm(_ x: MPSGraphTensor, _ prefix: String, trainable: Bool) throws -> MPSGraphTensor {
            let mean = graph.mean(of: x, axes: [2], name: nil)
            let variance = graph.variance(of: x, mean: mean, axes: [2], name: nil)
            let gamma = graph.reshape(try parameter(prefix + ".weight", trainable: trainable), shape: [1, 1, NSNumber(value: d)], name: nil)
            let beta = graph.reshape(try parameter(prefix + ".bias", trainable: trainable), shape: [1, 1, NSNumber(value: d)], name: nil)
            return graph.normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        }
        func linear(_ x: MPSGraphTensor, _ prefix: String, out: Int, trainable: Bool) throws -> MPSGraphTensor {
            let y = graph.matrixMultiplication(primary: x, secondary: try matrix(prefix + ".weight", trainable: trainable), name: nil)
            return graph.addition(y, graph.reshape(try parameter(prefix + ".bias", trainable: trainable), shape: [1, 1, NSNumber(value: out)], name: nil), name: nil)
        }
        func c(_ v: Double) -> MPSGraphTensor { graph.constant(v, dataType: .float32) }
        /// erf by Abramowitz–Stegun 7.1.26 from elementary ops, so it differentiates.
        func erf(_ x: MPSGraphTensor) -> MPSGraphTensor {
            let sign = graph.sign(with: x, name: nil)
            let a = graph.absolute(with: x, name: nil)
            let t = graph.division(c(1), graph.addition(c(1), graph.multiplication(c(0.3275911), a, name: nil), name: nil), name: nil)
            var poly = c(1.061405429)
            for coefficient in [-1.453152027, 1.421413741, -0.284496736, 0.254829592] {
                poly = graph.addition(graph.multiplication(poly, t, name: nil), c(coefficient), name: nil)
            }
            poly = graph.multiplication(poly, t, name: nil)
            let y = graph.subtraction(c(1), graph.multiplication(poly, graph.exponent(with: graph.negative(with: graph.multiplication(a, a, name: nil), name: nil), name: nil), name: nil), name: nil)
            return graph.multiplication(sign, y, name: nil)
        }

        var maskValues = [Float](repeating: 0, count: rows * rows)
        for i in 0..<rows { for j in (i + 1)..<rows where j < rows { maskValues[i * rows + j] = -1e4 } }
        let mask = graph.constant(Data(bytes: &maskValues, count: rows * rows * 4), shape: shape([rows, rows]), dataType: .float32)
        let ids = graph.placeholder(shape: shape([batch, rows]), dataType: .int32, name: "ids")
        let targets = graph.placeholder(shape: shape([batch, rows]), dataType: .int32, name: "targets")
        let tokenTable = try parameter("tokens.weight", trainable: false)
        var x = graph.gather(withUpdatesTensor: tokenTable, indicesTensor: ids, axis: 0, batchDimensions: 0, name: nil)
        let (positionData, _) = try floats("positions.weight")
        x = graph.addition(x, try fed(positionData.subdata(in: 0 ..< rows * d * 4), [rows, d], "positions.weight"), name: nil)
        for layer in 0..<layers {
            let trainable = layer >= layers - top
            let p = "blocks.\(layer)"
            let qkv = try linear(try layerNorm(x, p + ".ln1", trainable: trainable), p + ".qkv", out: 3 * d, trainable: trainable)
            let parts = graph.split(qkv, splitSizes: [NSNumber(value: d), NSNumber(value: d), NSNumber(value: d)], axis: 2, name: nil)
            func headsFirst(_ t: MPSGraphTensor) -> MPSGraphTensor {
                graph.transposeTensor(graph.reshape(t, shape: shape([batch, rows, heads, headDim]), name: nil), dimension: 1, withDimension: 2, name: nil)
            }
            let q = headsFirst(parts[0]), k = headsFirst(parts[1]), v = headsFirst(parts[2])
            var scores = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
            scores = graph.addition(graph.multiplication(scores, c(1.0 / sqrt(Double(headDim))), name: nil), mask, name: nil)
            let attended = graph.matrixMultiplication(primary: graph.softMax(with: scores, axis: 3, name: nil), secondary: v, name: nil)
            let merged = graph.reshape(graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil), shape: shape([batch, rows, d]), name: nil)
            x = graph.addition(x, try linear(merged, p + ".proj", out: d, trainable: trainable), name: nil)
            let hidden = try linear(try layerNorm(x, p + ".ln2", trainable: trainable), p + ".fc", out: 4 * d, trainable: trainable)
            let gelu = graph.multiplication(graph.multiplication(hidden, c(0.5), name: nil), graph.addition(erf(graph.multiplication(hidden, c(1.0 / sqrt(2.0)), name: nil)), c(1.0), name: nil), name: nil)
            x = graph.addition(x, try linear(gelu, p + ".out", out: d, trainable: trainable), name: nil)
        }
        x = try layerNorm(x, "ln", trainable: true)
        let logits = graph.matrixMultiplication(primary: x, secondary: graph.transposeTensor(tokenTable, dimension: 0, withDimension: 1, name: nil), name: nil)
        let labels = graph.oneHot(withIndicesTensor: targets, depth: vocabSize, axis: 2, dataType: .float32, name: nil)
        let perPosition = graph.reshape(graph.softMaxCrossEntropy(logits, labels: labels, axis: 2, reuctionType: .none, name: nil), shape: shape([batch, rows]), name: nil)
        let valid = graph.cast(graph.notEqual(targets, graph.constant(0, dataType: .int32), name: nil), to: .float32, name: nil)
        let count = graph.maximum(graph.reductionSum(with: valid, axes: [0, 1], name: nil), c(1), name: nil)
        let loss = graph.division(graph.reductionSum(with: graph.multiplication(perPosition, valid, name: nil), axes: [0, 1], name: nil), count, name: nil)

        let gradients = graph.gradients(of: loss, with: variables.map(\.tensor), name: nil)
        let learningRate = graph.placeholder(shape: [1], dataType: .float32, name: "lr")
        let beta1Power = graph.placeholder(shape: [1], dataType: .float32, name: "b1p")
        let beta2Power = graph.placeholder(shape: [1], dataType: .float32, name: "b2p")
        var updates: [MPSGraphOperation] = []
        for (name, variable, s) in variables {
            guard let gradient = gradients[variable] else { throw TrainerError.noGradient(name) }
            let zeros = Data(count: s.reduce(1, *) * 4)
            let momentum = graph.variable(with: zeros, shape: shape(s), dataType: .float32, name: name + ".m")
            let velocity = graph.variable(with: zeros, shape: shape(s), dataType: .float32, name: name + ".v")
            let result = graph.adam(learningRate: learningRate, beta1: c(0.9), beta2: c(0.999), epsilon: c(1e-8), beta1Power: beta1Power, beta2Power: beta2Power,
                                    values: graph.read(variable, name: nil), momentum: graph.read(momentum, name: nil), velocity: graph.read(velocity, name: nil),
                                    maximumVelocity: nil, gradient: gradient, name: nil)
            updates.append(graph.assign(variable, tensor: result[0], name: nil))
            updates.append(graph.assign(momentum, tensor: result[1], name: nil))
            updates.append(graph.assign(velocity, tensor: result[2], name: nil))
        }
        self.ids = ids; self.targets = targets; self.learningRate = learningRate; self.beta1Power = beta1Power; self.beta2Power = beta2Power
        self.loss = loss; self.updates = updates; self.variables = variables; self.frozen = frozen
        trainableNames = variables.map(\.name)
    }

    public var trainableParameterCount: Int { variables.map { $0.shape.reduce(1, *) }.reduce(0, +) }

    private func data(_ values: [Int32], _ shape: [Int]) -> MPSGraphTensorData {
        var v = values
        return MPSGraphTensorData(device: graphDevice, data: Data(bytes: &v, count: v.count * 4), shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    }
    private func data(_ values: [Float], _ shape: [Int]) -> MPSGraphTensorData {
        var v = values
        return MPSGraphTensorData(device: graphDevice, data: Data(bytes: &v, count: v.count * 4), shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    /// One Adam step on `batch` windows of `rows` + 1 tokens taken from `stream` at `starts`;
    /// returns the loss.
    public func step(stream: [Int32], starts: [Int], learningRate lr: Float) -> Float {
        GraphLock.shared.lock()
        defer { GraphLock.shared.unlock() }
        var idsBatch = [Int32](repeating: 0, count: batch * rows), targetBatch = [Int32](repeating: 0, count: batch * rows)
        for (b, start) in starts.prefix(batch).enumerated() {
            for t in 0..<rows where start + t + 1 < stream.count {
                idsBatch[b * rows + t] = stream[start + t]
                targetBatch[b * rows + t] = stream[start + t + 1]
            }
        }
        stepCount += 1
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            ids: data(idsBatch, [batch, rows]), targets: data(targetBatch, [batch, rows]),
            learningRate: data([lr], [1]), beta1Power: data([pow(0.9, Float(stepCount))], [1]), beta2Power: data([pow(0.999, Float(stepCount))], [1]),
        ]
        feeds.merge(frozen) { a, _ in a }
        if stepExecutable == nil {
            var types: [MPSGraphTensor: MPSGraphShapedType] = [
                ids: MPSGraphShapedType(shape: [NSNumber(value: batch), NSNumber(value: rows)], dataType: .int32),
                targets: MPSGraphShapedType(shape: [NSNumber(value: batch), NSNumber(value: rows)], dataType: .int32),
                learningRate: MPSGraphShapedType(shape: [1], dataType: .float32),
                beta1Power: MPSGraphShapedType(shape: [1], dataType: .float32),
                beta2Power: MPSGraphShapedType(shape: [1], dataType: .float32),
            ]
            for (tensor, value) in frozen { types[tensor] = MPSGraphShapedType(shape: value.shape, dataType: value.dataType) }
            stepExecutable = graph.compile(with: graphDevice, feeds: types, targetTensors: [loss], targetOperations: updates, compilationDescriptor: GraphLock.gpuOnly)
        }
        guard let executable = stepExecutable, let order = executable.feedTensors else { return .nan }
        let results = executable.run(with: queue, inputs: order.compactMap { feeds[$0] }, results: nil, executionDescriptor: nil)
        var value: Float = 0
        results[0].mpsndarray().readBytes(&value, strideBytes: nil)
        return value
    }

    /// The tuned tensors as they are now, written as <base>.json + <base>.bin (fp32), the form
    /// `GraphScorer.Weights(path:)` reads and applies over the shipped weights.
    public func save(to base: String, config: [String: Int]) throws {
        GraphLock.shared.lock()
        defer { GraphLock.shared.unlock() }
        if readExecutable == nil {
            reads = variables.map { graph.read($0.tensor, name: nil) }
            readExecutable = graph.compile(with: graphDevice, feeds: [:], targetTensors: reads, targetOperations: nil, compilationDescriptor: GraphLock.gpuOnly)
        }
        guard let executable = readExecutable else { return }
        let output = executable.run(with: queue, inputs: [], results: nil, executionDescriptor: nil)
        var results: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (tensor, data) in zip(executable.targetTensors ?? reads, output) { results[tensor] = data }
        var blob = Data()
        var tensors: [[String: Any]] = []
        var offset = 0
        for ((name, _, shape), read) in zip(variables, reads) {
            let count = shape.reduce(1, *)
            var values = [Float](repeating: 0, count: count)
            results[read]!.mpsndarray().readBytes(&values, strideBytes: nil)
            blob.append(Data(bytes: &values, count: count * 4))
            tensors.append(["name": name, "shape": shape, "offset": offset, "count": count])
            offset += count
        }
        var header = config
        header["dtype_bits"] = 32
        let json = try JSONSerialization.data(withJSONObject: ["config": header, "tensors": tensors])
        try blob.write(to: URL(fileURLWithPath: base + ".bin"))
        try json.write(to: URL(fileURLWithPath: base + ".json"))
    }
}
