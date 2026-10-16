// Prototype of on-device learning: tune the language model's top blocks on a person's text with
// MPSGraph (automatic differentiation + Adam on the GPU), the frozen lower blocks kept as constants.
//
//     mpstrain Data/lm9/weights Data/lm9/vocab.json --train Data/personal/forum2.train.txt \
//              --eval Data/personal/forum2.lattice.tsv [--steps 200] [--lr 2e-5] [--top 2]
//
// Prints the perplexity of the person's later sentences before and after, so the run can be
// checked against Training/personalize_eval.py (PyTorch: forum2 28.6 → 19.5 at 200 steps, top2).
import Foundation
import Metal
import MetalPerformanceShadersGraph

var weightsPath = "", vocabPath = "", trainPath = "", evalPath = ""
var steps = 200, batch = 16, rows = 64, top = 2
var lr: Float = 2e-5
var arguments = CommandLine.arguments.dropFirst()[...]
var positional: [String] = []
while let a = arguments.popFirst() {
    switch a {
    case "--train": trainPath = arguments.popFirst() ?? ""
    case "--eval": evalPath = arguments.popFirst() ?? ""
    case "--steps": steps = Int(arguments.popFirst() ?? "200") ?? 200
    case "--batch": batch = Int(arguments.popFirst() ?? "16") ?? 16
    case "--lr": lr = Float(arguments.popFirst() ?? "2e-5") ?? 2e-5
    case "--top": top = Int(arguments.popFirst() ?? "2") ?? 2
    default: positional.append(a)
    }
}
guard positional.count == 2, !trainPath.isEmpty, !evalPath.isEmpty else {
    print("usage: mpstrain weights vocab.json --train text --eval lattice.tsv [--steps N --lr x --top k]"); exit(2)
}
weightsPath = positional[0]; vocabPath = positional[1]

struct TensorInfo: Decodable { let name: String; let shape: [Int]; let offset: Int; let count: Int }
struct Header: Decodable { let config: [String: Int]; let tensors: [TensorInfo] }
let header = try JSONDecoder().decode(Header.self, from: Data(contentsOf: URL(fileURLWithPath: weightsPath + ".json")))
let blob = try Data(contentsOf: URL(fileURLWithPath: weightsPath + ".bin"))
let d = header.config["d_model"]!, heads = header.config["n_head"]!, layers = header.config["n_layer"]!, vocabSize = header.config["vocab_size"]!
let headDim = d / heads
let byName = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
let vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: vocabPath)))
var tokenIds: [Character: Int32] = [:]
for (i, t) in vocab.enumerated() where t.count == 1 { tokenIds[Character(t)] = Int32(i) }
func encode(_ text: String) -> [Int32] { text.map { tokenIds[$0] ?? tokenIds[Character($0.lowercased())] ?? 2 } }

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let graph = MPSGraph()

/// fp32 data of a stored tensor.
func floats(_ name: String) -> (Data, [Int]) {
    let info = byName[name]!
    let halves = blob.subdata(in: info.offset * 2 ..< (info.offset + info.count) * 2)
    var out = [Float](repeating: 0, count: info.count)
    halves.withUnsafeBytes { raw in
        let h = raw.bindMemory(to: UInt16.self)
        for i in 0..<info.count { out[i] = Float(Float16(bitPattern: h[i])) }
    }
    return (Data(bytes: &out, count: info.count * 4), info.shape)
}
func shape(_ s: [Int]) -> [NSNumber] { s.map { NSNumber(value: $0) } }

var variables: [(name: String, tensor: MPSGraphTensor, shape: [Int])] = []
func parameter(_ name: String, trainable: Bool) -> MPSGraphTensor {
    let (data, s) = floats(name)
    if trainable {
        let v = graph.variable(with: data, shape: shape(s), dataType: .float32, name: name)
        variables.append((name, v, s))
        return graph.read(v, name: nil)
    }
    return graph.constant(data, shape: shape(s), dataType: .float32)
}
func matrix(_ name: String, trainable: Bool) -> MPSGraphTensor {  // PyTorch Linear [out, in] -> [in, out]
    graph.transposeTensor(parameter(name, trainable: trainable), dimension: 0, withDimension: 1, name: nil)
}
func layerNorm(_ x: MPSGraphTensor, _ prefix: String, trainable: Bool) -> MPSGraphTensor {
    let mean = graph.mean(of: x, axes: [2], name: nil)
    let variance = graph.variance(of: x, mean: mean, axes: [2], name: nil)
    let gamma = graph.reshape(parameter(prefix + ".weight", trainable: trainable), shape: [1, 1, NSNumber(value: d)], name: nil)
    let beta = graph.reshape(parameter(prefix + ".bias", trainable: trainable), shape: [1, 1, NSNumber(value: d)], name: nil)
    return graph.normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
}
func linear(_ x: MPSGraphTensor, _ prefix: String, out: Int, trainable: Bool) -> MPSGraphTensor {
    let y = graph.matrixMultiplication(primary: x, secondary: matrix(prefix + ".weight", trainable: trainable), name: nil)
    return graph.addition(y, graph.reshape(parameter(prefix + ".bias", trainable: trainable), shape: [1, 1, NSNumber(value: out)], name: nil), name: nil)
}

/// erf by Abramowitz–Stegun 7.1.26 (error < 1.5e-7), from elementary ops so it differentiates.
func erfApprox(_ x: MPSGraphTensor) -> MPSGraphTensor {
    func c(_ v: Double) -> MPSGraphTensor { graph.constant(v, dataType: .float32) }
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
let tokenTable = parameter("tokens.weight", trainable: false)  // embeddings stay (tied with the head)
var x = graph.gather(withUpdatesTensor: tokenTable, indicesTensor: ids, axis: 0, batchDimensions: 0, name: nil)
let (positionData, positionShape) = floats("positions.weight")
let positions = graph.constant(positionData.subdata(in: 0 ..< rows * d * 4), shape: shape([rows, d]), dataType: .float32)
_ = positionShape
x = graph.addition(x, positions, name: nil)
let scale = graph.constant(1.0 / sqrt(Double(headDim)), dataType: .float32)
for layer in 0..<layers {
    let trainable = layer >= layers - top
    let p = "blocks.\(layer)"
    let qkv = linear(layerNorm(x, p + ".ln1", trainable: trainable), p + ".qkv", out: 3 * d, trainable: trainable)
    let parts = graph.split(qkv, splitSizes: [NSNumber(value: d), NSNumber(value: d), NSNumber(value: d)], axis: 2, name: nil)
    func headsFirst(_ t: MPSGraphTensor) -> MPSGraphTensor {
        graph.transposeTensor(graph.reshape(t, shape: shape([batch, rows, heads, headDim]), name: nil), dimension: 1, withDimension: 2, name: nil)
    }
    let q = headsFirst(parts[0]), k = headsFirst(parts[1]), v = headsFirst(parts[2])
    var scores = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
    scores = graph.addition(graph.multiplication(scores, scale, name: nil), mask, name: nil)
    let attended = graph.matrixMultiplication(primary: graph.softMax(with: scores, axis: 3, name: nil), secondary: v, name: nil)
    let merged = graph.reshape(graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil), shape: shape([batch, rows, d]), name: nil)
    x = graph.addition(x, linear(merged, p + ".proj", out: d, trainable: trainable), name: nil)
    let hidden = linear(layerNorm(x, p + ".ln2", trainable: trainable), p + ".fc", out: 4 * d, trainable: trainable)
    let erf = erfApprox(graph.multiplication(hidden, graph.constant(1.0 / sqrt(2.0), dataType: .float32), name: nil))
    let gelu = graph.multiplication(graph.multiplication(hidden, graph.constant(0.5, dataType: .float32), name: nil), graph.addition(erf, graph.constant(1.0, dataType: .float32), name: nil), name: nil)
    x = graph.addition(x, linear(gelu, p + ".out", out: d, trainable: trainable), name: nil)
}
x = layerNorm(x, "ln", trainable: true)
let logits = graph.matrixMultiplication(primary: x, secondary: graph.transposeTensor(tokenTable, dimension: 0, withDimension: 1, name: nil), name: nil)  // [B, T, V]
// Per-position −log P(target) from the built-in softmax cross-entropy (it has a gradient), masked
// where target == 0.
let labels = graph.oneHot(withIndicesTensor: targets, depth: vocabSize, axis: 2, dataType: .float32, name: nil)
let perPosition = graph.softMaxCrossEntropy(logits, labels: labels, axis: 2, reuctionType: .none, name: nil)  // [B, T]
let picked = graph.negative(with: graph.reshape(perPosition, shape: shape([batch, rows]), name: nil), name: nil)  // log P
let valid = graph.cast(graph.notEqual(targets, graph.constant(0, dataType: .int32), name: nil), to: .float32, name: nil)
let count = graph.reductionSum(with: valid, axes: [0, 1], name: nil)
let loss = graph.division(graph.negative(with: graph.reductionSum(with: graph.multiplication(picked, valid, name: nil), axes: [0, 1], name: nil), name: nil), count, name: nil)

// Adam on the trainable variables.
let trainableTensors = variables.map(\.tensor)
let gradients = graph.gradients(of: loss, with: trainableTensors, name: nil)
let learningRate = graph.placeholder(shape: [1], dataType: .float32, name: "lr")
let beta1 = graph.constant(0.9, dataType: .float32), beta2 = graph.constant(0.999, dataType: .float32), epsilon = graph.constant(1e-8, dataType: .float32)
let beta1Power = graph.placeholder(shape: [1], dataType: .float32, name: "b1p"), beta2Power = graph.placeholder(shape: [1], dataType: .float32, name: "b2p")
var updates: [MPSGraphOperation] = []
for (name, variable, s) in variables {
    guard let gradient = gradients[variable] else { print("no gradient for \(name)"); exit(1) }
    let zeros = Data(count: s.reduce(1, *) * 4)
    let momentum = graph.variable(with: zeros, shape: shape(s), dataType: .float32, name: name + ".m")
    let velocity = graph.variable(with: zeros, shape: shape(s), dataType: .float32, name: name + ".v")
    let result = graph.adam(learningRate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon, beta1Power: beta1Power, beta2Power: beta2Power,
                            values: graph.read(variable, name: nil), momentum: graph.read(momentum, name: nil), velocity: graph.read(velocity, name: nil),
                            maximumVelocity: nil, gradient: gradient, name: nil)
    updates.append(graph.assign(variable, tensor: result[0], name: nil))
    updates.append(graph.assign(momentum, tensor: result[1], name: nil))
    updates.append(graph.assign(velocity, tensor: result[2], name: nil))
}
print("trainable: \(variables.count) tensors, \(variables.map { $0.shape.reduce(1, *) }.reduce(0, +) / 1_000_000) M parameters; batch \(batch) × \(rows), lr \(lr), \(steps) steps")

// Data.
let stream = encode(try String(contentsOfFile: trainPath, encoding: .utf8))
struct Sentence { let expected: [Int32]; let context: [Int32] }
var evalSentences: [Sentence] = []
for line in try String(contentsOfFile: evalPath, encoding: .utf8).split(separator: "\n") {
    let f = line.split(separator: "\t", omittingEmptySubsequences: false)
    guard f.count >= 3 else { continue }
    evalSentences.append(Sentence(expected: encode(String(f[1])), context: encode(f[2].replacingOccurrences(of: "\\n", with: "\n"))))
}
let mpsDevice = MPSGraphDevice(mtlDevice: device)
func tensorData(_ values: [Int32], _ s: [Int]) -> MPSGraphTensorData {
    var v = values
    return MPSGraphTensorData(device: mpsDevice, data: Data(bytes: &v, count: v.count * 4), shape: shape(s), dataType: .int32)
}
func tensorData(_ values: [Float], _ s: [Int]) -> MPSGraphTensorData {
    var v = values
    return MPSGraphTensorData(device: mpsDevice, data: Data(bytes: &v, count: v.count * 4), shape: shape(s), dataType: .float32)
}

/// Perplexity of the person's later sentences given their context (targets only on the sentence).
func perplexity() -> Double {
    var total = 0.0, n = 0
    var i = 0
    while i < evalSentences.count {
        var idsBatch = [Int32](repeating: 0, count: batch * rows), targetBatch = [Int32](repeating: 0, count: batch * rows)
        let chunk = Array(evalSentences[i ..< min(i + batch, evalSentences.count)])
        for (b, s) in chunk.enumerated() {
            let head = Array(([1] + s.context).suffix(max(1, rows - s.expected.count)))
            let row = Array((head + s.expected).prefix(rows))
            for (t, v) in row.enumerated() { idsBatch[b * rows + t] = v }
            for t in head.count ..< row.count { targetBatch[b * rows + t - 1] = row[t] }
        }
        let results = graph.run(with: queue, feeds: [ids: tensorData(idsBatch, [batch, rows]), targets: tensorData(targetBatch, [batch, rows])], targetTensors: [picked], targetOperations: nil)
        var values = [Float](repeating: 0, count: batch * rows)
        results[picked]!.mpsndarray().readBytes(&values, strideBytes: nil)
        for b in 0..<chunk.count { for t in 0..<rows where targetBatch[b * rows + t] != 0 { total -= Double(values[b * rows + t]); n += 1 } }
        i += batch
    }
    return exp(total / Double(max(1, n)))
}

print(String(format: "before: perplexity %.2f", perplexity()))
var rng = SystemRandomNumberGenerator()
let started = Date()
for step in 1...steps {
    var idsBatch = [Int32](repeating: 0, count: batch * rows), targetBatch = [Int32](repeating: 0, count: batch * rows)
    for b in 0..<batch {
        let start = Int.random(in: 0 ..< stream.count - rows - 1, using: &rng)
        for t in 0..<rows { idsBatch[b * rows + t] = stream[start + t]; targetBatch[b * rows + t] = stream[start + t + 1] }
    }
    let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
        ids: tensorData(idsBatch, [batch, rows]), targets: tensorData(targetBatch, [batch, rows]),
        learningRate: tensorData([lr], [1]), beta1Power: tensorData([pow(0.9, Float(step))], [1]), beta2Power: tensorData([pow(0.999, Float(step))], [1]),
    ]
    let results = graph.run(with: queue, feeds: feeds, targetTensors: [loss], targetOperations: updates)
    if step % 50 == 0 || step == steps {
        var value: Float = 0
        results[loss]!.mpsndarray().readBytes(&value, strideBytes: nil)
        print(String(format: "step %4d  loss %.3f  (%.1f s)", step, value, Date().timeIntervalSince(started)))
    }
}
print(String(format: "after: perplexity %.2f  (%.0f ms per training step)", perplexity(), Date().timeIntervalSince(started) * 1000 / Double(steps)))
