// Prototype of the on-device runtime: the language model's forward pass on the GPU with MPSGraph
// (a system framework, no dependency), from the weights written by Training/export_weights.py.
//
//     mpsproto Data/lm9/weights [--rows 32] [--batch 16] [--fp16]
//
// Scores a fixed test row (context 今天天气, candidate 很好) so the number can be checked against
// PyTorch, and times repeated calls at the engine's batch. If this is as fast as Core ML, inference
// and the learning of the top blocks can share one runtime and tuned weights apply at once.
import Foundation
import Metal
import MetalPerformanceShadersGraph

var path = ""
var rows = 32
var batch = 16
var half = false
var arguments = CommandLine.arguments.dropFirst()[...]
while let a = arguments.popFirst() {
    switch a {
    case "--rows": rows = Int(arguments.popFirst() ?? "32") ?? 32
    case "--batch": batch = Int(arguments.popFirst() ?? "16") ?? 16
    case "--fp16": half = true
    default: path = a
    }
}
guard !path.isEmpty else { print("usage: mpsproto Data/lm9/weights"); exit(2) }

struct TensorInfo: Decodable { let name: String; let shape: [Int]; let offset: Int; let count: Int }
struct Header: Decodable { let config: [String: Int]; let tensors: [TensorInfo] }
let header = try JSONDecoder().decode(Header.self, from: Data(contentsOf: URL(fileURLWithPath: path + ".json")))
let blob = try Data(contentsOf: URL(fileURLWithPath: path + ".bin"))
let d = header.config["d_model"]!, heads = header.config["n_head"]!, layers = header.config["n_layer"]!, vocab = header.config["vocab_size"]!
let headDim = d / heads
let byName = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let graph = MPSGraph()
let compute: MPSDataType = half ? .float16 : .float32

/// A weight as a graph constant, optionally transposed (PyTorch Linear stores [out, in]).
func weight(_ name: String, transpose: Bool = false, rowsLimit: Int? = nil) -> MPSGraphTensor {
    let info = byName[name]!
    var shape = info.shape
    var slice = blob.subdata(in: info.offset * 2 ..< (info.offset + info.count) * 2)
    if let rowsLimit, shape.count == 2 {  // positions: first `rows` entries only
        slice = slice.subdata(in: 0 ..< rowsLimit * shape[1] * 2)
        shape[0] = rowsLimit
    }
    var tensor = graph.constant(slice, shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    if compute != .float16 { tensor = graph.cast(tensor, to: compute, name: nil) }
    if transpose { tensor = graph.transposeTensor(tensor, dimension: 0, withDimension: 1, name: nil) }
    return tensor
}

func layerNorm(_ x: MPSGraphTensor, _ prefix: String) -> MPSGraphTensor {
    let mean = graph.mean(of: x, axes: [2], name: nil)
    let variance = graph.variance(of: x, mean: mean, axes: [2], name: nil)
    let gamma = graph.reshape(weight(prefix + ".weight"), shape: [1, 1, NSNumber(value: d)], name: nil)
    let beta = graph.reshape(weight(prefix + ".bias"), shape: [1, 1, NSNumber(value: d)], name: nil)
    return graph.normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
}

func linear(_ x: MPSGraphTensor, _ prefix: String, out: Int) -> MPSGraphTensor {
    let y = graph.matrixMultiplication(primary: x, secondary: weight(prefix + ".weight", transpose: true), name: nil)
    let bias = graph.reshape(weight(prefix + ".bias"), shape: [1, 1, NSNumber(value: out)], name: nil)
    return graph.addition(y, bias, name: nil)
}

// Causal mask: 0 on and below the diagonal, -1e4 above (as in training).
var maskValues = [Float](repeating: 0, count: rows * rows)
for i in 0..<rows { for j in (i + 1)..<rows where j < rows { maskValues[i * rows + j] = -1e4 } }
var mask = graph.constant(Data(bytes: &maskValues, count: rows * rows * 4), shape: [NSNumber(value: rows), NSNumber(value: rows)], dataType: .float32)
if compute == .float16 { mask = graph.cast(mask, to: .float16, name: nil) }

let ids = graph.placeholder(shape: [NSNumber(value: batch), NSNumber(value: rows)], dataType: .int32, name: "ids")
let tokenTable = weight("tokens.weight")
var x = graph.gather(withUpdatesTensor: tokenTable, indicesTensor: ids, axis: 0, batchDimensions: 0, name: nil)  // [B, T, d]
x = graph.addition(x, weight("positions.weight", rowsLimit: rows), name: nil)
let scale = graph.constant(1.0 / sqrt(Double(headDim)), dataType: compute)
for layer in 0..<layers {
    let p = "blocks.\(layer)"
    let qkv = linear(layerNorm(x, p + ".ln1"), p + ".qkv", out: 3 * d)
    let parts = graph.split(qkv, splitSizes: [NSNumber(value: d), NSNumber(value: d), NSNumber(value: d)], axis: 2, name: nil)
    func headsFirst(_ t: MPSGraphTensor) -> MPSGraphTensor {
        let r = graph.reshape(t, shape: [NSNumber(value: batch), NSNumber(value: rows), NSNumber(value: heads), NSNumber(value: headDim)], name: nil)
        return graph.transposeTensor(r, dimension: 1, withDimension: 2, name: nil)  // [B, H, T, hd]
    }
    let q = headsFirst(parts[0]), k = headsFirst(parts[1]), v = headsFirst(parts[2])
    var scores = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
    scores = graph.addition(graph.multiplication(scores, scale, name: nil), mask, name: nil)
    let attended = graph.matrixMultiplication(primary: graph.softMax(with: scores, axis: 3, name: nil), secondary: v, name: nil)  // [B, H, T, hd]
    let merged = graph.reshape(graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil), shape: [NSNumber(value: batch), NSNumber(value: rows), NSNumber(value: d)], name: nil)
    x = graph.addition(x, linear(merged, p + ".proj", out: d), name: nil)
    let hidden = linear(layerNorm(x, p + ".ln2"), p + ".fc", out: 4 * d)
    // Exact GELU: 0.5 x (1 + erf(x / sqrt 2)).
    let erf = graph.erf(with: graph.multiplication(hidden, graph.constant(1.0 / sqrt(2.0), dataType: compute), name: nil), name: nil)
    let gelu = graph.multiplication(graph.multiplication(hidden, graph.constant(0.5, dataType: compute), name: nil), graph.addition(erf, graph.constant(1.0, dataType: compute), name: nil), name: nil)
    x = graph.addition(x, linear(gelu, p + ".out", out: d), name: nil)
}
x = layerNorm(x, "ln")
let logits = graph.matrixMultiplication(primary: x, secondary: graph.transposeTensor(tokenTable, dimension: 0, withDimension: 1, name: nil), name: nil)  // [B, T, V]
let logits32 = compute == .float32 ? logits : graph.cast(logits, to: .float32, name: nil)
// log-softmax written out around the row maximum.
let top = graph.reductionMaximum(with: logits32, axis: 2, name: nil)
let shifted = graph.subtraction(logits32, top, name: nil)
let logSum = graph.logarithm(with: graph.reductionSum(with: graph.exponent(with: shifted, name: nil), axis: 2, name: nil), name: nil)
let logp = graph.subtraction(shifted, logSum, name: nil)  // [B, T, V]
// log P of the token that follows each position: gather logp[b, t, ids[b, t+1]].
let nextIds = graph.sliceTensor(ids, dimension: 1, start: 1, length: rows - 1, name: nil)
let gathered = graph.gatherAlongAxis(2, updates: graph.sliceTensor(logp, dimension: 1, start: 0, length: rows - 1, name: nil),
                                     indices: graph.expandDims(nextIds, axis: 2, name: nil), name: nil)  // [B, T-1, 1]
let output = graph.reshape(gathered, shape: [NSNumber(value: batch), NSNumber(value: rows - 1)], name: nil)

let started = Date()
let executable = graph.compile(with: MPSGraphDevice(mtlDevice: device),
                               feeds: [ids: MPSGraphShapedType(shape: [NSNumber(value: batch), NSNumber(value: rows)], dataType: .int32)],
                               targetTensors: [output], targetOperations: nil, compilationDescriptor: nil)
print(String(format: "compiled in %.1f s (%@, %d layers, d %d, rows %d, batch %d)", Date().timeIntervalSince(started), half ? "fp16" : "fp32", layers, d, rows, batch))

// Test row: <bos> 今天天气 很好 (ids from Python), zero padded; the same row in every batch slot.
let testRow: [Int32] = CommandLine.arguments.contains("--row") ? [] : [1, 8, 9, 9, 10, 11, 12]  // replaced below if provided
var rowIds = [Int32](repeating: 0, count: batch * rows)
let env = ProcessInfo.processInfo.environment["ROW"] ?? ""
let provided = env.split(separator: ",").compactMap { Int32($0) }
let row = provided.isEmpty ? testRow : provided
for b in 0..<batch { for (t, v) in row.enumerated() where t < rows { rowIds[b * rows + t] = v } }
let input = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(bytes: &rowIds, count: rowIds.count * 4),
                               shape: [NSNumber(value: batch), NSNumber(value: rows)], dataType: .int32)
func run() -> [Float] {
    let results = executable.run(with: queue, inputs: [input], results: nil, executionDescriptor: nil)
    var values = [Float](repeating: 0, count: batch * (rows - 1))
    results[0].mpsndarray().readBytes(&values, strideBytes: nil)
    return values
}
_ = run()  // warm-up
let candidateStart = Int(ProcessInfo.processInfo.environment["START"] ?? "5") ?? 5
let candidateLength = Int(ProcessInfo.processInfo.environment["LENGTH"] ?? "2") ?? 2
let values = run()
var total: Float = 0
for p in candidateStart ..< candidateStart + candidateLength { total += values[p - 1] }  // logp[t-1] is for ids[t]
print(String(format: "sum logp of the candidate: %.4f", total))
let timing = Date()
let n = 30
for _ in 0..<n { _ = run() }
print(String(format: "%.2f ms per call", Date().timeIntervalSince(timing) * 1000 / Double(n)))
