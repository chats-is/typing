import Foundation

/// Writes what the top blocks learned into a copy of the shipped Core ML model, so the tuned
/// model runs where the shipped one runs (the Neural Engine) and the GPU is left to learning.
///
/// A compiled model keeps its constants in weights/weight.bin; model.mil names each constant
/// with the offset of its entry (a 64-byte header: sentinel, dtype, byte count, data offset).
/// The tuned tensors carry the shipped names, shapes and layout, which is checked before any
/// byte is written. The copy is built beside `destination` and moved into place when complete.
public enum TunedModel {
    public struct Failure: Error, CustomStringConvertible { public let description: String }

    public static func build(shipped: URL, tuned: GraphScorer.Weights, destination: URL) throws {
        let manager = FileManager.default
        let staging = destination.deletingLastPathComponent().appendingPathComponent(destination.lastPathComponent + ".building")
        try? manager.removeItem(at: staging)
        try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try manager.copyItem(at: shipped, to: staging)
        do {
            let entries = try constants(in: staging)
            let handle = try FileHandle(forUpdating: staging.appendingPathComponent("weights/weight.bin"))
            do {
                for (name, tensor) in tuned.tensors {
                    let milName = "model_" + name.replacingOccurrences(of: ".", with: "_") + "_to_fp16"
                    guard let entry = entries[milName] else { throw Failure(description: "\(name) is not a constant of the model") }
                    guard entry.shape == tensor.shape else { throw Failure(description: "\(name): shape \(entry.shape) in the model, \(tensor.shape) tuned") }
                    try handle.seek(toOffset: UInt64(entry.offset))
                    guard let header = try handle.read(upToCount: 24), header.count == 24 else { throw Failure(description: "\(name): short entry") }
                    let sentinel = header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
                    let dtype = header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }
                    let size = header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self) }
                    let dataOffset = header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 16, as: UInt64.self) }
                    guard sentinel == 0xDEAD_BEEF, dtype == 1, Int(size) == tensor.half.count else {
                        throw Failure(description: "\(name): unexpected storage (sentinel \(String(sentinel, radix: 16)), dtype \(dtype), \(size) bytes for \(tensor.half.count))")
                    }
                    try handle.seek(toOffset: dataOffset)
                    try handle.write(contentsOf: tensor.half)
                }
                try handle.close()
            } catch {
                try? handle.close()
                throw error
            }
            try? manager.removeItem(at: destination)
            try manager.moveItem(at: staging, to: destination)
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    /// The fp16 constants a compiled model stores in its weight file: name, shape, entry offset.
    static func constants(in model: URL) throws -> [String: (shape: [Int], offset: Int)] {
        let mil = try String(contentsOf: model.appendingPathComponent("model.mil"), encoding: .utf8)
        let pattern = #"const\(\)\[name = string\("([A-Za-z0-9_]+)"\), val = tensor<fp16, \[([0-9, ]*)\]>\(BLOBFILE\(path = string\("@model_path/weights/weight.bin"\), offset = uint64\(([0-9]+)\)\)\)\]"#
        let regex = try NSRegularExpression(pattern: pattern)
        var entries: [String: (shape: [Int], offset: Int)] = [:]
        for match in regex.matches(in: mil, range: NSRange(mil.startIndex..., in: mil)) {
            guard let nameRange = Range(match.range(at: 1), in: mil), let shapeRange = Range(match.range(at: 2), in: mil),
                  let offsetRange = Range(match.range(at: 3), in: mil), let offset = Int(mil[offsetRange]) else { continue }
            let shape = mil[shapeRange].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            entries[String(mil[nameRange])] = (shape, offset)
        }
        return entries
    }

    /// What a built copy was made from: the tuned file and the shipped model, by size and date.
    /// Stored beside the copy; when it no longer matches, the copy is rebuilt.
    public static func signature(tunedBase: String, shipped: URL) -> String {
        func stamp(_ path: String) -> String {
            let attributes = (try? FileManager.default.attributesOfItem(atPath: path)) ?? [:]
            let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            let date = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            return "\(size):\(date)"
        }
        return stamp(tunedBase + ".bin") + "|" + stamp(shipped.appendingPathComponent("weights/weight.bin").path)
    }
}
