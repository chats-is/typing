import Foundation

/// Memory-mapped word bigram probabilities produced by Tools/build_bigram.py.
/// "BGM2": uint32 rows, uint32 count, rows + 1 uint32 row starts, count uint32 second words
/// (sorted within a row), count float16 log2 P(w2 | w1). The older "BGM1" (sorted uint64 keys
/// w1 << 32 | w2, float32 probabilities) is still read.
public final class BigramModel {
    /// Word id standing for the start of a sentence.
    public static let sentenceStart: UInt32 = 0

    public enum BigramError: Error {
        case cannotOpen(String)
        case badFormat
    }

    private let base: UnsafeMutableRawPointer
    private let length: Int
    private let count: Int
    private enum Layout {
        case pairs(keys: UnsafePointer<UInt64>, values: UnsafeRawPointer)                       // BGM1
        case rows(starts: UnsafePointer<UInt32>, rows: Int, seconds: UnsafePointer<UInt32>, values: UnsafeRawPointer)  // BGM2
    }
    private let layout: Layout

    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw BigramError.cannotOpen(String(cString: strerror(errno))) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size >= 12 else { throw BigramError.badFormat }
        length = Int(info.st_size)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), mapped != MAP_FAILED else {
            throw BigramError.cannotOpen(String(cString: strerror(errno)))
        }
        base = mapped
        let magic = [UInt8](UnsafeRawBufferPointer(start: mapped, count: 4))
        let raw = UnsafeRawPointer(mapped)
        if magic == Array("BGM2".utf8) {
            let rows = Int(mapped.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
            count = Int(mapped.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let startsOffset = 12, secondsOffset = startsOffset + (rows + 1) * 4, valuesOffset = secondsOffset + count * 4
            guard length == valuesOffset + count * 2 else { munmap(mapped, length); throw BigramError.badFormat }
            layout = .rows(starts: raw.advanced(by: startsOffset).assumingMemoryBound(to: UInt32.self), rows: rows,
                           seconds: raw.advanced(by: secondsOffset).assumingMemoryBound(to: UInt32.self), values: raw.advanced(by: valuesOffset))
        } else if magic == Array("BGM1".utf8) {
            count = Int(mapped.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
            guard length == 8 + count * 12 else { munmap(mapped, length); throw BigramError.badFormat }
            layout = .pairs(keys: raw.advanced(by: 8).assumingMemoryBound(to: UInt64.self), values: raw.advanced(by: 8 + count * 8))
        } else {
            munmap(mapped, length)
            throw BigramError.badFormat
        }
    }

    deinit {
        munmap(base, length)
    }


    /// Pairs looked up so far and the key range of every first word met: the search repeats the
    /// same pairs across the strict and tolerant runs and from one keystroke to the next.
    private var pairCache: [UInt64: Double] = [:]
    private var rangeCache: [UInt32: Range<Int>] = [:]
    private static let cacheLimit = 200_000

    private func lowerBound(_ key: UInt64, in range: Range<Int>, keys: UnsafePointer<UInt64>) -> Int {
        var low = range.lowerBound
        var high = range.upperBound
        while low < high {
            let middle = (low + high) / 2
            if keys[middle] < key { low = middle + 1 } else { high = middle }
        }
        return low
    }

    private func range(of first: UInt32) -> Range<Int> {
        if let cached = rangeCache[first] { return cached }
        let found: Range<Int>
        switch layout {
        case .rows(let starts, let rows, _, _):
            found = Int(first) + 1 < rows + 1 ? Int(starts[Int(first)]) ..< Int(starts[Int(first) + 1]) : 0 ..< 0
        case .pairs(let keys, _):
            let start = lowerBound(UInt64(first) << 32, in: 0..<count, keys: keys)
            found = start ..< lowerBound((UInt64(first) + 1) << 32, in: start..<count, keys: keys)
        }
        if rangeCache.count >= Self.cacheLimit { rangeCache.removeAll(keepingCapacity: true) }
        rangeCache[first] = found
        return found
    }

    public func probability(_ first: UInt32, _ second: UInt32) -> Double {
        let key = UInt64(first) << 32 | UInt64(second)
        if let cached = pairCache[key] { return cached }
        let within = range(of: first)
        let probability: Double
        switch layout {
        case .rows(_, _, let seconds, let values):
            var low = within.lowerBound, high = within.upperBound
            while low < high {
                let middle = (low + high) / 2
                if seconds[middle] < second { low = middle + 1 } else { high = middle }
            }
            probability = low < within.upperBound && seconds[low] == second
                ? exp2(Double(values.loadUnaligned(fromByteOffset: low * 2, as: Float16.self))) : 0
        case .pairs(let keys, let values):
            let index = lowerBound(key, in: within, keys: keys)
            probability = index < within.upperBound && keys[index] == key ? Double(values.loadUnaligned(fromByteOffset: index * 4, as: Float32.self)) : 0
        }
        if pairCache.count >= Self.cacheLimit { pairCache.removeAll(keepingCapacity: true) }
        pairCache[key] = probability
        return probability
    }
}
