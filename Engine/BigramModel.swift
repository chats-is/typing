import Foundation

/// Memory-mapped word bigram probabilities produced by Tools/build_bigram.py.
/// Layout: "BGM1", uint32 count, sorted uint64 keys (w1 << 32 | w2), float32 P(w2 | w1).
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
    private let keys: UnsafePointer<UInt64>
    private let values: UnsafeRawPointer

    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw BigramError.cannotOpen(String(cString: strerror(errno))) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size >= 8 else { throw BigramError.badFormat }
        length = Int(info.st_size)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), mapped != MAP_FAILED else {
            throw BigramError.cannotOpen(String(cString: strerror(errno)))
        }
        base = mapped
        let magic = [UInt8](UnsafeRawBufferPointer(start: mapped, count: 4))
        count = Int(mapped.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        guard magic == Array("BGM1".utf8), length == 8 + count * 12 else {
            munmap(mapped, length)
            throw BigramError.badFormat
        }
        keys = UnsafeRawPointer(mapped).advanced(by: 8).assumingMemoryBound(to: UInt64.self)
        values = UnsafeRawPointer(mapped).advanced(by: 8 + count * 8)
    }

    deinit {
        munmap(base, length)
    }


    /// Pairs looked up so far and the key range of every first word met: the search repeats the
    /// same pairs across the strict and tolerant runs and from one keystroke to the next.
    private var pairCache: [UInt64: Double] = [:]
    private var rangeCache: [UInt32: Range<Int>] = [:]
    private static let cacheLimit = 200_000

    private func lowerBound(_ key: UInt64, in range: Range<Int>) -> Int {
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
        let start = lowerBound(UInt64(first) << 32, in: 0..<count)
        let end = lowerBound((UInt64(first) + 1) << 32, in: start..<count)
        if rangeCache.count >= Self.cacheLimit { rangeCache.removeAll(keepingCapacity: true) }
        rangeCache[first] = start..<end
        return start..<end
    }

    public func probability(_ first: UInt32, _ second: UInt32) -> Double {
        let key = UInt64(first) << 32 | UInt64(second)
        if let cached = pairCache[key] { return cached }
        let within = range(of: first)
        let index = lowerBound(key, in: within)
        let probability = index < within.upperBound && keys[index] == key ? Double(values.loadUnaligned(fromByteOffset: index * 4, as: Float32.self)) : 0
        if pairCache.count >= Self.cacheLimit { pairCache.removeAll(keepingCapacity: true) }
        pairCache[key] = probability
        return probability
    }
}
