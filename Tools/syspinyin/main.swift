// Measures the system Pinyin input method on an eval file, the same way the engine is measured:
//
//     syspinyin --eval Data/eval_dialogue.tsv [--limit 300] [--no-context] [--out results.tsv] [--key-delay 0.03]
//
// For every "pin yin<TAB>sentence[<TAB>context]" line the context is placed in a text view, the pinyin letters
// are typed into the system Pinyin input method one by one, Space accepts the first candidate until nothing is
// left in composition, and the committed text is compared with the sentence. Prints the exact first-choice rate.
// Needs the system Pinyin input source (com.apple.inputmethod.SCIM.ITABC) enabled; the previous input source is
// restored at the end. The window takes keyboard focus while it runs.
import AppKit
import Carbon

var pinyinSourceID = "com.apple.inputmethod.SCIM.ITABC"
var measureLatency = false

var evalPath: String?
var outPath: String?
var limit = 300
var useContext = true
var keyDelay = 0.03
var useCGEvents = false
var arguments = CommandLine.arguments.dropFirst()[...]
while let argument = arguments.popFirst() {
    switch argument {
    case "--eval": evalPath = arguments.popFirst()
    case "--out": outPath = arguments.popFirst()
    case "--limit": limit = Int(arguments.popFirst() ?? "300") ?? 300
    case "--no-context": useContext = false
    case "--cgevent": useCGEvents = true
    case "--source": pinyinSourceID = arguments.popFirst() ?? pinyinSourceID
    case "--latency": measureLatency = true
    case "--log":
        if let path = arguments.popFirst() { freopen(path, "a", stdout); freopen(path, "a", stderr) }
    case "--key-delay": keyDelay = Double(arguments.popFirst() ?? "0.03") ?? 0.03
    default:
        FileHandle.standardError.write("unknown argument \(argument)\n".data(using: .utf8)!)
        exit(2)
    }
}
if evalPath == nil {
    FileHandle.standardError.write("usage: syspinyin --eval file.tsv [--limit N] [--no-context] [--out results.tsv]\n".data(using: .utf8)!)
    exit(2)
}
let evalFile = evalPath!

func inputSource(withID id: String) -> TISInputSource? {
    let list = TISCreateInputSourceList(nil, true).takeRetainedValue() as! [TISInputSource]
    return list.first { source in
        guard let pointer = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else { return false }
        return Unmanaged<CFString>.fromOpaque(pointer).takeUnretainedValue() as String == id
    }
}

func sourceID(_ source: TISInputSource) -> String {
    guard let pointer = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else { return "?" }
    return Unmanaged<CFString>.fromOpaque(pointer).takeUnretainedValue() as String
}

func spin(_ seconds: Double) {
    let until = Date(timeIntervalSinceNow: seconds)
    while Date() < until {
        if let event = NSApp.nextEvent(matching: .any, until: Date(timeIntervalSinceNow: 0.005), inMode: .default, dequeue: true) {
            NSApp.sendEvent(event)
        }
        RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.005))
    }
}

// Key codes for the letters and Space/Escape on the ABC layout.
let keyCodes: [Character: UInt16] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
    "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "'": 39, " ": 49, "\u{1b}": 53,
]



final class Runner: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var textView: NSTextView!
    var unhandledKeys = 0
    var keyLatencies: [Double] = []

    func currentMarkedText() -> String {
        guard textView.hasMarkedText() else { return "" }
        let range = textView.markedRange()
        return (textView.string as NSString).substring(with: NSRange(location: min(range.location, (textView.string as NSString).length), length: min(range.length, max(0, (textView.string as NSString).length - range.location))))
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 600, height: 300), styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.title = "系统拼音测试"
        let scroll = NSScrollView(frame: window.contentView!.bounds)
        scroll.autoresizingMask = [.width, .height]
        textView = NSTextView(frame: scroll.bounds)
        textView.autoresizingMask = [.width, .height]
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isAutomaticTextCompletionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.font = NSFont.systemFont(ofSize: 18)
        scroll.documentView = textView
        window.contentView?.addSubview(scroll)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window.makeFirstResponder(textView)
        DispatchQueue.main.async { self.run() }
    }

    func key(_ character: Character) {
        guard let code = keyCodes[character] else { return }
        if useCGEvents {
            // Real keyboard events through the window server, exactly as a keyboard would send them.
            for down in [true, false] {
                if let event = CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(code), keyDown: down) { event.post(tap: .cghidEventTap) }
                spin(keyDelay / 2)
            }
            return
        }
        let characters = character == "\u{1b}" ? "\u{1b}" : String(character)
        let location = window.mouseLocationOutsideOfEventStream
        for type in [NSEvent.EventType.keyDown, .keyUp] {
            if let event = NSEvent.keyEvent(with: type, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                            windowNumber: window.windowNumber, context: nil, characters: characters, charactersIgnoringModifiers: characters,
                                            isARepeat: false, keyCode: code) {
                if type == .keyDown {
                    if !(textView.inputContext?.handleEvent(event) ?? false) { unhandledKeys += 1 }
                } else {
                    _ = textView.inputContext?.handleEvent(event)
                }
            }
        }
        spin(keyDelay)
    }

    func typeSentence(_ letters: String, context: String) -> String {
        textView.string = context
        textView.setSelectedRange(NSRange(location: (context as NSString).length, length: 0))
        textView.inputContext?.discardMarkedText()
        spin(0.02)
        for letter in letters {
            if measureLatency {
                let before = currentMarkedText()
                let started = DispatchTime.now().uptimeNanoseconds
                key(letter)
                var waited = 0.0
                while currentMarkedText() == before && waited < 0.4 {
                    spin(0.001)
                    waited = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
                }
                keyLatencies.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6 - keyDelay * 1000)
            } else {
                key(letter)
            }
        }
        spin(0.08)
        var presses = 0
        while textView.hasMarkedText(), presses < 12 {
            key(" ")
            spin(0.08)
            presses += 1
        }
        if textView.hasMarkedText() {
            textView.inputContext?.discardMarkedText()
            key("\u{1b}")
        }
        spin(0.05)
        let text = textView.string
        guard text.hasPrefix(context) else { return text }
        return String(text.dropFirst(context.count))
    }

    func run() {
        let original = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
        guard let pinyin = inputSource(withID: pinyinSourceID) else {
            print("system Pinyin (\(pinyinSourceID)) is not available")
            NSApp.terminate(nil)
            return
        }
        if useCGEvents {
            let trusted = AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary)
            print("accessibility trusted: \(trusted)")
            if !trusted {
                for _ in 0..<600 where !AXIsProcessTrusted() { spin(0.5) }
                print("accessibility trusted now: \(AXIsProcessTrusted())")
            }
        }
        TISEnableInputSource(pinyin)
        TISSelectInputSource(pinyin)
        spin(0.5)
        for _ in 0..<40 where !(NSApp.isActive && window.isKeyWindow) {
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(textView)
            spin(0.1)
        }
        textView.inputContext?.activate()
        // Give the input method's server process time to start and connect.
        let serverBundle = pinyinSourceID.hasPrefix("com.apple.inputmethod.SCIM") ? "com.apple.inputmethod.SCIM" : "is.chats.zhiyin"
        for _ in 0..<100 where NSRunningApplication.runningApplications(withBundleIdentifier: serverBundle).isEmpty { spin(0.1) }
        spin(1.5)
        print("input method server running: \(!NSRunningApplication.runningApplications(withBundleIdentifier: serverBundle).isEmpty)")
        print("input source: \(sourceID(TISCopyCurrentKeyboardInputSource().takeRetainedValue()))  app active: \(NSApp.isActive)  key window: \(window.isKeyWindow)  context current: \(NSTextInputContext.current === textView.inputContext)")
        fflush(stdout)
        var lines: [String] = []
        var total = 0, exact = 0, characters = 0, correctCharacters = 0
        let content = (try? String(contentsOfFile: evalFile, encoding: .utf8)) ?? ""
        for line in content.split(separator: "\n") {
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard fields.count >= 2, total < limit else { continue }
            let syllables = fields[0].split(separator: " ")
            let expected = String(fields[1])
            guard syllables.count == expected.count else { continue }
            let context = fields.count > 2 && useContext ? fields[2].replacingOccurrences(of: "\\n", with: "\n") : ""
            let letters = syllables.joined()
            let actual = typeSentence(letters, context: context).trimmingCharacters(in: .whitespacesAndNewlines)
            total += 1
            if actual == expected { exact += 1 }
            characters += expected.count
            correctCharacters += zip(actual, expected).filter { $0 == $1 }.count
            lines.append("\(letters)\t\(expected)\t\(actual)\t\(actual == expected ? 1 : 0)")
            if let outPath { try? lines.joined(separator: "\n").appending("\n").write(toFile: outPath, atomically: true, encoding: .utf8) }
            if total % 50 == 0 {
                print("\(total) sentences, exact \(String(format: "%.1f", 100 * Double(exact) / Double(total)))%")
                fflush(stdout)
            }
        }
        if unhandledKeys > 0 { print("WARNING: \(unhandledKeys) keys were not taken by the input method") }
        if !keyLatencies.isEmpty {
            let sorted = keyLatencies.sorted()
            let mean = sorted.reduce(0, +) / Double(sorted.count)
            let p50 = sorted[sorted.count / 2], p95 = sorted[Int(Double(sorted.count - 1) * 0.95)], p99 = sorted[Int(Double(sorted.count - 1) * 0.99)]
            let timeouts = sorted.filter { $0 >= 400 - keyDelay * 1000 }.count
            print(String(format: "key latency ms (key sent → composition changed): mean %.1f  median %.1f  p95 %.1f  p99 %.1f  max %.1f  n=%d  timeouts=%d", mean, p50, p95, p99, sorted.last!, sorted.count, timeouts))
        }
        print("sentences: \(total)  exact: \(String(format: "%.2f", 100 * Double(exact) / Double(max(1, total))))%  characters: \(String(format: "%.2f", 100 * Double(correctCharacters) / Double(max(1, characters))))%")
        if let outPath { try? lines.joined(separator: "\n").appending("\n").write(toFile: outPath, atomically: true, encoding: .utf8) }
        TISSelectInputSource(original)
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let runner = Runner()
app.delegate = runner
app.run()
