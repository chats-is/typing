import Cocoa
import Engine
import InputMethodKit

// Stand-in for the app's EngineHost: no lexicon or model needed to test the symbol keys.
enum EngineHost {
    static let scorer: NeuralScorer? = nil
    static var sentenceScorer: SentenceScorer? = nil
    /// The real lexicon without the model (deterministic), so candidate paging can be tested.
    static let lexiconDecoder: LexiconDecoder? = {
        guard let lexicon = try? Lexicon(path: "Resources/lexicon.sqlite") else { return nil }
        return LexiconDecoder(lexicon: lexicon, scorer: nil, bigram: try? BigramModel(path: "Resources/bigram.bin"))
    }()
    static let decoder: CandidateDecoding? = lexiconDecoder.map(LaterLookDecoder.init)
}

/// The lexicon decoder, with the model's second look stood in for while `on` is set: it reverses
/// the candidates, so what the host does with the later update can be tested without a model.
final class LaterLookDecoder: CandidateDecoding {
    static var on = false
    let base: CandidateDecoding
    init(_ base: CandidateDecoding) { self.base = base }
    func decode(_ input: String, precedingText: String) -> DecodeResult {
        var result = base.decode(input, precedingText: precedingText)
        if Self.on, result.candidates.count > 1 {
            let quick = result
            result.refine = { _ in DecodeResult(candidates: quick.candidates.reversed(), sentences: quick.sentences, segmented: quick.segmented) }
        }
        return result
    }
}

/// A text field as the input method sees it: text, cursor, marked range.
final class FakeClient: NSObject, IMKTextInput {
    var text = NSMutableString()
    var cursor = 0
    var marked: NSRange?
    var log: [String] = []

    func insertText(_ string: Any!, replacementRange: NSRange) {
        let s = (string as? NSAttributedString)?.string ?? (string as? String) ?? ""
        var range: NSRange
        if replacementRange.location != NSNotFound { range = replacementRange }
        else if let marked { range = marked }
        else { range = NSRange(location: cursor, length: 0) }
        text.replaceCharacters(in: range, with: s)
        cursor = range.location + (s as NSString).length
        marked = nil
        log.append("insert \"\(s)\" at \(range)")
    }
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {
        let s = (string as? NSAttributedString)?.string ?? (string as? String) ?? ""
        let range = replacementRange.location != NSNotFound ? replacementRange : (marked ?? NSRange(location: cursor, length: 0))
        text.replaceCharacters(in: range, with: s)
        let length = (s as NSString).length
        marked = length > 0 ? NSRange(location: range.location, length: length) : nil
        cursor = range.location + selectionRange.location
        log.append("mark \"\(s)\"")
    }
    func selectedRange() -> NSRange { NSRange(location: cursor, length: 0) }
    func markedRange() -> NSRange { marked ?? NSRange(location: NSNotFound, length: 0) }
    func hasMarkedText() -> Bool { marked != nil }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! {
        guard range.location != NSNotFound, range.location + range.length <= text.length else { return nil }
        return NSAttributedString(string: text.substring(with: range))
    }
    func length() -> Int { text.length }
    func characterIndex(for point: NSPoint, tracking: IMKLocationToOffsetMappingMode, inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    /// What the application answers for the line rectangle (.zero: nothing).
    var lineAnswer = NSRect(x: 100, y: 100, width: 1, height: 16)
    func attributes(forCharacterIndex index: Int, lineHeightRectangle: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! { lineHeightRectangle?.pointee = lineAnswer; return [:] }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeName: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "test.fake" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "fake" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { text.substring(with: range) }
    /// What the application answers for the first rectangle of a range (.zero: nothing).
    var firstRectAnswer = NSRect.zero
    func firstRect(forCharacterRange range: NSRange, actualRange: NSRangePointer!) -> NSRect { firstRectAnswer }

    // What the application itself does with keys the input method leaves to it.
    func appHandles(_ event: NSEvent) {
        switch event.keyCode {
        case 123: cursor = max(0, cursor - 1); log.append("app: left")
        case 124: cursor = min(text.length, cursor + 1); log.append("app: right")
        case 51:
            if cursor > 0 { text.deleteCharacters(in: NSRange(location: cursor - 1, length: 1)); cursor -= 1 }
            log.append("app: delete")
        default:
            if let s = event.characters, !s.isEmpty, s.unicodeScalars.allSatisfy({ $0.value >= 32 }) {
                text.insert(s, at: cursor); cursor += (s as NSString).length; log.append("app: typed \(s)")
            }
        }
    }
}

_ = NSApplication.shared
UserDefaults.standard.removePersistentDomain(forName: "symtest")
ShortcutStore.shared.resetDefaults()
let client = FakeClient()
final class TestController: TypingInputController {
    static var fake: FakeClient?
    override func client() -> (any IMKTextInput & NSObjectProtocol)! { TestController.fake }
}
TestController.fake = client
let controller = TestController(server: nil, delegate: nil, client: nil)!

func key(_ chars: String, code: UInt16, shift: Bool = false) {
    let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: shift ? [.shift] : [], timestamp: 0, windowNumber: 0, context: nil,
                                 characters: chars, charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code)!
    let handled = controller.handle(event, client: client)
    if !handled { client.appHandles(event) }
}
let codes: [Character: UInt16] = [".": 47, ",": 43, "2": 19, "1": 18, "a": 0, "x": 7, "\"": 39, ":": 41, ";": 41, "b": 11, "3": 20]
func type(_ s: String) { for c in s { key(String(c), code: codes[c] ?? 0, shift: c == "\"" || c == ":") } }
func flags(_ code: UInt16, _ modifiers: NSEvent.ModifierFlags) {
    let event = NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0, context: nil,
                                 characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: code)!
    _ = controller.handle(event, client: client)
}
/// A letter typed with the lock on: the keyboard reports the capital, as macOS does.
func lockedKey(_ chars: String, code: UInt16, shift: Bool = false) {
    let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: shift ? [.capsLock, .shift] : [.capsLock], timestamp: 0, windowNumber: 0, context: nil,
                                 characters: chars, charactersIgnoringModifiers: chars.lowercased(), isARepeat: false, keyCode: code)!
    if !controller.handle(event, client: client) { client.appHandles(event) }
}
let LEFT: UInt16 = 123, RIGHT: UInt16 = 124, TAB: UInt16 = 48, DEL: UInt16 = 51, ESC: UInt16 = 53, SPACE: UInt16 = 49

var failures = 0
func check(_ name: String, text: String, cursor: Int) {
    let ok = client.text as String == text && client.cursor == cursor
    print((ok ? "ok   " : "FAIL ") + name + ": \"\(client.text)\" cursor \(client.cursor)" + (ok ? "" : "  expected \"\(text)\" cursor \(cursor)\n      " + client.log.joined(separator: "; ")))
    if !ok { failures += 1 }
    key("", code: ESC)  // leave any state behind
    // Leave the double quotes closed for the next test: type one and see which came.
    client.text = NSMutableString(); client.cursor = 0; client.marked = nil
    type("\""); key("", code: ESC)
    if client.text as String == "\u{201C}" { type("\""); key("", code: ESC) }
    client.text = NSMutableString(); client.cursor = 0; client.marked = nil; client.log = []
}

// In Chinese mode a punctuation key types its Chinese mark at once, and every key after it is
// typed as usual (no panel: the marks are as the system's Pinyin types them).
type(".2");                      check("。 then 2 → 。2", text: "。2", cursor: 2)
type("."); key("", code: LEFT);  check("。 then ← moves the cursor", text: "。", cursor: 0)
let firstX = EngineHost.decoder?.decode("x", precedingText: "").refined().candidates.first?.text ?? "x"
type("."); key("", code: LEFT); key("x", code: 7); key("", code: SPACE)
                                 check("。 ← then pinyin x + space lands before the mark (\(firstX)。)", text: firstX + "。", cursor: firstX.count)
type(":"); key("", code: DEL);   check("： then delete → empty", text: "", cursor: 0)
type("\"2\"");                   check("“ 2 ” around a digit", text: "“2”", cursor: 3)
type("\"\"");                    check("“ then ” pair", text: "“”", cursor: 2)
type("\""); key("", code: DEL); type("\"");  check("“ delete “ → “ again", text: "“", cursor: 1)
client.text = NSMutableString(string: "xy"); client.cursor = 1
type(".");                       check("mid-text x|y: 。 goes in at the cursor", text: "x。y", cursor: 2)
type("..");                      check("。。", text: "。。", cursor: 2)
type("."); key(" ", code: SPACE); check("。 then space → 。 and a space", text: "。 ", cursor: 2)
type("."); key("\r", code: 36);  check("。 then Return goes to the application", text: "。", cursor: 1)
// Caps Lock is the capitals lock, as in other third-party input methods: with it on, capitals go
// into the text as they are and nothing is composed; the language does not change.
flags(57, [.capsLock]); lockedKey("A", code: 0); lockedKey("A", code: 0, shift: true); lockedKey("B", code: 11)
do {
    let ok = client.marked == nil && client.text as String == "AAB"
    print((ok ? "ok   " : "FAIL ") + "lock on: a, Shift-a, b → AAB typed as they are: \"\(client.text)\""); if !ok { failures += 1 }
}
client.text = NSMutableString(); client.cursor = 0; client.marked = nil
flags(57, []); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "lock off: a composes pinyin (still Chinese)"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
// Shortcut settings: Shift alone toggles English by default; with its bindings removed it does not,
// and a bound key (Control-Space) does.
func keyMods(_ chars: String, code: UInt16, _ modifiers: NSEvent.ModifierFlags) {
    let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0, context: nil,
                                 characters: chars, charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code)!
    if !controller.handle(event, client: client) { client.appHandles(event) }
}
flags(56, [.shift]); flags(56, [])
let markEnglish = ModeIndicator.shared.isVisible && !ModeIndicator.shared.showsChinese
key("a", code: 0)
let markGoes = !ModeIndicator.shared.isVisible
print((markEnglish && markGoes ? "ok   " : "FAIL ") + "Shift alone → the mark shows English (hollow dots), and goes on the next key"); if !(markEnglish && markGoes) { failures += 1 }
print((client.marked == nil && client.text as String == "a" ? "ok   " : "FAIL ") + "Shift alone → English (default)"); if !(client.marked == nil && client.text as String == "a") { failures += 1 }
flags(56, [.shift]); flags(56, []); client.text = NSMutableString(); client.cursor = 0
let markChinese = ModeIndicator.shared.isVisible && ModeIndicator.shared.showsChinese
print((markChinese ? "ok   " : "FAIL ") + "Shift alone again → the mark shows Chinese (filled dots)"); if !markChinese { failures += 1 }
ModeIndicator.shared.hide()
// No cursor reported: no mark at all (anywhere else would be the wrong place).
do {
    ModeIndicator.shared.show(chinese: true, near: nil)
    let hidden = !ModeIndicator.shared.isVisible
    print((hidden ? "ok   " : "FAIL ") + "no cursor reported → no mark"); if !hidden { failures += 1 }
}
// Switched again just as the mark fades: the fade's end must not take the new mark away.
do {
    let caret = NSRect(x: 400, y: 400, width: 1, height: 18)
    ModeIndicator.shared.show(chinese: true, near: caret)
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.08))  // the fade has begun
    ModeIndicator.shared.show(chinese: false, near: caret)
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.35))  // past the old fade's end
    let stays = ModeIndicator.shared.isVisible && ModeIndicator.shared.alphaValue == 1 && !ModeIndicator.shared.showsChinese
    print((stays ? "ok   " : "FAIL ") + "switched again while the mark fades → the new mark stays (visible \(ModeIndicator.shared.isVisible), alpha \(ModeIndicator.shared.alphaValue))"); if !stays { failures += 1 }
    ModeIndicator.shared.hide()
}
for binding in ShortcutStore.shared.bindings[.toggleLanguage] ?? [] where binding.keyCode != 57 { ShortcutStore.shared.remove(binding, from: .toggleLanguage) }
flags(56, [.shift]); flags(56, []); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "Shift alone with its binding removed → still Chinese"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
ShortcutStore.shared.add(KeyBinding(keyCode: 49, modifiers: NSEvent.ModifierFlags.control.rawValue, alone: false, label: "⌃Space"), to: .toggleLanguage)
keyMods(" ", code: 49, [.control]); key("a", code: 0)
print((client.marked == nil && client.text as String == "a" ? "ok   " : "FAIL ") + "Control-Space bound → English"); if !(client.marked == nil && client.text as String == "a") { failures += 1 }
keyMods(" ", code: 49, [.control]); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
ShortcutStore.shared.resetDefaults()
// Settings saved when Caps Lock switched by default: once, Caps Lock is taken off the switch and,
// with nothing left on it, Shift comes back.
do {
    let saved = #"{"toggleLanguage":[{"label":"Caps Lock","modifiers":0,"alone":true,"keyCode":57}]}"#
    UserDefaults.standard.set(Data(saved.utf8), forKey: "ShortcutBindings")
    UserDefaults.standard.removeObject(forKey: "CapsLockIsCapitalsLock")
    let codes = (ShortcutStore().bindings[.toggleLanguage] ?? []).map(\.keyCode).sorted()
    let ok = codes == [56, 60]
    print((ok ? "ok   " : "FAIL ") + "saved Caps Lock-only switch becomes left and right Shift: \(codes)"); if !ok { failures += 1 }
    UserDefaults.standard.removeObject(forKey: "ShortcutBindings")
}
// Caps Lock bound to the switch in Settings: lock on = English (capitals as typed), off = Chinese.
ShortcutStore.shared.add(KeyBinding(keyCode: 57, modifiers: 0, alone: true, label: "Caps Lock"), to: .toggleLanguage)
flags(57, [.capsLock]); lockedKey("A", code: 0)
print((client.marked == nil && client.text as String == "A" ? "ok   " : "FAIL ") + "Caps Lock bound: lock on → English"); if !(client.marked == nil && client.text as String == "A") { failures += 1 }
client.text = NSMutableString(); client.cursor = 0; client.marked = nil
flags(57, []); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "Caps Lock bound: lock off → Chinese"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
ShortcutStore.shared.resetDefaults()
// Arrow keys cross page boundaries: → past the 9th candidate goes to the next page, ← back.
if let decoder = EngineHost.decoder {
    let candidates = decoder.decode("ni", precedingText: "").refined().candidates.map(\.text)
    if candidates.count > 10 {
        type("ni"); for _ in 0..<9 { key("", code: RIGHT) }; key(" ", code: SPACE)
        check("ni, → ×9, space → 10th candidate (\(candidates[9]))", text: candidates[9], cursor: candidates[9].count)
        type("ni"); for _ in 0..<9 { key("", code: RIGHT) }; key("", code: LEFT); key(" ", code: SPACE)
        check("ni, → ×9, ← back to the 9th (\(candidates[8]))", text: candidates[8], cursor: candidates[8].count)
        type("ni"); key("", code: LEFT); key(" ", code: SPACE)
        check("ni, ← at the first stays (\(candidates[0]))", text: candidates[0], cursor: candidates[0].count)
    } else { print("skip paging test: only \(candidates.count) candidates") }
} else { print("skip paging test: no lexicon") }
// Regressions found in review: a separator after turning the page must not overflow the page;
// a mouse event must not be asked for a key code; a digit typed over pinyin keeps "." English;
// Return gives the pinyin as typed; Shift held for a capital does not toggle English.
if EngineHost.decoder != nil {
    type("a"); key("", code: 125); type("'"); key(" ", code: SPACE)
    print((client.marked == nil && !(client.text as String).isEmpty ? "ok   " : "FAIL ") + "a, ↓, ' , space → committed without overflow (\(client.text))"); if !(client.marked == nil && !(client.text as String).isEmpty) { failures += 1 }
    client.text = NSMutableString(); client.cursor = 0; client.marked = nil
}
let click = NSEvent.mouseEvent(with: .leftMouseDown, location: NSPoint(x: 10, y: 10), modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1)!
type("ni")
let clickHandled = controller.handle(click, client: client)
print((!clickHandled && client.marked == nil ? "ok   " : "FAIL ") + "mouse click while composing → pinyin committed raw, event passed on"); if clickHandled || client.marked != nil { failures += 1 }
check("…text after the click", text: "ni", cursor: 2)
type("ni0."); check("ni, 0, . → candidate, 0 and an English .", text: (EngineHost.decoder?.decode("ni", precedingText: "").refined().candidates.first?.text ?? "ni") + "0.", cursor: 3)
type("nihao"); key("\r", code: 36); check("nihao, Return → pinyin as typed", text: "nihao", cursor: 5)
// The "'" typed between letters stays when they go out as typed; picking a candidate is unchanged.
func typeApostrophe(_ s: String) { for c in s { if c == "'" { key("'", code: 39) } else { type(String(c)) } } }
typeApostrophe("xi'an")
let shownWhileTyping = client.text as String
print((shownWhileTyping == "xi'an" ? "ok   " : "FAIL ") + "xi'an shows as xi'an while typing: \"\(shownWhileTyping)\""); if shownWhileTyping != "xi'an" { failures += 1 }
key("\r", code: 36); check("xi'an, Return → xi'an", text: "xi'an", cursor: 5)
typeApostrophe("don't"); key("\r", code: 36); check("don't, Return → don't", text: "don't", cursor: 5)
// A "'" right after letters typed straight into the text stays "'" (the system's Pinyin: XI'an); elsewhere it is a quote.
key("X", code: 7, shift: true); key("I", code: 34, shift: true); typeApostrophe("'an"); key("\r", code: 36)
check("XI'an, Return → XI'an", text: "XI'an", cursor: 5)
key("'", code: 39); check("' on its own → ‘", text: "\u{2018}", cursor: 1)
key("X", code: 7, shift: true); typeApostrophe("i'an"); key(" ", code: SPACE); check("Xi'an (Shift-X), space → Xi'an", text: "Xi'an", cursor: 5)
if let decoder = EngineHost.decoder {
    let candidates = decoder.decode("xi'an", precedingText: "").refined().candidates
    if let first = candidates.first?.text {
        typeApostrophe("xi'an"); key(" ", code: SPACE); check("xi'an, space → \(first)", text: first, cursor: first.count)
    }
    // A pick of the front, then Return: no "'" is sent in front of the letters left.
    if let index = candidates.prefix(9).firstIndex(where: { $0.consumed < 5 }) {
        let front = candidates[index].text
        let rest = String("xi'an".dropFirst(candidates[index].consumed).drop { $0 == "'" })
        typeApostrophe("xi'an"); key(String(index + 1), code: 18); key("\r", code: 36)
        check("xi'an, pick \(front) for the front, Return → \(front)\(rest)", text: front + rest, cursor: front.count + rest.count)
    } else { print("skip: no candidate for the front alone on the first page of xi'an") }
}
// The marks of GB/T 19246-2003 table 1 that were missing; Tab still gives the key's own character.
// The marks of GB/T 19246-2003 table 1 that were missing, as the system's Pinyin types them.
key("`", code: 50); check("` → ·", text: "·", cursor: 1)
key("{", code: 33, shift: true); key("}", code: 30, shift: true); check("{ } → 「」", text: "「」", cursor: 2)
key("|", code: 42, shift: true); check("| → ｜", text: "｜", cursor: 1)
key("`", code: 50); key("3", code: 20); check("` then 3 → ·3", text: "·3", cursor: 2)
// - and = turn the pages; on the first page - sends the letters as typed and itself (ai-).
// Marks typed by their names come right after the name, never first: yinhao → 引号 「」 『』 …
if let decoder = EngineHost.decoder {
    for context in ["", "请输入"] {
        let offered = decoder.decode("yinhao", precedingText: context).refined().candidates.map(\.text)
        let name = offered.firstIndex(of: "引号"), mark = offered.firstIndex(of: "「」")
        let ok = offered.first == "引号" && name != nil && mark == name! + 1
        print((ok ? "ok   " : "FAIL ") + "yinhao after \"\(context)\": 引号 first, 「」 right after it (\(offered.prefix(4).joined(separator: " ")))"); if !ok { failures += 1 }
    }
}
// Characters no font on this Mac draws are not offered (they would show as a box); the rare
// characters it draws still are, in full lists of a syllable.
if let decoder = EngineHost.lexiconDecoder, let lexicon = try? Lexicon(path: "Resources/lexicon.sqlite") {
    func beyondExtensionA(_ text: String) -> Bool { text.unicodeScalars.contains { $0.value >= 0x20000 } }
    var drawable: (String, String)?, undrawable: (String, String)?
    for syllable in lexicon.syllables.sorted() where drawable == nil || undrawable == nil {
        for entry in lexicon.characters(syllable) where beyondExtensionA(entry.word) {
            if CandidatePanel.canShow(entry.word) { drawable = drawable ?? (syllable, entry.word) } else { undrawable = undrawable ?? (syllable, entry.word) }
        }
    }
    decoder.canShow = CandidatePanel.canShow
    if let (syllable, word) = drawable {
        let offered = decoder.decode(syllable, precedingText: "").refined().candidates.map(\.text)
        let ok = offered.contains(word)
        print((ok ? "ok   " : "FAIL ") + "\(syllable): \(word), which this Mac draws, is offered"); if !ok { failures += 1 }
    }
    if let (syllable, word) = undrawable {
        let offered = decoder.decode(syllable, precedingText: "").refined().candidates.map(\.text)
        let boxes = offered.filter { !CandidatePanel.canShow($0) }
        let ok = !offered.contains(word) && boxes.isEmpty
        print((ok ? "ok   " : "FAIL ") + "\(syllable): nothing this Mac cannot draw is offered (\(word) left out)"); if !ok { failures += 1 }
    }
    decoder.canShow = nil
}
type("ai"); key("-", code: 27); check("ai, - on the first page → letters as typed and -", text: "ai-", cursor: 3)
if let candidates = EngineHost.decoder?.decode("ni", precedingText: "").refined().candidates.map(\.text), candidates.count > 10 {
    type("ni"); key("=", code: 24); key(" ", code: SPACE); check("ni, = → second page, space → \(candidates[9])", text: candidates[9], cursor: candidates[9].count)
    type("ni"); key("=", code: 24); key("-", code: 27); key(" ", code: SPACE); check("ni, =, - → back on the first page, space → \(candidates[0])", text: candidates[0], cursor: candidates[0].count)
}
flags(56, [.shift]); keyMods("A", code: 0, [.shift]); flags(56, []); key("", code: SPACE)
let shiftTyped = client.text as String
print((!shiftTyped.isEmpty && (client.text as String) != "" ? "ok   " : "FAIL ") + "Shift held for a capital → no toggle (text \(shiftTyped))"); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "…and the next letter still composes pinyin"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
// English mode: a word typed there is recorded for learning; addresses, code and single letters are not.
ShortcutStore.shared.resetDefaults()
flags(56, [.shift]); flags(56, []); key("q", code: 12)
let english = client.marked == nil
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
print((english ? "ok   " : "FAIL ") + "Shift alone → English for the word test"); if !english { failures += 1 }
for c in "Hello, worls" { key(String(c), code: 0) }
key("", code: DEL); key("d", code: 0)  // worls → worl → world
for token in [" foo.bar ", "x ", "don't ", "v2 ", "https://a.cn "] { for c in token { key(String(c), code: c == " " ? SPACE : 0) } }
flags(56, [.shift]); flags(56, [])  // back to Chinese: the token in progress ends
Thread.sleep(forTimeInterval: 3.2)  // past the undo window of the last word
Learning.store.flush()
let recorded = Set(Learning.store.entries().map(\.text))
for (word, kept) in [("Hello", true), ("world", true), ("don't", true), ("foo.bar", false), ("x", false), ("v2", false), ("https://a.cn", false), ("worls", false)] {
    let ok = recorded.contains(word) == kept
    print((ok ? "ok   " : "FAIL ") + "English mode: \(word) \(kept ? "recorded" : "not recorded")"); if !ok { failures += 1 }
}
// Chinese mode: letters sent with Return are recorded when they are a word.
client.text = NSMutableString(); client.cursor = 0; client.marked = nil
for c in "dialog" { key(String(c), code: 0) }
key("\r", code: 36)
let sent = client.text as String == "dialog"
print((sent ? "ok   " : "FAIL ") + "Chinese mode: dialog + Return sends the letters"); if !sent { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
Thread.sleep(forTimeInterval: 3.2)
Learning.store.flush()
let learnedDialog = Learning.store.typedEnglish("dialog") == ["dialog"]
print((learnedDialog ? "ok   " : "FAIL ") + "…and dialog is recorded, offered for its letters without switching to English"); if !learnedDialog { failures += 1 }
// 以词定字: [ and ] put the first or the last character of the highlighted word for all its letters;
// nothing is learned from it (bianji does not spell 编).
if let word = EngineHost.decoder?.decode("bianji", precedingText: "").refined().candidates.first?.text, word.count == 2 {
    for c in "bianji" { key(String(c), code: 0) }; key("[", code: 33)
    check("bianji, [ → \(word.first!) (first of \(word))", text: String(word.first!), cursor: 1)
    for c in "bianji" { key(String(c), code: 0) }; key("]", code: 30)
    check("bianji, ] → \(word.last!) (last of \(word))", text: String(word.last!), cursor: 1)
    Thread.sleep(forTimeInterval: 3.2)
    Learning.store.flush()
    let learnedPick = Learning.store.entries().contains { $0.keys == "bianji" }
    print((!learnedPick ? "ok   " : "FAIL ") + "a character picked out of a word is not recorded"); if learnedPick { failures += 1 }
}
if let first = EngineHost.decoder?.decode("ni", precedingText: "").refined().candidates.first?.text, first.count == 1 {
    type("ni"); key("[", code: 33); check("ni, [ on one character → \(first)【 as before", text: first + "【", cursor: 2)
}
// 模糊音 from Settings: with z = zh, zidao offers 知道; off again, it does not.
if let decoder = EngineHost.lexiconDecoder {
    FuzzySettings.set(.zZh, on: true)
    let with = decoder.decode("zidao", precedingText: "").refined().candidates.prefix(9).map(\.text)
    FuzzySettings.set(.zZh, on: false)
    let without = decoder.decode("zidao", precedingText: "").refined().candidates.prefix(9).map(\.text)
    let ok = with.contains("知道") && !without.contains("知道") && FuzzySettings.rules.isEmpty
    print((ok ? "ok   " : "FAIL ") + "模糊音 z = zh: zidao offers 知道 only while it is on"); if !ok { failures += 1 }
}
// lue and nue are lüe and nüe without any setting (zhanlue → 战略); u = ü lets nu stand for nv (女人).
if let decoder = EngineHost.lexiconDecoder {
    let lue = decoder.decode("zhanlue", precedingText: "").refined().candidates.first?.text == "战略"
        && decoder.decode("nue", precedingText: "").refined().candidates.first?.text == "虐"
    print((lue ? "ok   " : "FAIL ") + "zhanlue → 战略, nue → 虐 (lue and nue are lüe and nüe)"); if !lue { failures += 1 }
    FuzzySettings.set(.uV, on: true)
    let with = decoder.decode("nuren", precedingText: "").refined().candidates.prefix(9).map(\.text)
    FuzzySettings.set(.uV, on: false)
    let without = decoder.decode("nuren", precedingText: "").refined().candidates.prefix(9).map(\.text)
    let ok = with.contains("女人") && !without.contains("女人") && FuzzySettings.rules.isEmpty
    print((ok ? "ok   " : "FAIL ") + "模糊音 u = ü: nuren offers 女人 only while it is on"); if !ok { failures += 1 }
}
// Quick forms: v and a number, a sum or a code point; digits and signs go into the composition.
func typeQuick(_ s: String) { for c in s { key(String(c), code: c == "-" ? 27 : c == "=" ? 24 : 0, shift: "*+()%^".contains(c)) } }
typeQuick("v1234"); key(" ", code: SPACE); check("v1234, space → 一千二百三十四", text: "一千二百三十四", cursor: 7)
typeQuick("v1234"); key("", code: RIGHT); key(" ", code: SPACE); check("v1234, →, space → 壹仟贰佰叁拾肆", text: "壹仟贰佰叁拾肆", cursor: 7)
typeQuick("v12*3+5"); key(" ", code: SPACE); check("v12*3+5, space → 41", text: "41", cursor: 2)
typeQuick("v10-3"); key(" ", code: SPACE); check("v10-3 (- is a minus here, no page) → 7", text: "7", cursor: 1)
typeQuick("vu4e00"); key(" ", code: SPACE); check("vu4e00, space → 一", text: "一", cursor: 1)
for c in "video" { key(String(c), code: 0) }; key("\r", code: 36); check("video stays letters (Return → video)", text: "video", cursor: 5)
if let today = QuickInput.dynamic(forName: "日期").first, EngineHost.decoder?.decode("riqi", precedingText: "").refined().candidates.first?.text == "日期" {
    for c in "riqi" { key(String(c), code: 0) }; key("", code: RIGHT); key(" ", code: SPACE)
    check("riqi, →, space → today \(today)", text: today, cursor: today.count)
}
if let weekday = QuickInput.dynamic(forName: "星期").first, EngineHost.decoder?.decode("xingqi", precedingText: "").refined().candidates.first?.text == "星期" {
    for c in "xingqi" { key(String(c), code: 0) }; key("", code: RIGHT); key(" ", code: SPACE)
    check("xingqi, →, space → \(weekday)", text: weekday, cursor: weekday.count)
}
Thread.sleep(forTimeInterval: 3.2)
Learning.store.flush()
let learnedQuick = Learning.store.entries().contains { $0.keys.hasPrefix("v1234") || $0.text == "一千二百三十四" || $0.text.hasPrefix("星期") }
print((!learnedQuick ? "ok   " : "FAIL ") + "quick results and today's weekday are not recorded"); if learnedQuick { failures += 1 }
// Shift-Delete on a candidate takes back what was learned for it, as its tag in Settings does.
Learning.store.record(keys: "shijie", text: "视界", context: "", undoWindow: 0)
Learning.store.flush()
if let index = EngineHost.decoder?.decode("shijie", precedingText: "").refined().candidates.prefix(9).firstIndex(where: { $0.text == "视界" }) {
    for c in "shijie" { key(String(c), code: 0) }
    for _ in 0..<index { key("", code: RIGHT) }
    keyMods("", code: 51, [.shift])
    let stillComposing = client.marked != nil
    key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
    let forgotten = !Learning.store.entries().contains { $0.keys == "shijie" && $0.text == "视界" } && Learning.store.isHeld(keys: "shijie")
    let ok = forgotten && stillComposing
    print((ok ? "ok   " : "FAIL ") + "shijie, → to 视界, Shift-Delete → its record is gone and its keys held; still composing"); if !ok { failures += 1 }
} else { print("skip Shift-Delete test: 视界 not on the first page of shijie") }
// Shuangpin: two keys a syllable, decoded as the full pinyin they spell (Xiaohe nihc → ni'hao).
if let decoder = EngineHost.lexiconDecoder, let nihao = decoder.decode("ni'hao", precedingText: "").refined().candidates.first?.text {
    InputSession.shuangpin = decoder.shuangpin(.xiaohe)
    for c in "nihc" { key(String(c), code: 0) }
    let shown = client.text as String
    key(" ", code: SPACE)
    check("小鹤 nihc shows \"\(shown)\", space → \(nihao)", text: nihao, cursor: nihao.count)
    print((shown == "ni hao" ? "ok   " : "FAIL ") + "小鹤 nihc is shown as ni hao: \"\(shown)\""); if shown != "ni hao" { failures += 1 }
    Thread.sleep(forTimeInterval: 3.2)
    Learning.store.flush()
    let byPinyin = Learning.store.entries().contains { $0.keys == "nihao" && $0.text == nihao }
    print((byPinyin ? "ok   " : "FAIL ") + "小鹤 nihc is learned by its full pinyin nihao"); if !byPinyin { failures += 1 }
    typeQuick("v1234"); key(" ", code: SPACE); check("小鹤 v1234 is still a quick form → 一千二百三十四", text: "一千二百三十四", cursor: 7)
    // vu is zhu in Xiaohe: a digit after it picks a candidate (no code point), capitals stay as typed.
    let zhu = decoder.decode("zhu", precedingText: "").refined().candidates.map(\.text)
    if zhu.count > 1 {
        for c in "vu" { key(String(c), code: 0) }; key("2", code: 19)
        check("小鹤 vu, 2 → \(zhu[1]) (the second candidate of zhu)", text: zhu[1], cursor: zhu[1].count)
    }
    for c in "dv" { key(String(c), code: 0) }; key("P", code: 35, shift: true); key("C", code: 8, shift: true)
    let withCapitals = client.text as String
    key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
    print((withCapitals.hasPrefix("dui PC") ? "ok   " : "FAIL ") + "小鹤 dv then PC is read dui PC: \"\(withCapitals)\""); if !withCapitals.hasPrefix("dui PC") { failures += 1 }
    InputSession.shuangpin = decoder.shuangpin(.microsoft)
    if let xing = decoder.decode("xing", precedingText: "").refined().candidates.first?.text {
        key("x", code: 7); key(";", code: 41); key(" ", code: SPACE)
        check("微软 x; (ing on ;) → \(xing)", text: xing, cursor: xing.count)
        // After a whole syllable ; is the mark again: nihk; is 你好 and "；".
        for c in "nihk" { key(String(c), code: 0) }; key(";", code: 41)
        let afterSyllable = client.text as String
        key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
        print((afterSyllable.hasSuffix("；") ? "ok   " : "FAIL ") + "微软 nihk then ; → the mark after the syllables: \"\(afterSyllable)\""); if !afterSyllable.hasSuffix("；") { failures += 1 }
    }
    // Sogou (rime-ice): a and e finals also as that letter and the final's key (aj → an), ing on ;.
    InputSession.shuangpin = decoder.shuangpin(.sogou)
    if let an = decoder.decode("an", precedingText: "").refined().candidates.first?.text {
        for c in "aj" { key(String(c), code: 0) }; key(" ", code: SPACE)
        check("搜狗 aj → \(an) (an)", text: an, cursor: an.count)
    }
    InputSession.shuangpin = nil
}
// Learning records keep nothing sensitive: long numbers and addresses are taken out, and what follows
// a prompt for a password or a code is not recorded.
Learning.store.record(keys: "haode", text: "好的", context: "电话13812345678，邮箱a@b.cn，", undoWindow: 0)
Learning.store.record(keys: "yiqi", text: "一起", context: "请输入验证码", undoWindow: 0)
Learning.store.record(keys: "zaijian", text: "再见", context: "", undoWindow: 0)
Learning.store.flush()
let redactedContext = Learning.store.entries().first { $0.text == "好的" }?.context
print((redactedContext == "电话＊，邮箱＊，" ? "ok   " : "FAIL ") + "learning record: number and e-mail taken out of the context (\(redactedContext ?? "-"))"); if redactedContext != "电话＊，邮箱＊，" { failures += 1 }
// The same rules as the training text (Training/test_cleaning.py has the full list).
for (text, expected) in [("网址http://a.cn/x好", "网址＊好"), ("回复 @联合光子a 你说得对", "回复 @＊ 你说得对"), ("微信号：caffe007", "微信号：＊"),
                         ("南门11楼1211 提前电联", "南门＊ 提前电联"), ("灰色豫AG0T02宝马", "灰色＊宝马"), ("姓名学没什么道理", "姓名学没什么道理"),
                         ("我们2016-07-20 11:30见", "我们2016-07-20 11:30见")] {
    let got = Redaction.redact(text)
    print((got == expected ? "ok   " : "FAIL ") + "redaction: \(text) → \(got)"); if got != expected { failures += 1 }
}
// The store redacts its records anew at every start: a second run must change nothing.
for text in ["群里@所有人今天下午三点开会请大家准时到场不要迟到谢谢配合", "联系人：张三 电话13812345678 微信号：abc123 邮箱a@b.com 网址http://a.cn/x好"] {
    let once = Redaction.redact(text), twice = Redaction.redact(once)
    print((once == twice ? "ok   " : "FAIL ") + "redaction twice = once: \(once)"); if once != twice { failures += 1 }
}
let afterPrompt = Learning.store.entries().contains { $0.text == "一起" }
print((!afterPrompt ? "ok   " : "FAIL ") + "learning record: nothing kept after a prompt for a code"); if afterPrompt { failures += 1 }
let offered = Learning.store.typedEnglish("dont") == ["don't"]
print((offered ? "ok   " : "FAIL ") + "a recorded English word is offered for its letters (dont → don't)"); if !offered { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil

// Where the candidates go: the line rectangle the application reports (100, 100) when it lies in one
// of the application's windows; otherwise its first rectangle of the marked range, if that one does.
func panelPlace(windows: [NSRect], line: NSRect = NSRect(x: 100, y: 100, width: 1, height: 16), firstRect answer: NSRect) -> NSPoint {
    let windowFrames = TypingInputController.windowFrames
    TypingInputController.windowFrames = { _ in windows }
    client.lineAnswer = line
    client.firstRectAnswer = answer
    type("ni")
    let place = TypingInputController.panel.anchor.origin
    key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
    client.firstRectAnswer = .zero
    client.lineAnswer = NSRect(x: 100, y: 100, width: 1, height: 16)
    TypingInputController.windowFrames = windowFrames
    return place
}
let around = NSRect(x: 0, y: 0, width: 800, height: 600)        // holds the line rectangle
let elsewhere = NSRect(x: 300, y: 600, width: 600, height: 60)  // does not
let inside = NSRect(x: 320, y: 620, width: 1, height: 16)
let garbage = NSRect(x: 1.6e-314, y: 8272.9, width: 1.6e-314, height: -1)
for (name, windows, answer, expected) in [
    ("line rectangle in the window → it, whatever the first rectangle", [around], inside, NSPoint(x: 100, y: 100)),
    ("windows unknown → the line rectangle", [], inside, NSPoint(x: 100, y: 100)),
    ("line rectangle outside, first rectangle garbage → the line rectangle", [elsewhere], garbage, NSPoint(x: 100, y: 100)),
    ("line rectangle outside, first rectangle outside too → the line rectangle", [elsewhere], NSRect(x: 100, y: 117, width: 7, height: 17), NSPoint(x: 100, y: 100)),
    ("line rectangle outside, first rectangle in the window → the first rectangle", [elsewhere], inside, inside.origin),
] {
    let place = panelPlace(windows: windows, firstRect: answer)
    let ok = place == expected
    print((ok ? "ok   " : "FAIL ") + "candidates placed: \(name) (\(place))"); if !ok { failures += 1 }
}
// No line rectangle at all (Spotlight): the first rectangle where it lies in a window, or, the windows
// unknown, on a screen; otherwise the candidates keep their last place.
let onScreen = NSScreen.screens.contains { $0.frame.contains(NSPoint(x: inside.minX, y: inside.midY)) }
for (name, windows, answer, expected) in [
    ("no line rectangle, first rectangle in the window → the first rectangle", [elsewhere], inside, inside.origin),
    ("no line rectangle, windows unknown, first rectangle on a screen → the first rectangle", [], inside, inside.origin),
    ("no line rectangle, first rectangle outside the window → the last place", [around.offsetBy(dx: 900, dy: 0)], NSRect(x: 100, y: 117, width: 7, height: 17), inside.origin),
    ("no line rectangle, first rectangle garbage → the last place", [], garbage, inside.origin),
] where onScreen || !windows.isEmpty {
    let place = panelPlace(windows: windows, line: .zero, firstRect: answer)
    let ok = place == expected
    print((ok ? "ok   " : "FAIL ") + "candidates placed: \(name) (\(place))"); if !ok { failures += 1 }
}

// The model's later update of the same keystroke keeps the candidates where the first showing put
// them when the application reports no cursor (it used to send them to the screen's corner).
LaterLookDecoder.on = true
let reordered = EngineHost.decoder?.decode("nihao", precedingText: "").refine?({ false }) != nil ? "nihao" : nil
if let word = reordered {
    let windowFrames = TypingInputController.windowFrames
    TypingInputController.windowFrames = { _ in [] }
    type(String(word.prefix(1)))
    let before = TypingInputController.panel.anchor.origin
    client.lineAnswer = .zero
    type(String(word.dropFirst()))
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.5))  // the model's look has landed
    let after = TypingInputController.panel.anchor.origin
    let ok = after == before && after != .zero
    print((ok ? "ok   " : "FAIL ") + "candidates stay put through the model's update when no cursor is reported (\(word): \(before) → \(after))"); if !ok { failures += 1 }
    key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
    client.lineAnswer = NSRect(x: 100, y: 100, width: 1, height: 16)
    TypingInputController.windowFrames = windowFrames
} else {
    print("FAIL no second look to test the later update with"); failures += 1
}
LaterLookDecoder.on = false

// Webpage <input> that updates URL parameters and rewrites its value on `input`: the browser
// clears its marked range and calls commitComposition / deactivateServer (+ activateServer on a
// re-mounted <input>), yet the candidate panel stays open and the composition continues cleanly.
if let expected = EngineHost.decoder?.decode("ni", precedingText: "").refined().candidates.first?.text {
    type("n")
    client.marked = nil  // browser discards mark when JS rewrites input.value
    controller.commitComposition(client)
    controller.deactivateServer(client)
    let remounted = TestController(server: nil, delegate: nil, client: nil)!
    remounted.activateServer(client)
    let panelStaysOpen = TypingInputController.panel.isVisible
    print((panelStaysOpen ? "ok   " : "FAIL ") + "URL-switching <input>: panel stays open across commitComposition/deactivateServer"); if !panelStaysOpen { failures += 1 }
    let iEvent = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                                  characters: "i", charactersIgnoringModifiers: "i", isARepeat: false, keyCode: 34)!
    _ = remounted.handle(iEvent, client: client)
    client.marked = nil
    remounted.commitComposition(client)
    let spaceEvent = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                                      characters: " ", charactersIgnoringModifiers: " ", isARepeat: false, keyCode: SPACE)!
    _ = remounted.handle(spaceEvent, client: client)
    check("URL-switching <input>: n, i, space → \(expected) without duplicate letters", text: expected, cursor: expected.count)
    type("n")
    client.marked = nil
    controller.commitComposition(client)
    key("", code: DEL)
    check("URL-switching <input>: n then Delete clears the unmarked letter", text: "", cursor: 0)
    // After the reaction window expires (user pauses and then leaves the field), commitComposition
    // commits raw letters and closes the candidate panel as usual.
    type("ni")
    Thread.sleep(forTimeInterval: 0.3)
    controller.commitComposition(client)
    let closedAfterPause = !TypingInputController.panel.isVisible && client.marked == nil
    print((closedAfterPause ? "ok   " : "FAIL ") + "commitComposition after pause closes the panel and commits raw"); if !closedAfterPause { failures += 1 }
    check("…text after delayed commitComposition", text: "ni", cursor: 2)
    // If deactivateServer is not followed by activateServer (e.g. page navigated away with no input),
    // the panel closes once the reaction window elapses.
    type("ni")
    controller.deactivateServer(client)
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.3))
    let closedWhenUnfocused = !TypingInputController.panel.isVisible
    print((closedWhenUnfocused ? "ok   " : "FAIL ") + "deactivateServer without re-activation closes the panel"); if !closedWhenUnfocused { failures += 1 }
    key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil; client.log = []
}

print(failures == 0 ? "ALL OK" : "\(failures) FAILURES")
exit(failures == 0 ? 0 : 1)
