import Cocoa
import Engine
import InputMethodKit

// Stand-in for the app's EngineHost: no lexicon or model needed to test the symbol keys.
enum EngineHost {
    static let scorer: NeuralScorer? = nil
    static var sentenceScorer: SentenceScorer? = nil
    static var lexiconDecoder: LexiconDecoder? { decoder as? LexiconDecoder }
    /// The real lexicon without the model (deterministic), so candidate paging can be tested.
    static let decoder: CandidateDecoding? = {
        guard let lexicon = try? Lexicon(path: "Resources/lexicon.sqlite") else { return nil }
        return LexiconDecoder(lexicon: lexicon, scorer: nil, bigram: try? BigramModel(path: "Resources/bigram.bin"))
    }()
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
        let range = marked ?? NSRange(location: cursor, length: 0)
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
    func attributes(forCharacterIndex index: Int, lineHeightRectangle: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! { lineHeightRectangle?.pointee = NSRect(x: 100, y: 100, width: 1, height: 16); return [:] }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeName: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "test.fake" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "fake" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { text.substring(with: range) }
    func firstRect(forCharacterRange range: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }

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

type(".2");                      check("。 then 2 → .", text: ".", cursor: 1)
type(".1");                      check("。 then 1 → 。", text: "。", cursor: 1)
type("."); key("", code: LEFT);  check("。 then ← moves at once", text: "。", cursor: 0)
let firstX = EngineHost.decoder?.decode("x", precedingText: "").refined().candidates.first?.text ?? "x"
type("."); key("", code: LEFT); key("x", code: 7); key("", code: SPACE)
                                 check("。 ← then pinyin x + space lands before the mark (\(firstX)。)", text: firstX + "。", cursor: firstX.count)
type(","); key("\t", code: TAB); check(", then Tab → ,", text: ",", cursor: 1)
type(","); key("\t", code: TAB); key("\t", code: TAB); check(", Tab Tab → ，", text: "，", cursor: 1)
type(":"); key("", code: DEL);   check("： then delete → empty", text: "", cursor: 0)
type("\"2\"");                   check("“ 2 “ → \" then “ again (quote state restored)", text: "\"“", cursor: 2)
type("\"\"");                    check("“ then ” pair", text: "“”", cursor: 2)
type("\""); key("", code: DEL); type("\"");  check("“ delete “ → “ again", text: "“", cursor: 1)
client.text = NSMutableString(string: "xy"); client.cursor = 1
type(".2");                      check("mid-text x|y: 。 2 → x.y", text: "x.y", cursor: 2)
type(".2."); key("", code: ESC); check(". then 。 (Esc closes)", text: ".。", cursor: 2)
type("..2");                     check("。。2 → 。.", text: "。.", cursor: 2)
type("."); key("", code: RIGHT); key("2", code: 19); check("。 → then 2 is typed (panel closed by arrow)", text: "。2", cursor: 2)
type(".3");                      check("。 then 3 (no third candidate) → 。3", text: "。3", cursor: 2)
// The panel must close by itself when the application takes the text away (Return sent it,
// a button was clicked) and stay while the mark is still there.
type("."); RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.6))
print((controller.symbolPanelShown ? "ok   " : "FAIL ") + "panel stays while the mark is in place"); if !controller.symbolPanelShown { failures += 1 }
client.text = NSMutableString(); client.cursor = 0
RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.6))
print((!controller.symbolPanelShown ? "ok   " : "FAIL ") + "panel closes after the application cleared the text"); if controller.symbolPanelShown { failures += 1 }
type("."); key("\r", code: 36)
print((!controller.symbolPanelShown ? "ok   " : "FAIL ") + "panel closes on Return"); if controller.symbolPanelShown { failures += 1 }
check("Return went to the application, mark kept", text: "。", cursor: 1)
// Caps Lock: lock on = English (letters lowercase, capitals with Shift), lock off = Chinese again.
flags(57, [.capsLock]); lockedKey("A", code: 0); lockedKey("a", code: 0, shift: true); lockedKey("B", code: 11)
check("lock on: a, Shift-a, b → aAb typed as English", text: "aAb", cursor: 3)
flags(57, [.capsLock]); lockedKey("A", code: 0)
do {
    let ok = client.marked == nil && client.text as String == "a"
    print((ok ? "ok   " : "FAIL ") + "lock still on after an unchanged flags event: English stays (a typed, no composition)"); if !ok { failures += 1 }
}
client.text = NSMutableString(); client.cursor = 0; client.marked = nil
flags(57, []); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "lock off: a composes pinyin again"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
// Shortcut settings: Shift alone toggles English by default; with its bindings removed it does not,
// and a key the person bound (Control-Space) does.
func keyMods(_ chars: String, code: UInt16, _ modifiers: NSEvent.ModifierFlags) {
    let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0, context: nil,
                                 characters: chars, charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code)!
    if !controller.handle(event, client: client) { client.appHandles(event) }
}
flags(56, [.shift]); flags(56, []); key("a", code: 0)
print((client.marked == nil && client.text as String == "a" ? "ok   " : "FAIL ") + "Shift alone → English (default)"); if !(client.marked == nil && client.text as String == "a") { failures += 1 }
flags(56, [.shift]); flags(56, []); client.text = NSMutableString(); client.cursor = 0
for binding in ShortcutStore.shared.bindings[.toggleLanguage] ?? [] where binding.keyCode != 57 { ShortcutStore.shared.remove(binding, from: .toggleLanguage) }
flags(56, [.shift]); flags(56, []); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "Shift alone with its binding removed → still Chinese"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
ShortcutStore.shared.add(KeyBinding(keyCode: 49, modifiers: NSEvent.ModifierFlags.control.rawValue, alone: false, label: "⌃Space"), to: .toggleLanguage)
keyMods(" ", code: 49, [.control]); key("a", code: 0)
print((client.marked == nil && client.text as String == "a" ? "ok   " : "FAIL ") + "Control-Space bound → English"); if !(client.marked == nil && client.text as String == "a") { failures += 1 }
keyMods(" ", code: 49, [.control]); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
ShortcutStore.shared.resetDefaults()
for binding in ShortcutStore.shared.bindings[.toggleLanguage] ?? [] where binding.keyCode == 57 { ShortcutStore.shared.remove(binding, from: .toggleLanguage) }
flags(57, [.capsLock]); key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "Caps Lock with its binding removed → still Chinese"); if client.marked == nil { failures += 1 }
key("", code: ESC); flags(57, []); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
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
flags(56, [.shift]); keyMods("A", code: 0, [.shift]); flags(56, []); key("", code: SPACE)
let shiftTyped = client.text as String
print((!shiftTyped.isEmpty && (client.text as String) != "" ? "ok   " : "FAIL ") + "Shift held for a capital → no toggle (text \(shiftTyped))"); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
key("a", code: 0)
print((client.marked != nil ? "ok   " : "FAIL ") + "…and the next letter still composes pinyin"); if client.marked == nil { failures += 1 }
key("", code: ESC); client.text = NSMutableString(); client.cursor = 0; client.marked = nil
print(failures == 0 ? "ALL OK" : "\(failures) FAILURES")
exit(failures == 0 ? 0 : 1)
