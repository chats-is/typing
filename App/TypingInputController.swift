import Carbon
import Cocoa
import Engine
import InputMethodKit

@objc(TypingInputController)
class TypingInputController: IMKInputController {
    static let panel = CandidatePanel()
    private static let pageSize = 9
    private static let noReplacement = NSRange(location: NSNotFound, length: NSNotFound)

    private static let punctuation: [Character: String] = [
        ",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；",
        "(": "（", ")": "）", "<": "《", ">": "》", "[": "【", "]": "】",
        "\\": "、", "^": "……", "_": "——", "$": "￥", "~": "～",
        "`": "·", "{": "「", "}": "」", "|": "｜",
    ]

    /// Shared with the candidate panel across controllers: a webpage that replaces its <input> on
    /// URL-parameter updates creates a new client controller mid-composition.
    private static let sharedSession = InputSession(decoder: EngineHost.decoder)
    private var session: InputSession { Self.sharedSession }
    private static var sharedPage = 0
    private var page: Int {
        get { Self.sharedPage }
        set { Self.sharedPage = newValue }
    }
    private static var sharedHighlight = 0
    private var highlight: Int {
        get { Self.sharedHighlight }
        set { Self.sharedHighlight = newValue }
    }
    /// Chinese or plain English typing: Shift alone toggles (the keys are set in Settings; Caps Lock,
    /// if it is bound, means its state: lock on = English). One state for every
    /// application, as with the system's input methods (a controller exists per client).
    private static var asciiMode = false
    private static var capsLockOn = false
    /// A modifier key pressed by itself, until it is released (or another key comes).
    private var modifierPressedAlone: UInt16?
    private var lastKeyWasDigit = false
    private var doubleQuoteOpen = false
    private var singleQuoteOpen = false
    /// The text just before the cursor, as far as it is known; context for the language model.
    private var recentText = ""
    /// What is being typed in English mode up to the next space, and the text before it: a word
    /// typed there is recorded for learning when it ends (only words are kept, see
    /// LearningStore.isEnglishWord).
    private var englishToken = ""
    private var englishContext = ""

    private static let recentTextLength = 60

    private enum Key {
        static let returnKey: UInt16 = 36, tab: UInt16 = 48, space: UInt16 = 49, delete: UInt16 = 51, escape: UInt16 = 53, capsLock: UInt16 = 57
        static let forwardDelete: UInt16 = 117
        static let enter: UInt16 = 76
        static let pageUp: UInt16 = 116, pageDown: UInt16 = 121
        static let left: UInt16 = 123, right: UInt16 = 124, down: UInt16 = 125, up: UInt16 = 126
    }

    // MARK: - IMK

    override func recognizedEvents(_ sender: Any!) -> Int {
        Int(NSEvent.EventTypeMask([.keyDown, .flagsChanged, .leftMouseDown, .rightMouseDown]).rawValue)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, let client = sender as? IMKTextInput else { return false }
        Self.isServerActive = true
        Self.deactivationID &+= 1
        session.onUpdate = { [weak self] in
            guard let self, self.session.isComposing, let client = self.client() else { return }
            self.updateUI(client, refined: true)
        }
        // A key may move the caret; the model's later update of the same composition does not.
        cachedCaret = nil
        Learning.noteActivity()
        switch event.type {
        case .leftMouseDown, .rightMouseDown:
            // A click moves the cursor: what is being composed is committed as typed and every panel closes.
            modifierPressedAlone = nil
            finishEnglishToken()
            commitRaw(client)
            Self.panel.hide()
            ModeIndicator.shared.hide()
            return false
        case .flagsChanged:
            handleFlagsChanged(event, client: client)
            return false
        case .keyDown:
            // (keyCode is only defined for key events.)
            if event.keyCode == Key.delete, !session.isComposing { Learning.noteDelete() }
            modifierPressedAlone = nil
            ModeIndicator.shared.hide()  // typing goes on: the candidates take its place
            let wasDigit = lastKeyWasDigit
            lastKeyWasDigit = false
            return handleKeyDown(event, client: client, afterDigit: wasDigit)
        default:
            return false
        }
    }

    // MARK: - Input menu

    override func menu() -> NSMenu! {
        let menu = NSMenu()
        let item = NSMenuItem(title: "打开设置…", action: #selector(openSettings(_:)), keyEquivalent: "")
        item.target = self
        // Shows the key set in Settings; the key itself is handled with the other shortcuts.
        if let shortcut = ShortcutStore.shared.menuKey(for: .openSettings) {
            item.keyEquivalent = shortcut.key
            item.keyEquivalentModifierMask = shortcut.modifiers
        }
        menu.addItem(item)
        return menu
    }

    @objc private func openSettings(_ sender: Any?) {
        SettingsWindow.show()
    }

    override func commitComposition(_ sender: Any!) {
        guard let client = sender as? IMKTextInput else { return }
        // A webpage <input> that updates URL parameters or rewrites its value on `input` triggers
        // discardMarkedText / commitComposition immediately after setMarkedText while no mouse
        // click has occurred; keep the composition and candidate panel open.
        if session.isComposing, Self.isReactingToMark(client: client) {
            Self.didDiscardMark = true
            return
        }
        commitRaw(client)
    }

    override func activateServer(_ sender: Any!) {
        Self.isServerActive = true
        Self.deactivationID &+= 1
        super.activateServer(sender)
        // The lock may have been switched while another input source was active.
        let on = NSEvent.modifierFlags.contains(.capsLock)
        if on != Self.capsLockOn {
            Self.capsLockOn = on
            if ShortcutStore.shared.function(forModifierKey: Key.capsLock) == .toggleLanguage { Self.asciiMode = on }
        }
    }

    override func deactivateServer(_ sender: Any!) {
        finishEnglishToken()
        // When a webpage switches URL parameters on `input`, the browser may blur/re-mount the
        // <input> and call deactivateServer before re-activating; defer closing briefly so the
        // re-mounted <input> keeps the composition, or close if nothing re-activates.
        if session.isComposing, Self.isReactingToMark(client: sender as? IMKTextInput) {
            Self.didDiscardMark = true
            Self.isServerActive = false
            Self.deactivationID &+= 1
            let id = Self.deactivationID
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.markReactionWindow) {
                if Self.deactivationID == id, !Self.isServerActive {
                    Self.hidePanel()
                }
            }
        } else {
            if let client = sender as? IMKTextInput { commitRaw(client) }
            Self.hidePanel()
        }
        recentText = ""
        super.deactivateServer(sender)
    }

    /// Time window after `setMarkedText` during which a client-initiated `commitComposition` or
    /// `deactivateServer` without any mouse/trackpad click or app/source switch is treated as the
    /// webpage reacting to the `input` event (e.g. updating URL parameters or re-mounting `<input>`).
    private static let markReactionWindow: TimeInterval = 0.25
    private static var lastMarkTime: TimeInterval = -.infinity
    private static var didDiscardMark = false
    private static var isServerActive = false
    private static var deactivationID = 0

    private static func isReactingToMark(client: IMKTextInput?) -> Bool {
        let elapsed = ProcessInfo.processInfo.systemUptime - lastMarkTime
        guard elapsed >= 0, elapsed < markReactionWindow,
              NSEvent.pressedMouseButtons == 0,
              isCurrentInputSource
        else { return false }
        if Bundle.main.bundleIdentifier != nil {
            for button in [CGEventType.leftMouseDown, .rightMouseDown, .otherMouseDown] {
                if CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: button) <= elapsed {
                    return false
                }
            }
            if let clientBundle = client?.bundleIdentifier(),
               let frontBundle = NSWorkspace.shared.frontmostApplication?.bundleIdentifier,
               clientBundle != frontBundle {
                return false
            }
        }
        return true
    }

    private static var isCurrentInputSource: Bool {
        guard let bundleID = Bundle.main.bundleIdentifier,
              let source = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
              let ptr = TISGetInputSourceProperty(source, kTISPropertyBundleID)
        else { return true }
        let currentID = Unmanaged<CFString>.fromOpaque(ptr).takeUnretainedValue() as String
        return currentID == bundleID
    }

    // MARK: - Keys

    private func handleFlagsChanged(_ event: NSEvent, client: IMKTextInput) {
        if event.keyCode == Key.capsLock {
            // Only a real change of the lock counts (the system may have taken the key itself).
            let on = event.modifierFlags.contains(.capsLock)
            if on != Self.capsLockOn {
                Self.capsLockOn = on
                modifierPressedAlone = nil
                if let function = ShortcutStore.shared.function(forModifierKey: Key.capsLock) {
                    _ = perform(function, lockState: on, client: client)
                }
            }
            return
        }
        // A modifier key pressed and released on its own, with nothing else in between.
        let modifiers = event.modifierFlags.intersection(ShortcutStore.relevant)
        if let flag = ShortcutStore.modifierKeys[event.keyCode], modifiers == flag {
            modifierPressedAlone = event.keyCode
        } else if modifiers.isEmpty, let pressed = modifierPressedAlone, pressed == event.keyCode {
            modifierPressedAlone = nil
            if let function = ShortcutStore.shared.function(forModifierKey: pressed) { _ = perform(function, client: client) }
        } else {
            modifierPressedAlone = nil
        }
    }

    /// Runs the function bound to a key; false when it does not apply right now.
    private func perform(_ function: ShortcutFunction, lockState: Bool? = nil, client: IMKTextInput) -> Bool {
        switch function {
        case .toggleLanguage:
            finishEnglishToken()
            commitRaw(client)
            // Caps Lock means its state (lock on = English); any other key toggles.
            Self.asciiMode = lockState ?? !Self.asciiMode
            ModeIndicator.shared.show(chinese: !Self.asciiMode, near: reportedCaret(client))
            return true
        case .openSettings:
            commitRaw(client)
            SettingsWindow.show()
            return true
        }
    }

    private func handleKeyDown(_ event: NSEvent, client: IMKTextInput, afterDigit: Bool) -> Bool {
        if let function = ShortcutStore.shared.function(forKeyDown: event), perform(function, client: client) { return true }
        if !event.modifierFlags.intersection([.command, .control, .option]).isEmpty {
            finishEnglishToken()
            commitRaw(client)
            recentText = ""
            return false
        }
        // Caps Lock is the capitals lock, as in other third-party input methods: with it on the
        // keyboard reports capitals, and capitals go into the text as they are (not as pinyin).
        let character = event.characters?.count == 1 ? event.characters?.first : nil
        if Self.asciiMode {
            trackEnglish(event, character: character, client: client)
            return false
        }
        // Lowercase letters are pinyin; capitals typed while composing stay as they are (对PC行业 from duiPChangye).
        if let character, character.isASCII, character.isLowercase || (session.isComposing && character.isUppercase) {
            if !session.isComposing {
                lastPreedit = ""
                Self.didDiscardMark = false
                let before = textBeforeCursor(client)
                if !before.isEmpty { recentText = before }
                session.precedingText = recentText
            }
            session.insert(character)
            page = 0
            highlight = 0
            updateUI(client)
            return true
        }
        // Deleting a quote just typed: the next quote key types the same one again (“ stays “).
        if !session.isComposing, event.keyCode == Key.delete, let last = recentText.last {
            if last == "\u{201C}" || last == "\u{201D}" { doubleQuoteOpen.toggle() }
            if last == "\u{2018}" || last == "\u{2019}" { singleQuoteOpen.toggle() }
        }
        return session.isComposing
            ? handleComposingKey(event, character: character, client: client)
            : handleIdleKey(character: character, afterDigit: afterDigit, client: client)
    }

    private func handleIdleKey(character: Character?, afterDigit: Bool, client: IMKTextInput) -> Bool {
        // Anything typed here other than pinyin separates the next word from the previous one.
        guard let character, character.isLetter || character.isNumber || character.isPunctuation || character.isSymbol || character == " " else {
            // Arrows, delete, return...: the cursor is no longer where the remembered text ends.
            recentText = ""
            return false
        }
        if character.isNumber {
            lastKeyWasDigit = true
            remember(String(character))
            return false
        }
        // Keep "3.14" and "12:30" typable.
        if afterDigit && (character == "." || character == ":") {
            remember(String(character))
            return false
        }
        // A "'" right after a letter belongs to the word (XI'an), as the system's Pinyin types it.
        if character == "'", let previous = textBeforeCursor(client).last ?? recentText.last, previous.isASCII, previous.isLetter {
            remember("'")
            return false
        }
        guard let text = punctuationText(for: character) else {
            remember(String(character))
            return false
        }
        insertMark(text, client: client)
        return true
    }

    private func handleComposingKey(_ event: NSEvent, character: Character?, client: IMKTextInput) -> Bool {
        let pageCount = (session.candidates.count + Self.pageSize - 1) / Self.pageSize
        let onPage = min(Self.pageSize, session.candidates.count - page * Self.pageSize)

        switch event.keyCode {
        case Key.space:
            if session.candidates.isEmpty { commitRaw(client) } else { select(page * Self.pageSize + highlight, client: client) }
            return true
        case Key.returnKey, Key.enter:
            sendLetters(client)
            return true
        case Key.delete where event.modifierFlags.contains(.shift), Key.forwardDelete where event.modifierFlags.contains(.shift):
            // Shift-Delete on a candidate: what was learned for it is taken back, as with its tag in
            // Settings; the candidates are read again without it.
            let index = page * Self.pageSize + highlight
            if session.candidates.indices.contains(index) {
                let candidate = session.candidates[index]
                Learning.undo(keys: session.pinyinKeys(consumed: candidate.consumed), word: candidate.text)
                session.refreshCandidates()
                page = 0
                highlight = 0
                updateUI(client)
            }
            return true
        case Key.delete:
            session.backspace()
            page = 0
            highlight = 0
            updateUI(client)
            return true
        case Key.escape:
            session.reset()
            updateUI(client)
            return true
        case Key.left:
            moveHighlight(-1, pageCount: pageCount, onPage: onPage, client: client)
            return true
        case Key.tab where event.modifierFlags.contains(.shift):
            // Shift-Tab walks the candidates backwards, as the system's Pinyin does.
            moveHighlight(-1, pageCount: pageCount, onPage: onPage, client: client)
            return true
        case Key.right, Key.tab:
            moveHighlight(1, pageCount: pageCount, onPage: onPage, client: client)
            return true
        case Key.up, Key.pageUp:
            turnPage(-1, pageCount: pageCount, client: client)
            return true
        case Key.down, Key.pageDown:
            turnPage(1, pageCount: pageCount, client: client)
            return true
        default:
            break
        }

        guard let character else { return true }
        // Shuangpin layouts that give ";" a final (ing in Microsoft and Sogou): after an initial key it
        // is the syllable's second key, not a mark.
        if character == ";", let layout = InputSession.shuangpin, layout.usesSemicolon, layout.waitsForFinal(session.remaining) {
            session.insert(";")
            page = 0
            highlight = 0
            updateUI(client)
            return true
        }
        // v and a number, a sum or a code point (QuickInput): digits and signs are part of it, not
        // candidate numbers or marks.
        if QuickInput.accepts(character, after: session.rawText) {
            session.insert(character)
            page = 0
            highlight = 0
            updateUI(client)
            return true
        }
        // 以词定字: [ and ] take the first and the last character of the highlighted word.
        let current = page * Self.pageSize + highlight
        let pickable = session.candidates.indices.contains(current) && session.candidates[current].text.count > 1
            && session.candidates[current].text.unicodeScalars.allSatisfy { Self.isHanzi($0) }
        switch character {
        case "[" where pickable, "]" where pickable:
            // Its letters stand for the whole word, so what is picked is no reading to learn.
            if let text = session.selectCharacter(current, last: character == "]") {
                commit(text, client: client, learned: false)
            } else {
                page = 0
                highlight = 0
                updateUI(client)
            }
        case "'":
            session.insert("'")
            page = 0
            highlight = 0
            updateUI(client)
        case "-" where page > 0:
            turnPage(-1, pageCount: pageCount, client: client)
        case "-":
            // On the first page there is nothing to turn back to: the letters go out as typed and
            // the hyphen follows (ai- → "ai-").
            sendLetters(client)
            remember("-")
            return false
        case "=":
            turnPage(1, pageCount: pageCount, client: client)
        case "1"..."9":
            let index = Int(character.asciiValue! - UInt8(ascii: "1"))
            if index < onPage { select(page * Self.pageSize + index, client: client) }
        default:
            // Anything else ends the composition with the highlighted candidate, then types itself.
            if !session.candidates.isEmpty, let text = session.select(page * Self.pageSize + highlight) {
                commit(text, client: client)
            } else {
                commitRaw(client)
            }
            if character.isNumber {
                // As when idle: a following . or : stays English (版本1.2, 12:30).
                lastKeyWasDigit = true
                remember(String(character))
                return false
            }
            guard character.isASCII, !character.isLetter, let text = punctuationText(for: character) else {
                recentText = ""
                return false
            }
            insertMark(text, client: client)
        }
        return true
    }

    private func punctuationText(for character: Character) -> String? {
        switch character {
        case "\"":
            doubleQuoteOpen.toggle()
            return doubleQuoteOpen ? "“" : "”"
        case "'":
            singleQuoteOpen.toggle()
            return singleQuoteOpen ? "‘" : "’"
        default:
            return Self.punctuation[character]
        }
    }

    private static func isHanzi(_ scalar: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(scalar.value) || (0x3400...0x4DBF).contains(scalar.value) || scalar.value >= 0x20000
    }

    // MARK: - Marks

    /// A punctuation key in Chinese mode: its Chinese mark goes into the text, as the system's
    /// Pinyin types it. There is no panel to swap it: that needs the application to report and
    /// replace its text, which a terminal does not; English marks come from English mode.
    private func insertMark(_ text: String, client: IMKTextInput) {
        client.insertText(text, replacementRange: Self.noReplacement)
        remember(text)
    }

    // MARK: - Composition

    /// One candidate left or right; past the end of a page it goes on to the next page's first
    /// candidate, before the start back to the previous page's last one.
    private func moveHighlight(_ delta: Int, pageCount: Int, onPage: Int, client: IMKTextInput) {
        let target = highlight + delta
        if target >= 0 && target < onPage {
            highlight = target
        } else if delta > 0 {
            guard page + 1 < pageCount else { return }
            page += 1
            highlight = 0
        } else {
            guard page > 0 else { return }
            page -= 1
            highlight = min(Self.pageSize, session.candidates.count - page * Self.pageSize) - 1
        }
        updateUI(client)
    }

    private func turnPage(_ delta: Int, pageCount: Int, client: IMKTextInput) {
        let target = page + delta
        guard target >= 0, target < pageCount else { return }
        page = target
        highlight = 0
        updateUI(client)
    }

    private func select(_ index: Int, client: IMKTextInput) {
        if let text = session.select(index) {
            commit(text, client: client, learned: !session.lastCommitNotLearned)
        } else {
            page = 0
            highlight = 0
            updateUI(client)
        }
    }

    /// Letters sent as typed on purpose (Return, "-" or "="): when they are an English word
    /// (don't, Xi'an) it is recorded for learning, so English is learned without switching to
    /// English mode.
    private func sendLetters(_ client: IMKTextInput) {
        let typed = session.typedText
        let context = session.precedingText
        commitRaw(client)
        if LearningStore.isEnglishWord(typed) {
            Learning.recordCommit(keys: typed.filter(\.isLetter).lowercased(), text: typed, context: context)
        }
    }

    /// Commits whatever is being composed as shown: the candidates picked so far plus the
    /// unconverted letters as typed, "'" included.
    private func commitRaw(_ client: IMKTextInput) {
        guard session.isComposing else { return }
        let text = session.typedText
        session.reset()
        commit(text, client: client, learned: false)
    }

    private func commit(_ text: String, client: IMKTextInput, learned: Bool = true) {
        let replacement = replacementRange(for: client)
        Self.didDiscardMark = false
        lastPreedit = ""
        client.insertText(text, replacementRange: replacement)
        remember(text)
        if learned, !session.lastCommittedKeys.isEmpty {
            Learning.recordCommit(keys: session.lastCommittedKeys, text: text, context: session.lastCommittedContext)
        }
        page = 0
        highlight = 0
        Self.panel.hide()
    }

    // MARK: - Context

    /// Follows what is typed in English mode: characters gather into a token, Delete takes the last
    /// one back, and a space, Return, Tab or any key that is not a character ends it.
    private func trackEnglish(_ event: NSEvent, character: Character?, client: IMKTextInput) {
        if event.keyCode == Key.delete {
            if !englishToken.isEmpty { englishToken.removeLast() }
            return
        }
        guard let character, !character.isWhitespace,
              character.unicodeScalars.allSatisfy({ $0.value >= 0x20 && !(0xF700...0xF8FF).contains($0.value) }) else {
            finishEnglishToken()
            return
        }
        if englishToken.isEmpty { englishContext = textBeforeCursor(client) }
        englishToken.append(character)
    }

    /// The token typed in English mode is over: recorded when it is a word (punctuation around it
    /// left out), forgotten otherwise.
    private func finishEnglishToken() {
        defer { englishToken = "" }
        let word = englishToken.trimmingCharacters(in: CharacterSet.punctuationCharacters.union(.symbols))
        guard LearningStore.isEnglishWord(word) else { return }
        Learning.recordCommit(keys: word.lowercased().filter(\.isLetter), text: word, context: englishContext)
    }

    private func remember(_ text: String) {
        recentText = String((recentText + text).suffix(Self.recentTextLength))
    }

    /// The text in front of the cursor, from clients that can report it; "" otherwise.
    private func textBeforeCursor(_ client: IMKTextInput) -> String {
        let selection = client.selectedRange()
        guard selection.location != NSNotFound, selection.location > 0 else { return "" }
        let start = max(0, selection.location - Self.recentTextLength)
        let range = NSRange(location: start, length: selection.location - start)
        return client.attributedSubstring(from: range)?.string ?? ""
    }

    /// Where the text cursor is on screen, found the same way as while composing; the candidate
    /// panel's last place when the application reports none (the model's later update included).
    private func caretRect(_ client: IMKTextInput) -> NSRect {
        if cachedCaret == nil { cachedCaret = queryCaret(client) }
        let rect = cachedCaret ?? .zero
        return rect.origin == .zero ? Self.panel.anchor : rect
    }

    /// Where the text cursor is as the application reports it; nil when it reports none (the
    /// candidate panel then keeps its last place; the language mark is not shown).
    private func reportedCaret(_ client: IMKTextInput) -> NSRect? {
        if cachedCaret == nil {
            cachedCaret = queryCaret(client)
        }
        guard let rect = cachedCaret, rect.origin != .zero else { return nil }
        return rect
    }

    /// The line rectangle at the composition's first character, as applications report it. A
    /// caret lies in one of the application's own windows: when that rectangle is missing
    /// (Spotlight) or does not lie in one, the first rectangle of the marked range is taken instead
    /// if it does, or, the windows being unknown, if it is on a screen (it is garbage in Chrome and
    /// a line lower in Terminal, so never first).
    private func queryCaret(_ client: IMKTextInput) -> NSRect {
        var line = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &line)
        let windows = Self.windowFrames(client)
        let place = windows.isEmpty ? NSScreen.screens.map(\.frame) : windows
        if line.origin != .zero && (windows.isEmpty || Self.lies(line, in: windows)) { return line }
        let marked = client.markedRange()
        guard marked.location != NSNotFound else { return line }
        let first = client.firstRect(forCharacterRange: NSRange(location: marked.location, length: 0), actualRange: nil)
        guard first.origin != .zero, first.height.isFinite, first.height > 0, first.height < 200,
              Self.lies(first, in: place) else { return line }
        return NSRect(x: first.minX, y: first.minY, width: 1, height: first.height)
    }

    /// Whether a caret rectangle starts inside one of the frames.
    private static func lies(_ rect: NSRect, in frames: [NSRect]) -> Bool {
        guard rect.minX.isFinite, rect.midY.isFinite else { return false }
        let point = NSPoint(x: rect.minX, y: rect.midY)
        return frames.contains { $0.contains(point) }
    }

    /// The on-screen windows of the client's application, in screen coordinates (empty when
    /// unknown: the reported rectangle is then used as it is). Replaceable for the tests.
    static var windowFrames: (IMKTextInput) -> [NSRect] = { client in
        guard let id = client.bundleIdentifier(), let primary = NSScreen.screens.first?.frame else { return [] }
        let pids = Set(NSRunningApplication.runningApplications(withBundleIdentifier: id).map(\.processIdentifier))
        guard !pids.isEmpty,
              let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]]
        else { return [] }
        return list.compactMap { window in
            guard let pid = window[kCGWindowOwnerPID as String] as? pid_t, pids.contains(pid),
                  let bounds = window[kCGWindowBounds as String] as? NSDictionary,
                  let frame = CGRect(dictionaryRepresentation: bounds)
            else { return nil }
            // Window bounds count down from the top of the primary screen.
            return NSRect(x: frame.minX, y: primary.maxY - frame.maxY, width: frame.width, height: frame.height)
        }
    }

    /// Closes the shared candidate panel and the language mark (another application came to the front).
    static func hidePanel() {
        sharedSession.reset()
        sharedPage = 0
        sharedHighlight = 0
        sharedLastPreedit = ""
        didDiscardMark = false
        panel.hide()
        ModeIndicator.shared.hide()
    }

    // MARK: - Display

    /// The caret rectangle fetched from the client for the current key, and the composition last
    /// sent: both client calls are synchronous round trips, so the model's later update of the
    /// same composition reuses the rectangle and skips the marked text when it has not changed.
    private var cachedCaret: NSRect?
    private static var sharedLastPreedit = ""
    private var lastPreedit: String {
        get { Self.sharedLastPreedit }
        set { Self.sharedLastPreedit = newValue }
    }

    /// The range to replace when updating or committing the composition: normally the client's own
    /// marked range (noReplacement), or, when the client has cleared its mark while rewriting its
    /// value on input, the preedit text still sitting before the cursor.
    private func replacementRange(for client: IMKTextInput) -> NSRange {
        guard Self.didDiscardMark, !lastPreedit.isEmpty, client.markedRange().location == NSNotFound else {
            return Self.noReplacement
        }
        let selection = client.selectedRange()
        for candidate in [lastPreedit, lastPreedit.filter { !$0.isWhitespace }] where !candidate.isEmpty {
            let length = candidate.utf16.count
            if selection.location != NSNotFound, selection.location >= length {
                let range = NSRange(location: selection.location - length, length: length)
                if client.attributedSubstring(from: range)?.string == candidate { return range }
            }
        }
        return Self.noReplacement
    }

    private func updateUI(_ client: IMKTextInput, refined: Bool = false) {
        let preedit = session.preedit
        if !refined, session.isComposing {
            Self.lastMarkTime = ProcessInfo.processInfo.systemUptime
        }
        if !refined || preedit != lastPreedit {
            let replacement = replacementRange(for: client)
            Self.didDiscardMark = false
            if preedit.isEmpty, replacement.location != NSNotFound {
                client.insertText("", replacementRange: replacement)
            } else {
                let attributes: [NSAttributedString.Key: Any] = [.underlineStyle: NSUnderlineStyle.single.rawValue]
                client.setMarkedText(
                    NSAttributedString(string: preedit, attributes: attributes),
                    selectionRange: NSRange(location: preedit.utf16.count, length: 0),
                    replacementRange: replacement
                )
            }
            lastPreedit = preedit
        }

        let candidates = session.candidates
        guard !candidates.isEmpty else {
            Self.panel.hide()
            return
        }
        // The candidates may have become fewer since the page and highlight were chosen (a
        // separator typed, the model's reorder): keep both inside the list.
        let pageCount = (candidates.count + Self.pageSize - 1) / Self.pageSize
        page = min(page, pageCount - 1)
        let start = page * Self.pageSize
        let end = min(start + Self.pageSize, candidates.count)
        highlight = min(highlight, end - start - 1)
        Self.panel.show(
            candidates: candidates[start..<end].map(\.text),
            highlighted: highlight,
            hasMore: end < candidates.count,
            near: caretRect(client)
        )
    }
}
