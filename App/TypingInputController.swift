import Cocoa
import Engine
import InputMethodKit

@objc(TypingInputController)
class TypingInputController: IMKInputController {
    private static let panel = CandidatePanel()
    private static let pageSize = 9
    private static let noReplacement = NSRange(location: NSNotFound, length: NSNotFound)

    private static let punctuation: [Character: String] = [
        ",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；",
        "(": "（", ")": "）", "<": "《", ">": "》", "[": "【", "]": "】",
        "\\": "、", "^": "……", "_": "——", "$": "￥", "~": "～",
    ]

    private let session: InputSession = {
        let session = InputSession(decoder: EngineHost.decoder)
        return session
    }()
    private var page = 0
    private var highlight = 0
    /// Chinese or plain English typing: Caps Lock switches like the system's Chinese keyboards
    /// (lock on = English, off = Chinese), Shift alone toggles as well. One state for every
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
    /// A mark just typed with its alternative, the key's own character, as candidates 1 and 2 in
    /// the panel. It is shown as marked text until it finishes: a digit or Tab picks the other one;
    /// Esc, any other key, a click or `symbolFinishDelay` without a key finishes it as shown.
    private var symbol: (candidates: [String], choice: Int)?
    private var symbolTimer: Timer?
    /// A mark usually ends what is being typed: left alone it finishes by itself. Typing on finishes
    /// it at once, so this only matters when the person stops, to look at the panel and choose.
    static let symbolFinishDelay: TimeInterval = 3
    /// For tests: whether the symbol panel is up.
    var symbolPanelShown: Bool { symbol != nil }

    private static let recentTextLength = 60

    private enum Key {
        static let returnKey: UInt16 = 36, tab: UInt16 = 48, space: UInt16 = 49, delete: UInt16 = 51, escape: UInt16 = 53, capsLock: UInt16 = 57
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
        if session.onUpdate == nil {
            session.onUpdate = { [weak self] in
                guard let self, self.session.isComposing, let client = self.client() else { return }
                self.updateUI(client, refined: true)
            }
        }
        // A key may move the caret; the model's later update of the same composition does not.
        cachedCaret = nil
        Learning.noteActivity()
        switch event.type {
        case .leftMouseDown, .rightMouseDown:
            // A click moves the cursor: what is being composed is committed as typed and every panel closes.
            modifierPressedAlone = nil
            commitRaw(client)
            Self.panel.hide()
            return false
        case .flagsChanged:
            handleFlagsChanged(event, client: client)
            return false
        case .keyDown:
            // (keyCode is only defined for key events.)
            if event.keyCode == Key.delete, !session.isComposing, symbol == nil { Learning.noteDelete() }
            modifierPressedAlone = nil
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
        commitRaw(client)
    }

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        // The lock may have been switched while another input source was active.
        let on = NSEvent.modifierFlags.contains(.capsLock)
        if on != Self.capsLockOn {
            Self.capsLockOn = on
            if ShortcutStore.shared.function(forModifierKey: Key.capsLock) == .toggleLanguage { Self.asciiMode = on }
        }
    }

    deinit {
        symbolTimer?.invalidate()
    }

    override func deactivateServer(_ sender: Any!) {
        if let client = sender as? IMKTextInput { commitRaw(client) }
        recentText = ""
        dismissSymbol()
        super.deactivateServer(sender)
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

    /// Runs a function the person bound to a key; false when it does not apply right now.
    private func perform(_ function: ShortcutFunction, lockState: Bool? = nil, client: IMKTextInput) -> Bool {
        switch function {
        case .toggleLanguage:
            commitRaw(client)
            // Caps Lock means its state (lock on = English); any other key toggles.
            Self.asciiMode = lockState ?? !Self.asciiMode
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
            commitRaw(client)
            recentText = ""
            return false
        }
        var character = event.characters?.count == 1 ? event.characters?.first : nil
        // With the lock on, the keyboard reports capitals; the person means ordinary letters
        // (pinyin, or English in English mode) and capitals only with Shift.
        let lockedLetter = event.modifierFlags.contains(.capsLock) && character.map { $0.isASCII && $0.isLetter } == true
        if lockedLetter, let letter = character {
            character = event.modifierFlags.contains(.shift) ? Character(letter.uppercased()) : Character(letter.lowercased())
        }
        if Self.asciiMode {
            guard lockedLetter, let letter = character else { return false }
            client.insertText(String(letter), replacementRange: Self.noReplacement)
            remember(String(letter))
            return true
        }
        if symbol != nil, let handled = handleSymbolKey(event, character: character, client: client) {
            return handled
        }
        // Lowercase letters are pinyin; capitals typed while composing stay as they are (对PC行业 from duiPChangye).
        if let character, character.isASCII, character.isLowercase || (session.isComposing && character.isUppercase) {
            if !session.isComposing {
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
        guard let text = punctuationText(for: character) else {
            remember(String(character))
            return false
        }
        composeSymbol(text, typed: character, client: client)
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
            commitRaw(client)
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
        switch character {
        case "'":
            session.insert("'")
            page = 0
            highlight = 0
            updateUI(client)
        case "-":
            turnPage(-1, pageCount: pageCount, client: client)
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
            composeSymbol(text, typed: character, client: client)
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

    // MARK: - Symbols

    /// A mark that was just typed, with its alternative (the key's own character). It stays marked
    /// text while the panel is up: an application hands every key to the input method while
    /// something is being composed, but with the mark already in the text some (a terminal, for
    /// one) keep keys such as Esc for themselves.
    private func composeSymbol(_ text: String, typed character: Character, client: IMKTextInput) {
        let other = String(character)
        remember(text)
        guard other != text else {
            client.insertText(text, replacementRange: Self.noReplacement)
            return
        }
        symbol = ([text, other], 0)
        showSymbol(client)
    }
    /// Shows the chosen mark as marked text with the panel beside it, and starts the clock that
    /// finishes it.
    private func showSymbol(_ client: IMKTextInput) {
        guard let current = symbol else { return }
        let mark = current.candidates[current.choice]
        client.setMarkedText(mark, selectionRange: NSRange(location: mark.utf16.count, length: 0), replacementRange: Self.noReplacement)
        Self.panel.show(candidates: current.candidates, highlighted: current.choice, hasMore: false, near: caretRect(client))
        symbolTimer?.invalidate()
        symbolTimer = Timer.scheduledTimer(withTimeInterval: Self.symbolFinishDelay, repeats: false) { [weak self] _ in
            guard let self else { return }
            if let client = self.client() { self.finishSymbol(client) } else { self.dismissSymbol() }
        }
    }
    /// The mark goes into the text as shown; the panel closes.
    private func finishSymbol(_ client: IMKTextInput) {
        guard let current = symbol else { return }
        dismissSymbol()
        client.insertText(current.candidates[current.choice], replacementRange: Self.noReplacement)
    }
    private func handleSymbolKey(_ event: NSEvent, character: Character?, client: IMKTextInput) -> Bool? {
        guard let current = symbol else { return nil }
        let count = current.candidates.count
        switch event.keyCode {
        case Key.tab:
            let step = event.modifierFlags.contains(.shift) ? count - 1 : 1
            swapSymbol(to: (current.choice + step) % count, client: client)
            return true
        case Key.escape:
            finishSymbol(client)
            return true
        case Key.delete:
            // The mark goes with the key; a quote's open/close state is as before it was typed.
            let mark = current.candidates[current.choice]
            if Self.isDoubleQuote(mark) { doubleQuoteOpen.toggle() }
            if Self.isSingleQuote(mark) { singleQuoteOpen.toggle() }
            recentText = String(recentText.dropLast(mark.count))
            dismissSymbol()
            client.setMarkedText("", selectionRange: NSRange(location: 0, length: 0), replacementRange: Self.noReplacement)
            return true
        default:
            break
        }
        if let character, let digit = character.wholeNumberValue, digit >= 1, digit <= count {
            swapSymbol(to: digit - 1, client: client)
            finishSymbol(client)
            return true
        }
        // Any other key finishes the mark and is then handled as usual.
        finishSymbol(client)
        return nil
    }
    /// Shows the candidate at `choice` in place of the mark.
    private func swapSymbol(to choice: Int, client: IMKTextInput) {
        guard var current = symbol, choice != current.choice else { return }
        let from = current.candidates[current.choice], to = current.candidates[choice]
        // The Chinese quotes' open/close state follows what is in the text.
        if Self.isDoubleQuote(from) != Self.isDoubleQuote(to) { doubleQuoteOpen.toggle() }
        if Self.isSingleQuote(from) != Self.isSingleQuote(to) { singleQuoteOpen.toggle() }
        recentText = String(recentText.dropLast(from.count)) + to
        current.choice = choice
        symbol = current
        showSymbol(client)
    }
    /// Closes the panel and forgets the mark (the caller has put it in the text or taken it away).
    private func dismissSymbol() {
        symbolTimer?.invalidate()
        symbolTimer = nil
        guard symbol != nil else { return }
        symbol = nil
        Self.panel.hide()
    }
    private static func isDoubleQuote(_ text: String) -> Bool { text == "\u{201C}" || text == "\u{201D}" }
    private static func isSingleQuote(_ text: String) -> Bool { text == "\u{2018}" || text == "\u{2019}" }

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
            commit(text, client: client)
        } else {
            page = 0
            highlight = 0
            updateUI(client)
        }
    }

    /// Commits whatever is being composed as shown: a mark, or the candidates picked so far plus
    /// the unconverted pinyin as typed.
    private func commitRaw(_ client: IMKTextInput) {
        finishSymbol(client)
        guard session.isComposing else { return }
        let text = session.rawText.replacingOccurrences(of: "'", with: "")
        session.reset()
        commit(text, client: client, learned: false)
    }

    private func commit(_ text: String, client: IMKTextInput, learned: Bool = true) {
        client.insertText(text, replacementRange: Self.noReplacement)
        remember(text)
        if learned, !session.lastCommittedKeys.isEmpty {
            Learning.recordCommit(keys: session.lastCommittedKeys, text: text, context: session.lastCommittedContext)
        }
        page = 0
        highlight = 0
        Self.panel.hide()
    }

    // MARK: - Context

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

    /// Where the text cursor is on screen, found the same way as while composing
    /// (asking for the rect of the selected range returns garbage in some applications).
    private func caretRect(_ client: IMKTextInput) -> NSRect {
        if let cachedCaret { return cachedCaret }
        var rect = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &rect)
        cachedCaret = rect
        return rect.origin == .zero ? Self.panel.anchor : rect
    }

    /// Closes the shared candidate panel (another application came to the front).
    static func hidePanel() { panel.hide() }

    // MARK: - Display

    /// The caret rectangle fetched from the client for the current key, and the composition last
    /// sent: both client calls are synchronous round trips, so the model's later update of the
    /// same composition reuses the rectangle and skips the marked text when it has not changed.
    private var cachedCaret: NSRect?
    private var lastPreedit = ""

    private func updateUI(_ client: IMKTextInput, refined: Bool = false) {
        let preedit = session.preedit
        if !refined || preedit != lastPreedit {
            let attributes: [NSAttributedString.Key: Any] = [.underlineStyle: NSUnderlineStyle.single.rawValue]
            client.setMarkedText(
                NSAttributedString(string: preedit, attributes: attributes),
                selectionRange: NSRange(location: preedit.utf16.count, length: 0),
                replacementRange: Self.noReplacement
            )
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
