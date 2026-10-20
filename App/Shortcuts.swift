import Cocoa

/// A key the person assigned to a function: a key with modifiers, or a modifier key by itself.
struct KeyBinding: Codable, Equatable, Identifiable {
    var keyCode: UInt16
    /// NSEvent.ModifierFlags raw value, command/control/option/shift only.
    var modifiers: UInt
    /// A modifier key pressed and released on its own (Shift, Caps Lock, ...).
    var alone: Bool
    var label: String
    var id: String { "\(keyCode)-\(modifiers)-\(alone)" }
}

enum ShortcutFunction: String, CaseIterable, Codable, Identifiable {
    case toggleLanguage
    case openSettings
    var id: String { rawValue }
    var title: String {
        switch self {
        case .toggleLanguage: return "中英文切换"
        case .openSettings: return "打开设置"
        }
    }
}

/// The person's key bindings, kept in the defaults; read by the input controller on every key and
/// edited in Settings. A key bound to one function is taken away from any other.
final class ShortcutStore: ObservableObject {
    static let shared = ShortcutStore()
    private static let defaultsKey = "ShortcutBindings"
    private static let capsLockMigratedKey = "CapsLockIsCapitalsLock"

    static let modifierKeys: [UInt16: NSEvent.ModifierFlags] = [
        54: .command, 55: .command, 56: .shift, 60: .shift, 58: .option, 61: .option, 59: .control, 62: .control,
    ]
    static let modifierNames: [UInt16: String] = [
        54: "右 Command", 55: "左 Command", 56: "左 Shift", 60: "右 Shift", 58: "左 Option", 61: "右 Option",
        59: "左 Control", 62: "右 Control", 57: "Caps Lock",
    ]
    static let relevant: NSEvent.ModifierFlags = [.command, .control, .option, .shift]

    static let defaults: [ShortcutFunction: [KeyBinding]] = [
        .toggleLanguage: [
            KeyBinding(keyCode: 56, modifiers: 0, alone: true, label: "左 Shift"),
            KeyBinding(keyCode: 60, modifiers: 0, alone: true, label: "右 Shift"),
        ],
        // ⌃⌥, : comma is the usual key for settings; ⌘, belongs to every app's own settings.
        .openSettings: [
            KeyBinding(keyCode: 43, modifiers: NSEvent.ModifierFlags([.control, .option]).rawValue, alone: false, label: "⌃⌥,"),
        ],
    ]

    @Published private(set) var bindings: [ShortcutFunction: [KeyBinding]]

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.defaultsKey),
           let stored = try? JSONDecoder().decode([String: [KeyBinding]].self, from: data) {
            var loaded: [ShortcutFunction: [KeyBinding]] = [:]
            for function in ShortcutFunction.allCases { loaded[function] = stored[function.rawValue] ?? Self.defaults[function] ?? [] }
            bindings = loaded
        } else {
            bindings = Self.defaults
        }
        // Caps Lock switched languages by default until 2026-10-20; it is the capitals lock now, as in
        // other third-party input methods. Once: take it off language switching, and give Shift back
        // when nothing else switches.
        if !UserDefaults.standard.bool(forKey: Self.capsLockMigratedKey) {
            if bindings[.toggleLanguage]?.contains(where: { $0.keyCode == 57 }) == true {
                bindings[.toggleLanguage]?.removeAll { $0.keyCode == 57 }
                if bindings[.toggleLanguage]?.isEmpty ?? true { bindings[.toggleLanguage] = Self.defaults[.toggleLanguage] }
                save()
            }
            UserDefaults.standard.set(true, forKey: Self.capsLockMigratedKey)
        }
        // Keys of the earlier, picker-based settings.
        for key in ["ToggleCapsLock", "ToggleShift", "PageKeys", "SuggestionsEnabled"] { UserDefaults.standard.removeObject(forKey: key) }
    }

    func add(_ binding: KeyBinding, to function: ShortcutFunction) {
        for other in ShortcutFunction.allCases { bindings[other]?.removeAll { $0.id == binding.id } }
        bindings[function, default: []].append(binding)
        save()
    }

    func remove(_ binding: KeyBinding, from function: ShortcutFunction) {
        bindings[function]?.removeAll { $0.id == binding.id }
        save()
    }

    func resetDefaults() {
        bindings = Self.defaults
        save()
    }

    private func save() {
        var stored: [String: [KeyBinding]] = [:]
        for (function, list) in bindings { stored[function.rawValue] = list }
        if let data = try? JSONEncoder().encode(stored) { UserDefaults.standard.set(data, forKey: Self.defaultsKey) }
    }

    /// The function bound to a key pressed with these modifiers, if any.
    func function(forKeyDown event: NSEvent) -> ShortcutFunction? {
        let modifiers = event.modifierFlags.intersection(Self.relevant).rawValue
        for (function, list) in bindings where list.contains(where: { !$0.alone && $0.keyCode == event.keyCode && $0.modifiers == modifiers }) {
            return function
        }
        return nil
    }

    /// The first key with modifiers bound to `function`, as a menu item shows it (only keys that
    /// are one character, which is how a menu item takes them).
    func menuKey(for function: ShortcutFunction) -> (key: String, modifiers: NSEvent.ModifierFlags)? {
        for binding in bindings[function] ?? [] where !binding.alone {
            let key = binding.label.drop { "⌃⌥⇧⌘".contains($0) }
            if key.count == 1 { return (key.lowercased(), NSEvent.ModifierFlags(rawValue: binding.modifiers)) }
        }
        return nil
    }

    /// The function bound to a modifier key pressed on its own, if any.
    func function(forModifierKey keyCode: UInt16) -> ShortcutFunction? {
        for (function, list) in bindings where list.contains(where: { $0.alone && $0.keyCode == keyCode }) { return function }
        return nil
    }

    /// How a key is shown: its modifiers as symbols, then the key's name.
    static func label(for event: NSEvent) -> String {
        let modifiers = event.modifierFlags.intersection(relevant)
        var text = ""
        if modifiers.contains(.control) { text += "⌃" }
        if modifiers.contains(.option) { text += "⌥" }
        if modifiers.contains(.shift) { text += "⇧" }
        if modifiers.contains(.command) { text += "⌘" }
        let names: [UInt16: String] = [
            49: "Space", 48: "Tab", 36: "Return", 76: "Enter", 51: "Delete", 117: "⌦", 53: "Esc",
            123: "←", 124: "→", 125: "↓", 126: "↑", 116: "PageUp", 121: "PageDown", 115: "Home", 119: "End",
            122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6", 98: "F7", 100: "F8", 101: "F9", 109: "F10", 103: "F11", 111: "F12",
        ]
        if let name = names[event.keyCode] { return text + name }
        let typed = (event.charactersIgnoringModifiers ?? "").uppercased()
        return text + (typed.isEmpty ? "键 \(event.keyCode)" : typed)
    }
}
