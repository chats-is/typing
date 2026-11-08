import Carbon
import Darwin

// Stop using this input method, then turn it off. Exit 1 if it is still in use or still on.
let bundleID = "is.chats.inputmethod.Typing"

func property(_ source: TISInputSource, _ key: CFString) -> String? {
    guard let raw = TISGetInputSourceProperty(source, key) else { return nil }
    return Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String
}

func flag(_ source: TISInputSource, _ key: CFString) -> Bool {
    guard let raw = TISGetInputSourceProperty(source, key) else { return false }
    return CFBooleanGetValue(Unmanaged<CFBoolean>.fromOpaque(raw).takeUnretainedValue())
}

func isOurs(_ source: TISInputSource) -> Bool {
    property(source, kTISPropertyBundleID) == bundleID
}

func ourSources() -> [TISInputSource] {
    let filter = [kTISPropertyBundleID as String: bundleID] as CFDictionary
    return TISCreateInputSourceList(filter, true)?.takeRetainedValue() as? [TISInputSource] ?? []
}

func enabled(_ source: TISInputSource) -> Bool {
    flag(source, kTISPropertyInputSourceIsEnabled)
}

func isKeyboardLayout(_ source: TISInputSource) -> Bool {
    property(source, kTISPropertyInputSourceType) == (kTISTypeKeyboardLayout as String)
}

func current() -> TISInputSource? {
    TISCopyCurrentKeyboardInputSource()?.takeRetainedValue()
}

// The current input source cannot be turned off. Leave it first.
// Another input method is used when one is on, so this process does not take a keyboard layout back.
func leaveIfCurrent() {
    guard let current = current(), isOurs(current) else { return }
    let others = (TISCreateInputSourceList(nil, false)?.takeRetainedValue() as? [TISInputSource] ?? [])
        .filter { !isOurs($0) && flag($0, kTISPropertyInputSourceIsSelectCapable) }
    guard let next = others.first(where: { !isKeyboardLayout($0) }) ?? others.first else { return }
    _ = TISSelectInputSource(next)
}

func turnedOff() -> Bool {
    let stillOn = ourSources().contains { enabled($0) }
    let stillCurrent = current().map(isOurs) ?? false
    return !stillOn && !stillCurrent
}

for _ in 0..<10 {
    leaveIfCurrent()
    for source in ourSources() where enabled(source) {
        if TISDisableInputSource(source) != noErr {
            break
        }
    }
    if turnedOff() {
        exit(0)
    }
    usleep(100_000)
}
exit(1)
