import Carbon
import Cocoa

/// Whether this input method should take the input source back after the system has replaced it
/// with the keyboard layout (ABC). A switch the user makes while the machine is awake stands.
struct InputSourceMemory {
    private(set) var wantsUs: Bool
    private var suspended = false

    init(wantsUs: Bool) { self.wantsUs = wantsUs }

    mutating func becameCurrent() { wantsUs = true }

    /// `stillCurrent`: the call is a client blur; the input source did not change.
    /// `userActed`: a key or a click just happened, so the change is the user's.
    /// While the machine is suspended the change is the system's fallback, and the wish stands.
    mutating func resigned(stillCurrent: Bool, userActed: Bool) {
        if stillCurrent || suspended || !userActed { return }
        wantsUs = false
    }

    mutating func beginSuspension() { suspended = true }

    /// A key arrived, so the machine is awake and this input method is still selected.
    mutating func noteAwake() { suspended = false }

    /// The first wake, unlock, or display-on after a suspension. A second notice in the same
    /// return does not ask again. The caller passes whether the source now showing is a keyboard
    /// layout rather than this input method or another one.
    mutating func endSuspension(currentIsKeyboardLayout: Bool) -> Bool {
        guard suspended else { return false }
        suspended = false
        return wantsUs && currentIsKeyboardLayout
    }

    /// A process that is starting, not waking.
    func restoreOnLaunch(currentIsKeyboardLayout: Bool) -> Bool {
        wantsUs && !suspended && currentIsKeyboardLayout
    }

    /// The source changed to the keyboard layout with the machine awake and no key or click.
    func restoreAfterDrop(currentIsKeyboardLayout: Bool, userActed: Bool) -> Bool {
        wantsUs && !suspended && !userActed && currentIsKeyboardLayout
    }
}

/// Selects this input method again when the system has put a keyboard layout in its place.
/// Started from `main`; the tests never start it, so they cannot change the machine's input source.
final class InputSourceKeeper: NSObject {
    static let shared = InputSourceKeeper()

    private let defaultsKey = "WantsInputSource"
    /// A key or a click this recent belongs to the user changing the input source.
    private let userActionWindow: TimeInterval = 3
    /// Typing refills the restore budget only after a restore has had time to settle.
    private let settleSeconds: TimeInterval = 5
    private var enabled = false
    private var memory = InputSourceMemory(wantsUs: false)
    private var attempts = 0
    private var restoreQueued = false
    private var lastRestoreUptime = -Double.infinity

    func start() {
        enabled = true
        memory = InputSourceMemory(wantsUs: UserDefaults.standard.bool(forKey: defaultsKey))
        let workspace = NSWorkspace.shared.notificationCenter
        workspace.addObserver(self, selector: #selector(suspend(_:)), name: NSWorkspace.willSleepNotification, object: nil)
        workspace.addObserver(self, selector: #selector(suspend(_:)), name: NSWorkspace.screensDidSleepNotification, object: nil)
        workspace.addObserver(self, selector: #selector(suspend(_:)), name: NSWorkspace.sessionDidResignActiveNotification, object: nil)
        workspace.addObserver(self, selector: #selector(resume(_:)), name: NSWorkspace.didWakeNotification, object: nil)
        workspace.addObserver(self, selector: #selector(resume(_:)), name: NSWorkspace.screensDidWakeNotification, object: nil)
        workspace.addObserver(self, selector: #selector(resume(_:)), name: NSWorkspace.sessionDidBecomeActiveNotification, object: nil)
        let distributed = DistributedNotificationCenter.default()
        distributed.addObserver(self, selector: #selector(suspend(_:)), name: Notification.Name("com.apple.screenIsLocked"), object: nil, suspensionBehavior: .deliverImmediately)
        distributed.addObserver(self, selector: #selector(resume(_:)), name: Notification.Name("com.apple.screenIsUnlocked"), object: nil, suspensionBehavior: .deliverImmediately)
        distributed.addObserver(self, selector: #selector(sourceChanged(_:)), name: Notification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String), object: nil, suspensionBehavior: .deliverImmediately)
        // The launch agent may still be settling the keyboard layout when this process appears.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in self?.restoreIfLaunching() }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in self?.restoreIfLaunching() }
    }

    func noteActivated() {
        guard enabled else { return }
        memory.becameCurrent()
        persist()
    }

    func noteDeactivated() {
        guard enabled else { return }
        apply(current())
    }

    /// A key reached this input method, so a restore has settled and a later drop may be undone again.
    func noteTyping() {
        guard enabled else { return }
        memory.noteAwake()
        guard ProcessInfo.processInfo.systemUptime - lastRestoreUptime > settleSeconds else { return }
        attempts = 0
    }

    @objc private func suspend(_: Notification) {
        memory.beginSuspension()
        attempts = 0
    }

    @objc private func resume(_: Notification) {
        let now = current()
        guard memory.endSuspension(currentIsKeyboardLayout: now.isKeyboardLayout && !now.isUs) else { return }
        restoreSoon()
    }

    @objc private func sourceChanged(_: Notification) {
        guard enabled else { return }
        apply(current())
    }

    private func restoreIfLaunching() {
        guard enabled else { return }
        let now = current()
        guard memory.restoreOnLaunch(currentIsKeyboardLayout: now.isKeyboardLayout && !now.isUs) else { return }
        restoreSoon()
    }

    private func apply(_ now: (isUs: Bool, isKeyboardLayout: Bool)) {
        if now.isUs {
            memory.becameCurrent()
            persist()
            return
        }
        let acted = userActedRecently()
        memory.resigned(stillCurrent: false, userActed: acted)
        persist()
        guard memory.restoreAfterDrop(currentIsKeyboardLayout: now.isKeyboardLayout, userActed: acted) else { return }
        restoreSoon()
    }

    private func restoreSoon() {
        guard !restoreQueued, attempts < 2 else { return }
        attempts += 1
        restoreQueued = true
        lastRestoreUptime = ProcessInfo.processInfo.systemUptime
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.restoreQueued = false
            let now = self.current()
            guard self.memory.wantsUs, !now.isUs, now.isKeyboardLayout else { return }
            self.selectThisInputMethod()
        }
    }

    private func userActedRecently() -> Bool {
        let kinds: [CGEventType] = [.keyDown, .flagsChanged, .leftMouseDown, .rightMouseDown, .otherMouseDown]
        return kinds.contains { CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: $0) < userActionWindow }
    }

    private func current() -> (isUs: Bool, isKeyboardLayout: Bool) {
        guard let source = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else { return (false, false) }
        let bundle = TISGetInputSourceProperty(source, kTISPropertyBundleID).map {
            Unmanaged<CFString>.fromOpaque($0).takeUnretainedValue() as String
        }
        let type = TISGetInputSourceProperty(source, kTISPropertyInputSourceType).map {
            Unmanaged<CFString>.fromOpaque($0).takeUnretainedValue() as String
        }
        return (bundle == Bundle.main.bundleIdentifier, type == kTISTypeKeyboardLayout as String)
    }

    private func selectThisInputMethod() {
        guard let bundleID = Bundle.main.bundleIdentifier else { return }
        let filter = [kTISPropertyBundleID as String: bundleID] as CFDictionary
        var sources = TISCreateInputSourceList(filter, false)?.takeRetainedValue() as? [TISInputSource] ?? []
        if sources.isEmpty {
            sources = TISCreateInputSourceList(filter, true)?.takeRetainedValue() as? [TISInputSource] ?? []
            guard let source = sources.first, TISEnableInputSource(source) == noErr else { return }
        }
        guard let source = sources.first else { return }
        let status = TISSelectInputSource(source)
        if status != noErr { NSLog("Typing: cannot select input source: \(status)") }
    }

    private func persist() {
        let defaults = UserDefaults.standard
        guard defaults.bool(forKey: defaultsKey) != memory.wantsUs else { return }
        defaults.set(memory.wantsUs, forKey: defaultsKey)
    }
}
