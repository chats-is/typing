import Cocoa
import Engine
import SwiftUI

/// The settings window: one page. Shortcuts, learning (switch, status, actions), the learning
/// data in two tabs with search and pages, and the about card at the end.
enum SettingsWindow {
    private static var controller: SettingsWindowController?

    static func show() {
        if controller == nil { controller = SettingsWindowController() }
        LearningStatus.shared.refresh()
        // A background-only app (LSBackgroundOnly, as input methods must be) cannot come forward until it
        // takes the accessory activation policy; without this the settings window opens behind.
        NSApp.setActivationPolicy(.accessory)
        controller?.showWindow(nil)
        controller?.window?.orderFrontRegardless()
        // Deprecated since macOS 14 (the compiler warns), but only this brings the window of a
        // background input method in front of the app being typed in: NSApp.activate() is cooperative
        // and the window stays behind (tested in the installed input method, 0.2.3).
        NSRunningApplication.current.activate(options: [.activateIgnoringOtherApps, .activateAllWindows])
        NSApp.activate(ignoringOtherApps: true)
        controller?.window?.makeKeyAndOrderFront(nil)
        // Nothing focused on opening: AppKit would otherwise put the cursor in the search field.
        DispatchQueue.main.async { controller?.window?.makeFirstResponder(nil) }
    }
}

final class SettingsWindowController: NSWindowController, NSWindowDelegate {
    init() {
        let host = NSHostingController(rootView: SettingsRootView())
        host.sizingOptions = []
        let window = NSWindow(contentViewController: host)
        window.title = "智能输入法"
        // The system's own title bar, drawn by each macOS in its style; the page scrolls under it.
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window.setContentSize(NSSize(width: 640, height: 680))
        window.minSize = NSSize(width: 560, height: 480)
        window.isReleasedWhenClosed = false
        super.init(window: window)
        window.delegate = self
        window.setFrameAutosaveName("Settings")
        window.center()
    }

    required init?(coder: NSCoder) { fatalError() }

    /// Coming back to the window (after typing elsewhere) shows what has been recorded since.
    func windowDidBecomeKey(_ notification: Notification) { LearningStatus.shared.refresh() }

    /// Back to a pure background process once the window closes, so the input method never
    /// competes with the app being typed in for the foreground.
    func windowWillClose(_ notification: Notification) { NSApp.setActivationPolicy(.prohibited) }
}

// MARK: - Learning data

/// What the learning cards show; refreshed when the window opens, after an action, after a session.
final class LearningStatus: ObservableObject {
    static let shared = LearningStatus()
    struct Change: Identifiable { let id: String; let keys: String; let learned: String; let general: String }
    /// One input: keys and text, typed `count` times (the store keeps it once).
    struct Record: Identifiable { let id: String; let keys: String; let text: String; let count: Int; let time: Double }
    enum Tab: Int, CaseIterable, Identifiable { case changes, records; var id: Int { rawValue }; var title: String { self == .changes ? "改变的首选词" : "训练用的输入" } }

    @Published var text = Learning.statusText()
    @Published var changes: [Change] = []
    @Published var records: [Record] = []
    @Published var loadingChanges = false
    @Published var loadingRecords = false
    private var recordsGeneration = 0
    @Published var training = Learning.isRunning
    @Published var tab: Tab = .changes { didSet { page = 0 } }
    @Published var query = "" { didSet { page = 0 } }
    @Published var page = 0
    /// The tag under the pointer, whose × shows fully.
    @Published var hovered: String?
    /// Small enough that the page needs no scrolling at the window's default height.
    static let pageSize = 30

    init() {
        Learning.sessionEnded = { [weak self] in self?.training = false; self?.refresh() }
    }

    func refresh() {
        text = Learning.statusText()
        loadingRecords = true
        recordsGeneration += 1
        let generation = recordsGeneration
        DispatchQueue.global(qos: .userInitiated).async {
            let found = Learning.store.entries().reversed().map {
                Record(id: "\($0.time)", keys: $0.keys, text: $0.text, count: $0.count, time: $0.time)
            }
            DispatchQueue.main.async {
                guard generation == self.recordsGeneration else { return }  // a newer load is under way
                self.records = found
                self.loadingRecords = false
                self.clampPage()
            }
        }
        loadingChanges = true
        Learning.requestLearnedChanges { found in
            self.changes = found.map { Change(id: $0.keys, keys: $0.keys, learned: $0.learned, general: $0.general) }
            self.loadingChanges = false
            self.clampPage()
        }
    }

    /// After a deletion the last page may be gone: show the last one there is.
    private func clampPage() { page = min(page, pageCount - 1) }

    /// The current tab's rows after the search box, and the page of them on show.
    var filteredChanges: [Change] {
        let q = query.trimmingCharacters(in: .whitespaces)
        return q.isEmpty ? changes : changes.filter { [$0.keys, $0.learned, $0.general].contains { $0.localizedCaseInsensitiveContains(q) } }
    }
    var filteredRecords: [Record] {
        let q = query.trimmingCharacters(in: .whitespaces)
        return q.isEmpty ? records : records.filter { $0.keys.localizedCaseInsensitiveContains(q) || $0.text.localizedCaseInsensitiveContains(q) }
    }
    var filteredCount: Int { tab == .changes ? filteredChanges.count : filteredRecords.count }
    /// The current tab's data is being read.
    var loading: Bool { tab == .changes ? loadingChanges : loadingRecords }
    var pageCount: Int { max(1, (filteredCount + Self.pageSize - 1) / Self.pageSize) }
    func pageRange(_ count: Int) -> Range<Int> {
        let start = min(page * Self.pageSize, max(0, count))
        return start ..< min(start + Self.pageSize, count)
    }
}

// MARK: - The page

/// One learned item as a tag; the × to remove it shows fully when the pointer is over it.
struct Tag: View {
    let id: String
    let label: String
    let detail: String?
    /// The detail is what the first choice used to be: shown struck through.
    var struck = false
    @ObservedObject var status: LearningStatus
    let remove: () -> Void
    private var hovered: Bool { status.hovered == id }

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
            if let detail { Text(detail).strikethrough(struck).foregroundStyle(.secondary) }
            // The × always has its place at the trailing end, so nothing moves when it shows.
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
                .opacity(hovered ? 1 : 0.25)
                .frame(width: 16)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Capsule().fill(hovered ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)))
        .onHover { over in status.hovered = over ? id : (status.hovered == id ? nil : status.hovered) }
    }
}

/// Lays its children out in rows, wrapping like words in a paragraph.
struct TagFlow: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 520
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > width { x = 0; y += rowHeight + spacing; rowHeight = 0 }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: width, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX && x + size.width > bounds.maxX { x = bounds.minX; y += rowHeight + spacing; rowHeight = 0 }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

/// The system search field.
struct SearchField: NSViewRepresentable {
    let prompt: String
    @Binding var text: String

    func makeNSView(context: Context) -> NSSearchField {
        let field = NSSearchField()
        field.placeholderString = prompt
        field.delegate = context.coordinator
        return field
    }

    func updateNSView(_ field: NSSearchField, context: Context) {
        if field.stringValue != text { field.stringValue = text }
    }

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    final class Coordinator: NSObject, NSSearchFieldDelegate {
        let text: Binding<String>
        init(text: Binding<String>) { self.text = text }
        func controlTextDidChange(_ notification: Notification) {
            if let field = notification.object as? NSSearchField { text.wrappedValue = field.stringValue }
        }
    }
}

struct SettingsRootView: View {
    @AppStorage("LearningEnabled") private var learningEnabled = true
    @ObservedObject private var status = LearningStatus.shared

    /// A grouped form: the system draws the cards, row insets and separators, each macOS in its own style.
    var body: some View {
        Form {
            ShortcutsSection()
            learningGroup
            dataGroup
            aboutGroup
        }
        .formStyle(.grouped)
    }

    private var learningGroup: some View {
        Section {
            HStack {
                Toggle("自动学习", isOn: $learningEnabled)
                    .toggleStyle(.switch)
                    .onChange(of: learningEnabled) { _, _ in Learning.enabledChanged(); status.refresh() }
            }
            HStack {
                Text(status.training ? "正在训练…" : status.text).font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("立即训练") { status.training = true; Learning.trainNow() }
                    .disabled(status.training || !learningEnabled)
                Button("重置学习") { Learning.reset(); status.refresh() }
                    .disabled(status.training)
            }
        }
    }

    private var dataGroup: some View {
        Section {
            HStack {
                Picker("", selection: $status.tab) {
                    ForEach(LearningStatus.Tab.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                Spacer()
                SearchField(prompt: "搜索拼音或文字", text: $status.query).frame(width: 220)
                Spacer()
                Button("刷新") { status.refresh() }
            }
            switch status.tab {
            case .changes:
                if status.loading && status.changes.isEmpty {
                    loadingRow
                } else if status.filteredChanges.isEmpty {
                    HStack { Text(status.query.isEmpty ? "还没有数据" : "没有匹配的").font(.callout).foregroundStyle(.secondary); Spacer() }
                } else {
                    let rows = status.filteredChanges
                    HStack {
                        TagFlow(spacing: 6) {
                            ForEach(rows[status.pageRange(rows.count)]) { change in
                                Tag(id: "c:" + change.id, label: "\(change.keys) \(change.learned)", detail: change.general, struck: true, status: status) {
                                    Learning.undo(keys: change.keys, word: change.learned); status.refresh()
                                }
                            }
                        }
                    }
                }
            case .records:
                if status.loading && status.records.isEmpty {
                    loadingRow
                } else if status.filteredRecords.isEmpty {
                    HStack { Text(status.query.isEmpty ? "还没有数据" : "没有匹配的").font(.callout).foregroundStyle(.secondary); Spacer() }
                } else {
                    let rows = status.filteredRecords
                    HStack {
                        TagFlow(spacing: 6) {
                            ForEach(rows[status.pageRange(rows.count)]) { record in
                                Tag(id: "r:" + record.id, label: record.keys.isEmpty ? record.text : "\(record.keys) \(record.text)", detail: record.count > 1 ? "×\(record.count)" : nil, status: status) {
                                    Learning.deleteEntries(times: [record.time]); status.refresh()
                                }
                            }
                        }
                    }
                }
            }
            if status.pageCount > 1 || status.filteredCount > 0 {
                HStack {
                    Text("共 \(status.filteredCount) 条").font(.callout).foregroundStyle(.secondary)
                    if status.loading { ProgressView().controlSize(.small) }
                    Spacer()
                    Button { status.page -= 1 } label: { Image(systemName: "chevron.left") }.disabled(status.page == 0)
                    Text("\(status.page + 1) / \(status.pageCount)").font(.callout).monospacedDigit()
                    Button { status.page += 1 } label: { Image(systemName: "chevron.right") }.disabled(status.page + 1 >= status.pageCount)
                }
            }
        }
    }

    private var loadingRow: some View {
        HStack {
            ProgressView().controlSize(.small)
            Text("正在加载…").font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var version: String {
        let info = Bundle.main.infoDictionary
        return "版本 \(info?["CFBundleShortVersionString"] as? String ?? "—")（\(info?["CFBundleVersion"] as? String ?? "—")）"
    }

    private var aboutGroup: some View {
        Section {
            HStack {
                Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 40, height: 40)
                VStack(alignment: .leading, spacing: 2) {
                    Link("智能输入法", destination: URL(string: "https://github.com/chats-is/typing")!).font(.system(size: 14, weight: .semibold))
                    Text("一款原生的 macOS 本地模型智能输入法").font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(version).font(.callout).foregroundStyle(.secondary)
                    Text("2026 Chats.is").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Shortcuts

/// Records the next key pressed for a function: a key with modifiers, or a modifier key alone.
final class ShortcutRecorder: ObservableObject {
    @Published var recording: ShortcutFunction?
    private var monitor: Any?
    private var pendingModifier: UInt16?

    private var closeObserver: Any?

    func start(_ function: ShortcutFunction) {
        stop()
        recording = function
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { [weak self] event in
            // Only keys meant for the settings window.
            guard event.window === NSApp.keyWindow else { return event }
            self?.handle(event, for: function)
            return nil
        }
        closeObserver = NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: nil, queue: .main) { [weak self] _ in self?.stop() }
    }

    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        monitor = nil
        closeObserver = nil
        recording = nil
        pendingModifier = nil
    }

    /// Keys that may be bound on their own: the function keys. Anything else needs ⌘, ⌃ or ⌥,
    /// or a plain letter, digit or Space bound to a function could never be typed again.
    private static let functionKeys: Set<UInt16> = [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111]

    private func handle(_ event: NSEvent, for function: ShortcutFunction) {
        let store = ShortcutStore.shared
        let modifiers = event.modifierFlags.intersection(ShortcutStore.relevant)
        if event.type == .flagsChanged {
            if event.keyCode == 57 {
                store.add(KeyBinding(keyCode: 57, modifiers: 0, alone: true, label: "Caps Lock"), to: function)
                stop()
            } else if let flag = ShortcutStore.modifierKeys[event.keyCode], modifiers == flag {
                pendingModifier = event.keyCode
            } else if modifiers.isEmpty, let pressed = pendingModifier, pressed == event.keyCode {
                store.add(KeyBinding(keyCode: pressed, modifiers: 0, alone: true, label: ShortcutStore.modifierNames[pressed] ?? "键 \(pressed)"), to: function)
                stop()
            } else {
                pendingModifier = nil
            }
            return
        }
        pendingModifier = nil
        if event.keyCode == 53, modifiers.isEmpty { stop(); return }
        guard !modifiers.intersection([.command, .control, .option]).isEmpty || Self.functionKeys.contains(event.keyCode) else {
            NSSound.beep()  // a plain key (or Shift + key) is for typing; keep recording
            return
        }
        store.add(KeyBinding(keyCode: event.keyCode, modifiers: modifiers.rawValue, alone: false, label: ShortcutStore.label(for: event)), to: function)
        stop()
    }
}

/// The key bindings: each function lists its keys as tags; "添加…" records the next key pressed
/// (a key with modifiers, or a modifier key alone), Esc cancels.
struct ShortcutsSection: View {
    @ObservedObject private var store = ShortcutStore.shared
    @StateObject private var recorder = ShortcutRecorder()

    var body: some View {
        Section {
            ForEach(ShortcutFunction.allCases, id: \.self) { function in
                HStack {
                    Text(function.title).frame(width: 96, alignment: .leading)
                    ForEach(store.bindings[function] ?? []) { binding in
                        // The same tag as the learning data's.
                        Tag(id: "k:\(function.rawValue):\(binding.id)", label: binding.label, detail: nil, status: LearningStatus.shared) {
                            store.remove(binding, from: function)
                        }
                    }
                    Spacer()
                    Button(recorder.recording == function ? "请按键…（Esc 取消）" : "添加…") {
                        if recorder.recording == function { recorder.stop() } else { recorder.start(function) }
                    }
                }
            }
            HStack {
                Text("按其中任意一个键执行对应的功能").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("恢复默认") { store.resetDefaults() }
            }
        }
        .onDisappear { recorder.stop() }
    }
}
