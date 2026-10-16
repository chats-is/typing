import Cocoa
import Engine
import SwiftUI

/// The settings window.
enum SettingsWindow {
    private static var window: NSWindow?

    static func show() {
        if window == nil {
            let controller = NSHostingController(rootView: SettingsView())
            let window = NSWindow(contentViewController: controller)
            window.title = "智能输入 设置"
            window.styleMask = [.titled, .closable, .miniaturizable]
            window.setContentSize(NSSize(width: 760, height: 600))
            window.isReleasedWhenClosed = false
            window.center()
            Self.window = window
        }
        LearningStatus.shared.refresh()
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }
}

/// What the learning section shows: the status line, the changes learning made and the material it
/// learned from; refreshed when the window opens or after a correction.
final class LearningStatus: ObservableObject {
    static let shared = LearningStatus()
    struct Change: Identifiable { let id: String; let keys: String; let learned: String; let general: String }
    struct Record: Identifiable { let id: Double; let keys: String; let text: String }
    @Published var text = Learning.statusText()
    @Published var changes: [Change] = []
    @Published var records: [Record] = []
    @Published var loading = false
    @Published var hovered: String?

    func refresh() {
        text = Learning.statusText()
        records = Learning.store.entries().suffix(200).reversed().map { Record(id: $0.time, keys: $0.keys, text: $0.text) }
        loading = true
        Learning.requestLearnedChanges { found in
            self.changes = found.map { Change(id: $0.keys, keys: $0.keys, learned: $0.learned, general: $0.general) }
            self.loading = false
        }
    }
}

enum SettingsPage: String, CaseIterable, Identifiable {
    case general, learning
    var id: String { rawValue }
    var title: String { self == .general ? "通用" : "学习" }
    var symbol: String { self == .general ? "keyboard" : "brain.head.profile" }
}

/// Which page is shown (a class: with this SDK, assigning a @State value from a button's closure
/// does not compile in a package build).
final class SettingsSelection: ObservableObject { @Published var page = SettingsPage.general }

struct SettingsView: View {
    @StateObject private var selection = SettingsSelection()

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(SettingsPage.allCases) { item in
                    Button { selection.page = item } label: {
                        Label(item.title, systemImage: item.symbol)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(RoundedRectangle(cornerRadius: 6).fill(selection.page == item ? Color.accentColor.opacity(0.18) : Color.clear))
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
            .padding(10)
            .frame(width: 170)
            .background(Color(nsColor: .windowBackgroundColor))
            Divider()
            Group {
                switch selection.page {
                case .general: GeneralSettingsView()
                case .learning: LearningSettingsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 720, minHeight: 560)
    }
}

/// Basic settings: keys. More (candidates per page, font size) belong here as they come.
struct GeneralSettingsView: View {
    var body: some View {
        Form {
            ShortcutsSection()
            Section {
                Text("词库决定能打出什么，语言模型按上下文排序。不联网，不记录你的选择。")
                    .font(.caption).foregroundStyle(.secondary)
            } header: {
                Text("关于").font(.headline)
            }
        }
        .formStyle(.grouped)
    }
}

/// Learning: the switch and status, what learning changed, what it learned from.
struct LearningSettingsView: View {
    @AppStorage("LearningEnabled") private var learningEnabled = true
    @ObservedObject private var status = LearningStatus.shared

    var body: some View {
        Form {
            Section {
                Toggle("学习我的输入习惯", isOn: $learningEnabled)
                    .onChange(of: learningEnabled) { _, _ in Learning.enabledChanged(); status.refresh() }
                HStack {
                    Text(status.text).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("重置学习") { Learning.reset(); status.refresh() }
                }
            } header: {
                Text("学习").font(.headline)
            } footer: {
                Text("学习只在这台电脑上，用你确认过的文字微调语言模型的最上两层；每次学习后都用你最近的输入和一组大众句子检验，变差就回滚。关闭后立即恢复出厂模型；重置会删除学习记录和学到的参数。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                if status.loading {
                    Text("正在对照…").font(.caption).foregroundStyle(.secondary)
                } else if status.changes.isEmpty {
                    Text("还没有因学习而改变的首选。").font(.caption).foregroundStyle(.secondary)
                } else {
                    TagFlow(spacing: 6) {
                        ForEach(status.changes) { change in
                            Tag(id: "c:" + change.id, label: "\(change.keys) \(change.learned)", detail: change.general, struck: true, status: status) {
                                Learning.undo(word: change.learned); status.refresh()
                            }
                        }
                    }
                }
            } header: {
                Text("学到的变化").font(.headline)
            } footer: {
                Text("首选已和大众顺序不同的输入。移到标签上点 × 撤销：这个词恢复大众顺序，以后的学习也不再改变它。").font(.caption).foregroundStyle(.secondary)
            }
            Section {
                if status.records.isEmpty {
                    Text("还没有记录。").font(.caption).foregroundStyle(.secondary)
                } else {
                    TagFlow(spacing: 6) {
                        ForEach(status.records) { record in
                            Tag(id: "r:\(record.id)", label: record.keys.isEmpty ? record.text : "\(record.keys) \(record.text)", detail: nil, struck: false, status: status) {
                                Learning.deleteEntries(times: [record.id]); status.refresh()
                            }
                        }
                    }
                }
            } header: {
                Text("学习材料").font(.headline)
            } footer: {
                Text("最近 200 条上屏记录（按键 文字）。移到标签上点 × 删除；删除的记录下次学习不再使用，每次学习都从出厂模型重新学当前的记录。").font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}


/// One learned item as a tag; the × to remove it appears when the pointer is over it.
struct Tag: View {
    let id: String
    let label: String
    let detail: String?
    /// The detail is what the first choice used to be: shown struck through.
    var struck = false
    @ObservedObject var status: LearningStatus
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
            if let detail { Text(detail).strikethrough(struck).foregroundStyle(.secondary) }
            // The × always has its place at the trailing end, so nothing moves when it shows.
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
                .opacity(status.hovered == id ? 1 : 0.25)
                .frame(width: 16)
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Capsule().fill(status.hovered == id ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)))
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
            ForEach(ShortcutFunction.allCases) { function in
                HStack(alignment: .center, spacing: 6) {
                    Text(function.title).frame(width: 96, alignment: .leading)
                    ForEach(store.bindings[function] ?? []) { binding in
                        HStack(spacing: 3) {
                            Text(binding.label)
                            Button { store.remove(binding, from: function) } label: { Image(systemName: "xmark").font(.caption2) }
                                .buttonStyle(.plain).foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(RoundedRectangle(cornerRadius: 5).fill(Color.secondary.opacity(0.15)))
                    }
                    Spacer()
                    Button(recorder.recording == function ? "请按键…（Esc 取消）" : "添加…") {
                        if recorder.recording == function { recorder.stop() } else { recorder.start(function) }
                    }
                }
            }
            HStack { Spacer(); Button("恢复默认") { store.resetDefaults() } }
        } header: {
            Text("快捷键").font(.headline)
        } footer: {
            Text("固定的：空格选中高亮候选，数字选词，回车把拼音原样上屏，Esc 取消；Tab、Shift-Tab、←→ 在候选间移动，- 和 =、↑↓、PageUp 和 PageDown 翻页；符号出现后按 2 或 Tab 换成英文符号。")
                .font(.caption).foregroundStyle(.secondary)
        }
        .onDisappear { recorder.stop() }
    }
}
