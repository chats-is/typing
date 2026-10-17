import Cocoa
import Engine
import SwiftUI

/// The settings window: the standard macOS settings shell. A split view whose sidebar item gets the
/// system's translucent sidebar material, a unified toolbar so the title sits over the content column,
/// and the selected page's name as the window title. The pages themselves are SwiftUI forms.
enum SettingsWindow {
    private static var controller: SettingsWindowController?

    static func show(page: SettingsPage? = nil) {
        if controller == nil { controller = SettingsWindowController() }
        if let page { controller?.select(page) }
        LearningStatus.shared.refresh()
        // A background-only app (LSBackgroundOnly, as input methods must be) cannot come forward until it
        // takes the accessory activation policy; without this the settings window opens behind.
        NSApp.setActivationPolicy(.accessory)
        controller?.showWindow(nil)
        controller?.window?.orderFrontRegardless()
        NSRunningApplication.current.activate(options: [.activateIgnoringOtherApps, .activateAllWindows])
        NSApp.activate(ignoringOtherApps: true)
        controller?.window?.makeKeyAndOrderFront(nil)
    }
}

final class SettingsWindowController: NSWindowController, NSWindowDelegate, NSToolbarDelegate {
    private let split = SettingsSplitViewController()

    init() {
        let window = NSWindow(contentViewController: split)
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none
        // No title bar material: the content column is one flat surface, as in System Settings.
        window.titlebarAppearsTransparent = true
        window.backgroundColor = .windowBackgroundColor
        window.setContentSize(NSSize(width: 740, height: 611))
        window.minSize = NSSize(width: 700, height: 520)
        window.isReleasedWhenClosed = false
        super.init(window: window)
        window.delegate = self
        let toolbar = NSToolbar(identifier: "SettingsToolbar")
        toolbar.delegate = self
        toolbar.displayMode = .iconOnly
        toolbar.allowsUserCustomization = false
        window.toolbar = toolbar
        window.setFrameAutosaveName("SettingsWindow")
        window.center()
        split.onSelect = { [weak self] page in self?.window?.title = page.title }
        split.select(.general)
    }

    required init?(coder: NSCoder) { fatalError() }

    func select(_ page: SettingsPage) { split.select(page) }

    /// Back to a pure background process once the window closes, so the input method never
    /// competes with the app being typed in for the foreground.
    func windowWillClose(_ notification: Notification) { NSApp.setActivationPolicy(.prohibited) }

    // The toolbar only carries the sidebar toggle and the separator that tracks the sidebar edge,
    // which is what moves the title over the content column.
    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.toggleSidebar, .sidebarTrackingSeparator, .flexibleSpace]
    }
    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        toolbarDefaultItemIdentifiers(toolbar)
    }
    // Required by the protocol; the system builds its own standard items, so nothing to supply.
    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier, willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? { nil }
}

/// Sidebar on the left, the selected page on the right.
final class SettingsSplitViewController: NSSplitViewController {
    var onSelect: ((SettingsPage) -> Void)?
    private let sidebar = SettingsSidebarController()
    private let content = NSViewController()
    private var page: NSViewController?

    override func viewDidLoad() {
        super.viewDidLoad()
        let background = OpaqueBackgroundView()
        content.view = background
        let side = NSSplitViewItem(sidebarWithViewController: sidebar)
        side.minimumThickness = 200
        side.maximumThickness = 300
        side.preferredThicknessFraction = 0.31
        side.canCollapse = false
        addSplitViewItem(side)
        addSplitViewItem(NSSplitViewItem(viewController: content))
        sidebar.onSelect = { [weak self] page in self?.select(page) }
    }

    func select(_ selected: SettingsPage) {
        loadViewIfNeeded()
        page?.view.removeFromSuperview()
        page?.removeFromParent()
        let host: NSViewController
        switch selected {
        case .general: host = NSHostingController(rootView: GeneralSettingsView())
        case .learning: host = NSHostingController(rootView: LearningSettingsView())
        case .about: host = NSHostingController(rootView: AboutSettingsView())
        }
        content.addChild(host)
        host.view.translatesAutoresizingMaskIntoConstraints = false
        content.view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: content.view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: content.view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: content.view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: content.view.bottomAnchor),
        ])
        page = host
        sidebar.highlight(selected)
        onSelect?(selected)
    }
}

/// A flat, opaque window-coloured surface. Being opaque keeps SwiftUI from treating what sits on it as
/// vibrant (which turns the grouped form's white cards into translucent grey ones).
final class OpaqueBackgroundView: NSView {
    override var isOpaque: Bool { true }
    override var allowsVibrancy: Bool { false }
    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()
    }
}

/// The sidebar: a source list with one row per page.
final class SettingsSidebarController: NSViewController, NSTableViewDataSource, NSTableViewDelegate {
    var onSelect: ((SettingsPage) -> Void)?
    private let table = NSTableView()
    private let pages = SettingsPage.allCases

    override func loadView() {
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.documentView = table
        table.style = .sourceList
        table.headerView = nil
        table.backgroundColor = .clear
        table.rowHeight = 34
        table.selectionHighlightStyle = .sourceList
        table.allowsEmptySelection = false
        table.addTableColumn(NSTableColumn(identifier: NSUserInterfaceItemIdentifier("page")))
        table.dataSource = self
        table.delegate = self
        view = scroll
    }

    func highlight(_ page: SettingsPage) {
        guard let row = pages.firstIndex(of: page), table.selectedRow != row else { return }
        table.selectRowIndexes(IndexSet(integer: row), byExtendingSelection: false)
    }

    func numberOfRows(in tableView: NSTableView) -> Int { pages.count }

    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let page = pages[row]
        let cell = NSTableCellView()
        // The icon as System Settings draws it: a white symbol on a coloured rounded tile.
        let tile = NSView()
        tile.wantsLayer = true
        tile.layer?.backgroundColor = page.tint.cgColor
        tile.layer?.cornerRadius = 6
        let icon = NSImageView(image: NSImage(systemSymbolName: page.symbol, accessibilityDescription: page.title) ?? NSImage())
        icon.symbolConfiguration = .init(pointSize: 13, weight: .medium)
        icon.contentTintColor = .white
        let text = NSTextField(labelWithString: page.title)
        text.font = .systemFont(ofSize: 13)
        text.lineBreakMode = .byTruncatingTail
        cell.imageView = icon
        cell.textField = text
        for subview in [tile, icon, text] { subview.translatesAutoresizingMaskIntoConstraints = false }
        cell.addSubview(tile)
        cell.addSubview(icon)
        cell.addSubview(text)
        NSLayoutConstraint.activate([
            tile.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 4),
            tile.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
            tile.widthAnchor.constraint(equalToConstant: 24),
            tile.heightAnchor.constraint(equalToConstant: 24),
            icon.centerXAnchor.constraint(equalTo: tile.centerXAnchor),
            icon.centerYAnchor.constraint(equalTo: tile.centerYAnchor),
            text.leadingAnchor.constraint(equalTo: tile.trailingAnchor, constant: 8),
            text.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: -6),
            text.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
        ])
        return cell
    }

    func tableViewSelectionDidChange(_ notification: Notification) {
        guard table.selectedRow >= 0 else { return }
        onSelect?(pages[table.selectedRow])
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
    case general, learning, about
    var id: String { rawValue }
    var title: String {
        switch self {
        case .general: return "通用"
        case .learning: return "学习"
        case .about: return "关于"
        }
    }
    var symbol: String {
        switch self {
        case .general: return "keyboard"
        case .learning: return "brain.head.profile"
        case .about: return "info.circle"
        }
    }
    /// The tile colour behind the symbol in the sidebar, as System Settings colours its icons.
    var tint: NSColor {
        switch self {
        case .general: return .systemBlue
        case .learning: return .systemPurple
        case .about: return .systemGray
        }
    }
}

/// The look of System Settings' panes: a page of groups, each a white card with rows separated by
/// hairlines, a bold title above and a caption below. (SwiftUI's grouped Form draws translucent grey
/// cards in this window, so the cards are drawn here.)
struct SettingsGroup<Content: View>: View {
    let title: String
    var footer: String? = nil
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.system(size: 15, weight: .semibold)).padding(.horizontal, 4)
            VStack(alignment: .leading, spacing: 0) { content }
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.06)))
            if let footer {
                Text(footer).font(.caption).foregroundStyle(.secondary).padding(.horizontal, 4)
            }
        }
    }
}

/// One row inside a group: standard inset, a hairline above it unless it is the first.
struct SettingsRow<Content: View>: View {
    var first = false
    @ViewBuilder let content: Content
    var body: some View {
        VStack(spacing: 0) {
            if !first { Divider().padding(.leading, 16) }
            HStack(spacing: 8) { content }
                .padding(.horizontal, 16).padding(.vertical, 10)
                .frame(minHeight: 44)
        }
    }
}

/// A page: its groups stacked in a scroll view with System Settings' margins.
struct SettingsScroll<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) { content }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20).padding(.top, 8).padding(.bottom, 24)
        }
    }
}

/// Basic settings: keys. More (candidates per page, font size) belong here as they come.
struct GeneralSettingsView: View {
    var body: some View {
        SettingsScroll {
            ShortcutsSection()
        }
    }
}

/// About: what this is, which version, what it does with your text, where the code lives.
struct AboutSettingsView: View {
    private var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "—"
        let build = info?["CFBundleVersion"] as? String ?? "—"
        return "\(short)（\(build)）"
    }
    private static let learningFolder = "~/Library/Containers/is.chats.inputmethod.Typing"

    var body: some View {
        SettingsScroll {
            SettingsGroup(title: "智能输入") {
                SettingsRow(first: true) {
                    Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 48, height: 48)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("智能输入").font(.system(size: 15, weight: .semibold))
                        Text("macOS 拼音输入法").font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("版本 \(version)").font(.callout).foregroundStyle(.secondary)
                }
                SettingsRow {
                    Text("词库决定能打出什么，语言模型按上下文给候选排序；全拼、简拼、带错的拼法都能打，整句一次上屏。")
                        .font(.callout).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }
            SettingsGroup(title: "你的数据", footer: "删除输入法后，这个文件夹不会自动删除；不需要时可以一并删除。") {
                SettingsRow(first: true) {
                    Text("不联网").frame(width: 96, alignment: .leading)
                    Text("输入法运行在沙盒里，没有网络权限，你输入的任何内容都不会离开这台电脑。").font(.callout).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
                SettingsRow {
                    Text("学习数据").frame(width: 96, alignment: .leading)
                    Text("学习记录和学到的参数只保存在 \(Self.learningFolder)，可以在“学习”页逐条删除或整体重置。").font(.callout).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }
            SettingsGroup(title: "开源") {
                SettingsRow(first: true) {
                    Text("源代码、词库和模型的构建方法都在 GitHub 上。").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("打开项目主页") {
                        if let url = URL(string: "https://github.com/chats-is/typing") { NSWorkspace.shared.open(url) }
                    }
                }
            }
        }
    }
}

/// Learning: the switch and status, what learning changed, what it learned from.
struct LearningSettingsView: View {
    @AppStorage("LearningEnabled") private var learningEnabled = true
    @ObservedObject private var status = LearningStatus.shared

    var body: some View {
        SettingsScroll {
            SettingsGroup(title: "学习", footer: "学习只在这台电脑上，用你确认过的文字微调语言模型的最上两层；每次学习后都用你最近的输入和一组大众句子检验，变差就回滚。关闭后立即恢复出厂模型；重置会删除学习记录和学到的参数。") {
                SettingsRow(first: true) {
                    Toggle("学习我的输入习惯", isOn: $learningEnabled)
                        .toggleStyle(.switch)
                        .onChange(of: learningEnabled) { _, _ in Learning.enabledChanged(); status.refresh() }
                }
                SettingsRow {
                    Text(status.text).font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("重置学习") { Learning.reset(); status.refresh() }
                }
            }
            SettingsGroup(title: "学到的变化", footer: "首选已和大众顺序不同的输入。移到标签上点 × 撤销：这个词恢复大众顺序，以后的学习也不再改变它。") {
                SettingsRow(first: true) {
                    if status.loading {
                        Text("正在对照…").font(.callout).foregroundStyle(.secondary)
                    } else if status.changes.isEmpty {
                        Text("还没有因学习而改变的首选。").font(.callout).foregroundStyle(.secondary)
                    } else {
                        TagFlow(spacing: 6) {
                            ForEach(status.changes) { change in
                                Tag(id: "c:" + change.id, label: "\(change.keys) \(change.learned)", detail: change.general, struck: true, status: status) {
                                    Learning.undo(word: change.learned); status.refresh()
                                }
                            }
                        }
                    }
                    Spacer(minLength: 0)
                }
            }
            SettingsGroup(title: "学习材料", footer: "最近 200 条上屏记录（按键 文字）。移到标签上点 × 删除；删除的记录下次学习不再使用，每次学习都从出厂模型重新学当前的记录。") {
                SettingsRow(first: true) {
                    if status.records.isEmpty {
                        Text("还没有记录。").font(.callout).foregroundStyle(.secondary)
                    } else {
                        TagFlow(spacing: 6) {
                            ForEach(status.records) { record in
                                Tag(id: "r:\(record.id)", label: record.keys.isEmpty ? record.text : "\(record.keys) \(record.text)", detail: nil, struck: false, status: status) {
                                    Learning.deleteEntries(times: [record.id]); status.refresh()
                                }
                            }
                        }
                    }
                    Spacer(minLength: 0)
                }
            }
        }
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
        SettingsGroup(title: "快捷键", footer: "固定的：空格选中高亮候选，数字选词，回车把拼音原样上屏，Esc 取消；Tab、Shift-Tab、←→ 在候选间移动，- 和 =、↑↓、PageUp 和 PageDown 翻页；符号出现后按 2 或 Tab 换成英文符号。") {
            ForEach(Array(ShortcutFunction.allCases.enumerated()), id: \.element) { index, function in
                SettingsRow(first: index == 0) {
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
            SettingsRow {
                Spacer()
                Button("恢复默认") { store.resetDefaults() }
            }
        }
        .onDisappear { recorder.stop() }
    }
}
