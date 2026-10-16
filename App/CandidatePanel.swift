import Cocoa

/// Floating capsule of numbered candidates shown under the text cursor: a glass pill (Liquid
/// Glass on macOS 26 and later, the translucent material before), the highlighted candidate on
/// its own accent-coloured pill.
final class CandidatePanel: NSPanel {
    private let container = NSView()
    private var backing: NSView = NSView()
    /// The cursor rect the panel was last shown next to.
    private(set) var anchor = NSRect.zero

    private static let font = NSFont.systemFont(ofSize: 16)
    private static let indexFont = NSFont.systemFont(ofSize: 12)
    /// Lifts the smaller number so its middle sits at the middle of the candidate's characters
    /// (a Chinese character fills the em box; a digit stands on the baseline up to its cap height).
    private static let indexOffset: CGFloat = (font.ascender + font.descender) / 2 - indexFont.capHeight / 2
    private static let itemHeight: CGFloat = 26
    private static let itemPadding: CGFloat = 8
    private static let itemGap: CGFloat = 0
    private static let edge = NSSize(width: 4, height: 3)

    init() {
        super.init(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        level = .popUpMenu
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        hidesOnDeactivate = false

        container.wantsLayer = true
        backing = Self.makeBacking(for: container)
        contentView = backing
    }

    override var canBecomeKey: Bool { false }

    /// The capsule behind the candidates: Liquid Glass on macOS 26 (needs the macOS 26 SDK, i.e.
    /// Swift 6.2 or newer to compile), the popover material otherwise.
    private static func makeBacking(for container: NSView) -> NSView {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()
            glass.contentView = container
            return glass
        }
        #endif
        let effect = NSVisualEffectView()
        effect.material = .popover
        effect.state = .active
        effect.wantsLayer = true
        effect.layer?.masksToBounds = true
        effect.addSubview(container)
        return effect
    }

    private static func round(_ backing: NSView, radius: CGFloat) {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *), let glass = backing as? NSGlassEffectView {
            glass.cornerRadius = radius
            return
        }
        #endif
        backing.layer?.cornerRadius = radius
    }

    /// Shows `candidates` just below `cursor` (a screen rect), flipping above it at the screen bottom.
    func show(candidates: [String], highlighted: Int, hasMore: Bool, near cursor: NSRect) {
        anchor = cursor
        container.subviews.forEach { $0.removeFromSuperview() }
        var x = Self.edge.width
        for (index, candidate) in candidates.enumerated() {
            let chosen = index == highlighted
            let text = NSMutableAttributedString()
            text.append(NSAttributedString(string: "\(index + 1) ", attributes: [
                .font: Self.indexFont, .baselineOffset: Self.indexOffset,
                .foregroundColor: chosen ? NSColor.white.withAlphaComponent(0.85) : NSColor.secondaryLabelColor,
            ]))
            text.append(NSAttributedString(string: candidate, attributes: [
                .font: Self.font, .foregroundColor: chosen ? NSColor.white : NSColor.labelColor,
            ]))
            let label = NSTextField(labelWithAttributedString: text)
            label.sizeToFit()
            let item = NSView(frame: NSRect(x: x, y: Self.edge.height, width: label.frame.width + Self.itemPadding * 2, height: Self.itemHeight))
            item.wantsLayer = true
            if chosen {
                item.layer?.backgroundColor = NSColor.controlAccentColor.cgColor
                item.layer?.cornerRadius = Self.itemHeight / 2
            }
            label.frame.origin = NSPoint(x: Self.itemPadding, y: (Self.itemHeight - label.frame.height) / 2)
            item.addSubview(label)
            container.addSubview(item)
            x += item.frame.width + Self.itemGap
        }
        if hasMore, let symbol = NSImage(systemSymbolName: "chevron.right", accessibilityDescription: "更多")?
            .withSymbolConfiguration(.init(pointSize: 11, weight: .semibold)) {
            // A symbol, centred on the capsule's middle line (a text chevron sits low, on the x-height).
            let more = NSImageView(image: symbol)
            more.contentTintColor = .tertiaryLabelColor
            more.frame = NSRect(x: x + 2, y: Self.edge.height + (Self.itemHeight - 12) / 2, width: 10, height: 12)
            more.imageScaling = .scaleProportionallyUpOrDown
            container.addSubview(more)
            x += 10 + 6
        }
        let size = NSSize(width: x - Self.itemGap + Self.edge.width, height: Self.itemHeight + Self.edge.height * 2)
        container.frame = NSRect(origin: .zero, size: size)
        backing.frame = container.frame
        Self.round(backing, radius: size.height / 2)

        var origin = NSPoint(x: cursor.minX, y: cursor.minY - size.height - 4)
        let screen = NSScreen.screens.first { $0.frame.contains(cursor.origin) } ?? NSScreen.main
        if let visible = screen?.visibleFrame {
            if origin.y < visible.minY { origin.y = cursor.maxY + 4 }
            origin.x = max(visible.minX, min(origin.x, visible.maxX - size.width))
        }
        setFrame(NSRect(origin: origin, size: size), display: true)
        orderFront(nil)
    }

    func hide() {
        orderOut(nil)
    }
}
