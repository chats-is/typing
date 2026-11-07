import Cocoa

/// The language just switched to, shown for a moment under the text cursor in the same bubble the
/// system shows when the input source changes: the menu bar mark in white on the accent colour,
/// its dots filled for Chinese (the mark as the system shows it) and hollow for English. The menu
/// bar icon itself cannot follow the language: the system takes it from the bundle, and only input
/// modes, which this input method has none of, may carry icons of their own.
final class ModeIndicator: NSPanel {
    static let shared = ModeIndicator()

    private let mark = ModeMark()
    private var hideWork: DispatchWorkItem?
    /// Counts shows and hides: a fade that finishes after the mark was shown again leaves it alone.
    private var generation = 0
    /// How long the mark stays before it fades.
    private static let shownFor: TimeInterval = 1.0
    /// The system's bubble: a capsule 30 x 23 pt (27 x 23 pt indicator with 1.5 pt margins on
    /// left and right) around the 22 x 16 pt mark.
    private static let size = NSSize(width: 30, height: 23)

    private init() {
        super.init(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        level = .popUpMenu
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false  // the system's bubble has none
        hidesOnDeactivate = false
        ignoresMouseEvents = true
        contentView = mark
    }

    override var canBecomeKey: Bool { false }

    /// Which language the mark shows (filled dots: Chinese).
    var showsChinese: Bool { mark.chinese }

    /// Shows the mark for `chinese` (or English) 4 pt below `caret`, a screen rect, centred on its
    /// left edge as the system's bubble is (TUINSCursorUIController, TUICursorWindowMargin = 4); it
    /// goes by itself after a moment. Without a caret (the application reports none) nothing is
    /// shown: anywhere else would be the wrong place.
    func show(chinese: Bool, near caret: NSRect?) {
        guard let cursor = caret else {
            hide()
            return
        }
        mark.chinese = chinese
        mark.needsDisplay = true
        let size = Self.size
        var origin = NSPoint(x: cursor.minX - size.width / 2, y: cursor.minY - size.height - 4)
        let screen = NSScreen.screens.first { $0.frame.contains(cursor.origin) } ?? NSScreen.main
        if let visible = screen?.visibleFrame {
            if origin.y < visible.minY { origin.y = cursor.maxY + 4 }
            origin.x = max(visible.minX, min(origin.x, visible.maxX - size.width))
        }
        setFrame(NSRect(origin: origin, size: size), display: true)
        generation += 1
        let current = generation
        // A fade still running would carry on over the new mark: an instant animation replaces it.
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0
            animator().alphaValue = 1
        }
        orderFront(nil)
        hideWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.generation == current else { return }
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.2
                self.animator().alphaValue = 0
            }, completionHandler: { [weak self] in
                // Shown again while it faded: the new mark stays.
                guard let self, self.generation == current else { return }
                self.orderOut(nil)
            })
        }
        hideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.shownFor, execute: work)
    }

    /// Takes the mark away at once (typing goes on, another application came to the front).
    func hide() {
        generation += 1
        hideWork?.cancel()
        hideWork = nil
        orderOut(nil)
    }
}

/// The bubble: a capsule in the accent colour, and the menu bar mark (Tools/make_icons.swift,
/// menuIcon) in white at the size the system's cursor indicator draws it.
private final class ModeMark: NSView {
    var chinese = true

    override func draw(_ dirtyRect: NSRect) {
        NSColor.controlAccentColor.setFill()
        NSBezierPath(roundedRect: bounds, xRadius: bounds.height / 2, yRadius: bounds.height / 2).fill()
        // The system's indicator image is 30 x 23 pt with the 22 x 16 pt mark drawn at (4, 3):
        // centred across, 3 pt from the bottom (not 3.5), so it sits half a point low. The image
        // goes into the 27 x 23 pt indicator view between 1.5 pt side margins.
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.saveGState()
        defer { ctx.restoreGState() }
        ctx.translateBy(x: 1.5, y: 0)
        ctx.scaleBy(x: 27.0 / 30.0, y: 1.0)
        let x0: CGFloat = 4, y0: CGFloat = 3
        NSColor.white.setStroke()
        NSColor.white.setFill()
        let capsule = NSRect(x: x0 + 1.15, y: y0 + 16 - 12.35, width: 19.7, height: 8.7)
        let outline = NSBezierPath(roundedRect: capsule, xRadius: capsule.height / 2, yRadius: capsule.height / 2)
        outline.lineWidth = 1.4
        outline.stroke()
        for (cx, r) in [(CGFloat(6.0), CGFloat(2.3)), (11.5, 1.45), (16.0, 1.45)] {
            let dot = NSRect(x: x0 + cx - r, y: y0 + 8 - r, width: 2 * r, height: 2 * r)
            if chinese {
                NSBezierPath(ovalIn: dot).fill()
            } else {
                // Rings of the same size: the line drawn inside the dot's outline.
                let ring: CGFloat = 0.9
                let circle = NSBezierPath(ovalIn: dot.insetBy(dx: ring / 2, dy: ring / 2))
                circle.lineWidth = ring
                circle.stroke()
            }
        }
    }
}
