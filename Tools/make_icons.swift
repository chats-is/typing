import Cocoa
import CoreText

// Usage: make_icons <Resources dir>
// Draws every icon the input method shows: the menu bar icon (a template PDF the system tints:
// a rounded square with 智 cut out of it), the palette icon used by the input
// source switcher and Keyboard settings (colour PNG), and the app icon (.iconset → .icns).
guard CommandLine.arguments.count == 2 else { exit(2) }
let out = URL(fileURLWithPath: CommandLine.arguments[1])

func glyphPath(_ text: String, weight: NSFont.Weight, size: CGFloat, rounded: Bool = false) -> CGPath {
    var base = NSFont.systemFont(ofSize: size, weight: weight)
    if rounded, let d = base.fontDescriptor.withDesign(.rounded), let f = NSFont(descriptor: d, size: size) { base = f }
    let font = CTFontCreateForString(base as CTFont, text as CFString, CFRange(location: 0, length: text.utf16.count))
    var chars = Array(text.utf16)
    var glyphs = [CGGlyph](repeating: 0, count: chars.count)
    _ = CTFontGetGlyphsForCharacters(font, &chars, &glyphs, chars.count)
    let path = CGMutablePath()
    var x: CGFloat = 0
    for g in glyphs {
        if let p = CTFontCreatePathForGlyph(font, g, nil) { path.addPath(p, transform: CGAffineTransform(translationX: x, y: 0)) }
        var advance = CGSize.zero
        var glyph = g
        CTFontGetAdvancesForGlyphs(font, .horizontal, &glyph, &advance, 1)
        x += advance.width - size * 0.03
    }
    // Strokes of a CJK glyph overlap; resolving the winding keeps even-odd filling from cutting holes in them.
    return path.normalized(using: .winding)
}

func placed(_ p: CGPath, center: CGPoint, scale: CGFloat = 1) -> CGPath {
    let b = p.boundingBoxOfPath
    let m = CGMutablePath()
    m.addPath(p, transform: CGAffineTransform(translationX: center.x, y: center.y).scaledBy(x: scale, y: scale).translatedBy(x: -b.midX, y: -b.midY))
    return m
}

/// The mark on a canvas `width` pt wide and 16/22 as tall: 智, centred (the user chose 智 alone,
/// 2026-10-11; a medium weight stays legible when cut out of the square at 16 pt).
func mark(width: CGFloat) -> CGPath {
    let unit = width / 22
    // PingFang's heaviest weight is Semibold, lighter than the system's own badges: the strokes are
    // thickened by outlining them and merging the outline with the glyph.
    let glyph = glyphPath("智", weight: .semibold, size: 11 * unit)
    let thick = glyph.union(glyph.copy(strokingWithWidth: 0.25 * unit, lineCap: .round, lineJoin: .round, miterLimit: 1))
    return placed(thick, center: CGPoint(x: 11 * unit, y: 8 * unit))
}

// 1. Menu bar / input list template icon (22x16 pt PDF): the rounded badge of the system's own input
//    sources with the system "keyboard" symbol cut out of it (the user chose a keyboard on the badge,
//    2026-10-14). Drawn at 12x as an image with alpha; the template uses only the alpha.
func menuIcon(to url: URL) {
    var box = CGRect(x: 0, y: 0, width: 22, height: 16)
    guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { exit(1) }
    let k: CGFloat = 12
    let w = Int(22 * k), h = Int(16 * k)
    let bitmap = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    bitmap.addPath(CGPath(roundedRect: CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)), cornerWidth: 5 * k, cornerHeight: 5 * k, transform: nil))
    bitmap.setFillColor(CGColor(gray: 0, alpha: 1))
    bitmap.fillPath()
    let config = NSImage.SymbolConfiguration(pointSize: 160, weight: .semibold)
        .applying(NSImage.SymbolConfiguration(paletteColors: [.black]))
    if let symbol = NSImage(systemSymbolName: "keyboard", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
        var rect = CGRect(origin: .zero, size: symbol.size)
        if let cg = symbol.cgImage(forProposedRect: &rect, context: nil, hints: nil) {
            let scale = min(15 * k / rect.width, 10.5 * k / rect.height)
            let target = CGRect(x: (CGFloat(w) - rect.width * scale) / 2, y: (CGFloat(h) - rect.height * scale) / 2, width: rect.width * scale, height: rect.height * scale)
            bitmap.interpolationQuality = .high
            bitmap.setBlendMode(.destinationOut)
            bitmap.draw(cg, in: target)
        }
    }
    ctx.beginPDFPage(nil)
    ctx.interpolationQuality = .high
    ctx.draw(bitmap.makeImage()!, in: box)
    ctx.endPDFPage()
    ctx.closePDF()
}

// 1b. Menu bar icon (22x16 pt template PDF): a filled keyboard exactly the size of the input-source
//     badge (22x16, corner 5): the body is the badge, the keys are cut out in the layout of the system
//     "keyboard" symbol — two rows of six keys, then a key, the space bar, a key (user, 2026-10-14).
func menuOutlineIcon(to url: URL) {
    var box = CGRect(x: 0, y: 0, width: 22, height: 16)
    guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { exit(1) }
    let k: CGFloat = 12
    let w = Int(22 * k), h = Int(16 * k)
    let bitmap = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    bitmap.scaleBy(x: k, y: k)
    bitmap.addPath(CGPath(roundedRect: box, cornerWidth: 5, cornerHeight: 5, transform: nil))
    bitmap.setFillColor(CGColor(gray: 0, alpha: 1))
    bitmap.fillPath()
    bitmap.setBlendMode(.destinationOut)
    let key: CGFloat = 1.7, left: CGFloat = 4.2, right: CGFloat = 17.8
    let pitch = (right - left - key) / 5
    func cut(_ rect: CGRect) {
        bitmap.addPath(CGPath(roundedRect: rect, cornerWidth: 0.55, cornerHeight: 0.55, transform: nil))
        bitmap.fillPath()
    }
    for (row, top) in [CGFloat(4.15), 7.15].enumerated() {
        _ = row
        for col in 0..<6 { cut(CGRect(x: left + CGFloat(col) * pitch, y: 16 - top - key, width: key, height: key)) }
    }
    let bottom: CGFloat = 10.15  // keys centred vertically in the badge
    cut(CGRect(x: left, y: 16 - bottom - key, width: key, height: key))
    cut(CGRect(x: left + pitch, y: 16 - bottom - key, width: 4 * pitch - (pitch - key), height: key))
    cut(CGRect(x: left + 5 * pitch, y: 16 - bottom - key, width: key, height: key))
    ctx.beginPDFPage(nil)
    ctx.interpolationQuality = .high
    ctx.draw(bitmap.makeImage()!, in: box)
    ctx.endPDFPage()
    ctx.closePDF()
}

// 2./3. Colour icon: a rounded square in the system's input-source blue with the mark in white.
func colourIcon(pixels: Int, inset: CGFloat) -> CGImage {
    let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let side = CGFloat(pixels)
    let square = CGRect(x: 0, y: 0, width: side, height: side).insetBy(dx: side * inset, dy: side * inset)
    let radius = square.width * 0.225
    ctx.addPath(CGPath(roundedRect: square, cornerWidth: radius, cornerHeight: radius, transform: nil))
    ctx.clip()
    let colors = [CGColor(red: 0.16, green: 0.47, blue: 0.96, alpha: 1), CGColor(red: 0.07, green: 0.30, blue: 0.78, alpha: 1)] as CFArray
    let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: square.midX, y: square.maxY), end: CGPoint(x: square.midX, y: square.minY), options: [])
    // A keyboard (the system's "keyboard" symbol) in white, centred (the user chose a keyboard for
    // the app icon, 2026-10-14; the menu-bar badge stays 智).
    let config = NSImage.SymbolConfiguration(pointSize: square.width * 0.5, weight: .medium)
        .applying(NSImage.SymbolConfiguration(paletteColors: [.white]))
    if let symbol = NSImage(systemSymbolName: "keyboard", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
        var rect = CGRect(origin: .zero, size: symbol.size)
        if let cg = symbol.cgImage(forProposedRect: &rect, context: nil, hints: nil) {
            let target = CGRect(x: square.midX - rect.width / 2, y: square.midY - rect.height / 2, width: rect.width, height: rect.height)
            ctx.draw(cg, in: target)
        }
    }
    return ctx.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    let rep = NSBitmapImageRep(cgImage: image)
    try! rep.representation(using: .png, properties: [:])!.write(to: url)
}

menuIcon(to: out.appendingPathComponent("icon.pdf"))
menuOutlineIcon(to: out.appendingPathComponent("menu.pdf"))
writePNG(colourIcon(pixels: 64, inset: 0), to: out.appendingPathComponent("palette.png"))
writePNG(colourIcon(pixels: 128, inset: 0), to: out.appendingPathComponent("palette@2x.png"))
// App icon: macOS app icons keep a margin around the shape.
let iconset = out.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for (name, pixels) in [("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("128x128", 128), ("128x128@2x", 256), ("256x256", 256), ("256x256@2x", 512), ("512x512", 512), ("512x512@2x", 1024)] {
    writePNG(colourIcon(pixels: pixels, inset: 0.1), to: iconset.appendingPathComponent("icon_\(name).png"))
}
print("wrote icon.pdf, palette.png, AppIcon.iconset in \(out.path)")
