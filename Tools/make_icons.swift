import Cocoa

// Usage: make_icons <Resources dir>
// Draws every icon the input method shows. The mark is the input method's own candidate bar
// itself: a capsule holding three candidates, the first one selected, and a
// page arrow. No text, no cursor.
//   - menu.pdf / icon.pdf: the menu bar template icon (22x16 pt, alpha only): the capsule outline
//     with three dots.
//   - palette.png / palette@2x.png: the colour icon in the input-source switcher and Keyboard settings.
//   - AppIcon.iconset: the app icon (iconutil turns it into AppIcon.icns).
guard CommandLine.arguments.count == 2 else { exit(2) }
let out = URL(fileURLWithPath: CommandLine.arguments[1])

// 1. Menu bar template icon (22x16 pt PDF), drawn as vectors: the system tints the alpha.
func menuIcon(to url: URL) {
    var box = CGRect(x: 0, y: 0, width: 22, height: 16)
    guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { exit(1) }
    ctx.beginPDFPage(nil)
    ctx.setStrokeColor(CGColor(gray: 0, alpha: 1))
    ctx.setFillColor(CGColor(gray: 0, alpha: 1))
    // Capsule outline (y is flipped: PDF origin is bottom-left).
    let capsule = CGRect(x: 1.15, y: 16 - 12.35, width: 19.7, height: 8.7)
    ctx.setLineWidth(1.4)
    ctx.addPath(CGPath(roundedRect: capsule, cornerWidth: capsule.height / 2, cornerHeight: capsule.height / 2, transform: nil))
    ctx.strokePath()
    // Three candidates: the first selected (bigger), two more after it.
    for (cx, r) in [(CGFloat(6.0), CGFloat(2.3)), (11.5, 1.45), (16.0, 1.45)] {
        ctx.fillEllipse(in: CGRect(x: cx - r, y: 8 - r, width: 2 * r, height: 2 * r))
    }
    ctx.endPDFPage()
    ctx.closePDF()
}

// 2./3. Colour icon: a green rounded square, a white capsule with the first candidate selected in
//       green, two paler ones after it, and a page arrow. Geometry in a 192-unit square.
func colourIcon(pixels: Int, inset: CGFloat) -> CGImage {
    let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let side = CGFloat(pixels)
    let square = CGRect(x: 0, y: 0, width: side, height: side).insetBy(dx: side * inset, dy: side * inset)
    let u = square.width / 192
    func rect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> CGRect {
        CGRect(x: square.minX + x * u, y: square.maxY - (y + h) * u, width: w * u, height: h * u)
    }
    func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: square.minX + x * u, y: square.maxY - y * u) }
    let green = CGColor(red: 0.118, green: 0.620, blue: 0.416, alpha: 1)      // #1E9E6A
    let greenDeep = CGColor(red: 0.090, green: 0.540, blue: 0.357, alpha: 1)  // bottom of the square
    let pale = CGColor(red: 0.725, green: 0.871, blue: 0.812, alpha: 1)       // #B9DECF
    let radius = square.width * 0.225
    ctx.saveGState()
    ctx.addPath(CGPath(roundedRect: square, cornerWidth: radius, cornerHeight: radius, transform: nil))
    ctx.clip()
    let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [green, greenDeep] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: square.midX, y: square.maxY), end: CGPoint(x: square.midX, y: square.minY), options: [])
    ctx.restoreGState()
    // Capsule.
    ctx.setFillColor(CGColor(gray: 1, alpha: 1))
    let capsule = rect(24, 66, 144, 60)
    ctx.addPath(CGPath(roundedRect: capsule, cornerWidth: capsule.height / 2, cornerHeight: capsule.height / 2, transform: nil))
    ctx.fillPath()
    // Candidates.
    ctx.setFillColor(green)
    ctx.fillEllipse(in: rect(36, 78, 36, 36))
    ctx.setFillColor(pale)
    ctx.fillEllipse(in: rect(80, 84, 24, 24))
    ctx.fillEllipse(in: rect(112, 84, 24, 24))
    // Page arrow.
    ctx.setStrokeColor(pale)
    ctx.setLineWidth(4 * u)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    ctx.move(to: point(146, 90))
    ctx.addLine(to: point(154, 96))
    ctx.addLine(to: point(146, 102))
    ctx.strokePath()
    return ctx.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    let rep = NSBitmapImageRep(cgImage: image)
    try! rep.representation(using: .png, properties: [:])!.write(to: url)
}

menuIcon(to: out.appendingPathComponent("menu.pdf"))
menuIcon(to: out.appendingPathComponent("icon.pdf"))
writePNG(colourIcon(pixels: 64, inset: 0), to: out.appendingPathComponent("palette.png"))
writePNG(colourIcon(pixels: 128, inset: 0), to: out.appendingPathComponent("palette@2x.png"))
// App icon: macOS app icons keep a margin around the shape.
let iconset = out.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for (name, pixels) in [("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("128x128", 128), ("128x128@2x", 256), ("256x256", 256), ("256x256@2x", 512), ("512x512", 512), ("512x512@2x", 1024)] {
    writePNG(colourIcon(pixels: pixels, inset: 0.1), to: iconset.appendingPathComponent("icon_\(name).png"))
}
print("wrote menu.pdf, icon.pdf, palette.png, AppIcon.iconset in \(out.path)")
