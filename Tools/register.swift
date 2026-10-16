import Carbon
import Foundation

// Usage: register <path to Zhiyin.app>
// Registers the installed input method with the system and enables it.
guard CommandLine.arguments.count == 2 else { exit(2) }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let bundleID = Bundle(url: url)?.bundleIdentifier else {
    print("not an app bundle: \(url.path)")
    exit(1)
}

let status = TISRegisterInputSource(url as CFURL)
guard status == noErr else {
    print("TISRegisterInputSource failed: \(status)")
    exit(1)
}

let filter = [kTISPropertyBundleID as String: bundleID] as CFDictionary
let sources = TISCreateInputSourceList(filter, true)?.takeRetainedValue() as? [TISInputSource] ?? []
guard !sources.isEmpty else {
    print("registered, but the input source is not listed yet; log out and back in, then add it in System Settings")
    exit(1)
}
for source in sources {
    let enableStatus = TISEnableInputSource(source)
    print(enableStatus == noErr ? "enabled \(bundleID)" : "TISEnableInputSource failed: \(enableStatus)")
}
