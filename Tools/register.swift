import Carbon
import Foundation

// Usage: register <path to Typing.app>
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
if sources.isEmpty {
    // Since macOS 27 a third-party input method is a "legacy plug-in": the system lists it only after
    // the person turns it on under 系统设置 › 通用 › 登录项与扩展 › 扩展 › 输入方式. That approval is kept
    // in the protected preference com.apple.inputsources (AppleEnabledThirdPartyInputSources), which
    // only System Settings may write, so an installer can at most open the pane.
    let approved = (CFPreferencesCopyAppValue("AppleEnabledThirdPartyInputSources" as CFString, "com.apple.inputsources" as CFString) as? [[String: Any]] ?? [])
        .contains { ($0["Bundle ID"] as? String) == bundleID }
    if !approved {
        print("已安装。还要在 系统设置 › 通用 › 登录项与扩展 › 扩展 › 输入方式 里打开“智能输入”，然后到 键盘 › 输入法 添加。")
        let open = Process()
        open.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        open.arguments = ["x-apple.systempreferences:com.apple.LoginItems-Settings.extension"]
        try? open.run()
        exit(3)
    }
    print("registered, but the input source is not listed yet; log out and back in, then add it in System Settings")
    exit(1)
}
for source in sources {
    let enableStatus = TISEnableInputSource(source)
    print(enableStatus == noErr ? "enabled \(bundleID)" : "TISEnableInputSource failed: \(enableStatus)")
}
