// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Typing",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "Engine", path: "Engine", linkerSettings: [.linkedLibrary("sqlite3")]),
        .executableTarget(name: "Typing", dependencies: ["Engine"], path: "App"),
        .executableTarget(name: "enginecli", dependencies: ["Engine"], path: "Tools/enginecli"),
    ],
    swiftLanguageModes: [.v5]
)
