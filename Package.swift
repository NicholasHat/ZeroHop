// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ZeroHop",
    platforms: [.macOS("15.0")],
    targets: [
        .target(name: "HarnessCore"),
        .target(name: "HarnessM0", dependencies: ["HarnessCore"]),
        .target(name: "HarnessM1", dependencies: ["HarnessCore"]),
        .executableTarget(
            name: "zerohop",
            dependencies: ["HarnessCore", "HarnessM0", "HarnessM1"]
        ),
    ]
)
