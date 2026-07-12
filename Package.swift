// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ZeroHop",
    platforms: [.macOS("15.0")],
    dependencies: [
        // M3 target-model lane only. The measurement targets (HarnessCore,
        // M0-M2) and the `zerohop` binary stay dependency-free (spec §11:
        // no third-party code on the critical path) — MLX links only into
        // the separate `zerohop-m3` executable.
        // Pinned to 2.x: 3.x/main moved downloading into macro-based
        // integration packages; 2.31.x keeps loadContainer(hub:configuration:).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.31.3")),
    ],
    targets: [
        .target(name: "HarnessCore"),
        .target(name: "HarnessM0", dependencies: ["HarnessCore"]),
        .target(name: "HarnessM1", dependencies: ["HarnessCore"]),
        .target(name: "HarnessM2", dependencies: ["HarnessCore", "HarnessM1"]),
        .executableTarget(
            name: "zerohop",
            dependencies: ["HarnessCore", "HarnessM0", "HarnessM1", "HarnessM2"]
        ),
        .target(
            name: "HarnessM3",
            dependencies: [
                "HarnessCore", "HarnessM1",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "zerohop-m3",
            dependencies: ["HarnessCore", "HarnessM3"]
        ),
    ]
)
