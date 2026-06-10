// swift-tools-version: 6.2
import PackageDescription

// mlx-mel-roformer-swift — the MLXEngine `audioSeparation` package over Mel-Band-RoFormer.
// A thin conformance layer: it wraps the standalone inference engine mel-roformer-mlx-swift
// (product `SwiftRoFormer`) the same way mlx-voxcpm2-tts-swift wraps mlx-voxcpm-swift and
// mlx-kokoro-tts-swift wraps mlx-audio-swift. The engine contract (MLXToolKit) is a local-path
// dep for in-workspace dev; the SwiftRoFormer core is pinned to a tagged release.
//
// Swift-port naming: `-swift` on the package/repo; module/product stays clean `MLXMelRoFormer`.
let package = Package(
    name: "mlx-mel-roformer-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXMelRoFormer", targets: ["MLXMelRoFormer"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/mel-roformer-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXMelRoFormer",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // Package identity derives from the repo URL's last path component.
                .product(name: "SwiftRoFormer", package: "mel-roformer-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // SwiftRoFormer's `MelRoFormer` / `RoFormerSeparator` (MLX + swift-transformers) aren't
            // Sendable-audited, so awaiting the nonisolated `fromPretrained` back into the
            // `@InferenceActor` trips strict region-isolation ("sending non-Sendable"). The engine
            // serializes all lifecycle on InferenceActor (no real concurrency), so v5 mode keeps
            // that a warning while `@InferenceActor` isolation still holds — same lever as Kokoro/VoxCPM2.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXMelRoFormerTests",
            dependencies: [
                "MLXMelRoFormer",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // Test-only: admissibility / manifest checks through the engine contract.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
