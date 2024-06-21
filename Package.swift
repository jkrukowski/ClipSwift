// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "ClipSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "ClipSwift",
            targets: ["ClipSwift"]
        ),
        .executable(
            name: "clipswift-cli",
            targets: ["ClipSwiftCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.12.1"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.8"),
    ],
    targets: [
        .target(
            name: "ClipSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "ClipSwiftTests",
            dependencies: ["ClipSwift"],
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "ClipSwiftCLI",
            dependencies: [
                "ClipSwift",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        )
    ]
)
