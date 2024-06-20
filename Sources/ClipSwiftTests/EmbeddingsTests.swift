import Foundation
import MLX
import XCTest

@testable import ClipSwift

final class EmbeddingsTests: XCTestCase {
    private var modelUrl: URL!

    override func setUp() async throws {
        try await super.setUp()
        let url = Bundle.module.url(
            forResource: "model",
            withExtension: "safetensors"
        )?.deletingLastPathComponent()
        modelUrl = try XCTUnwrap(url, "Coudn't find model directory")
    }

    func testTextEmdeddings() throws {
        let storedData = try loadEmbeddings(
            fileName: "textEmbdeddings",
            fileExtension: "txt"
        )
        let storedEmbeddings = MLXArray(storedData, [storedData.count])
            .asType(Float.self)
            .expandedDimensions(axes: [0])

        let clip = try Clip(url: modelUrl)
        let textEmbeddings = clip.encode(text: "a photo of a cat")

        XCTAssertTrue(
            MLX.allClose(storedEmbeddings, textEmbeddings, atol: 1e-7).item(),
            "Embeddings don't match"
        )
    }

    func testSimilarity() throws {
        let imageUrl = try XCTUnwrap(Bundle.module.url(forResource: "dog", withExtension: "jpeg"))
        let image = try XCTUnwrap(Image(contentsOf: imageUrl))
        let clip = try Clip(url: modelUrl)

        let similarity1 = try clip.similarity(
            text: "a photo of a dog",
            image: image
        ).item(Float.self)
        let similarity2 = try clip.similarity(
            text: "an astronaut riding a horse",
            image: image
        ).item(Float.self)

        XCTAssertTrue(similarity1 > similarity2)
    }

    private func loadEmbeddings(
        fileName: String,
        fileExtension: String
    ) throws -> [Float] {
        let textEmbdeddingsUrl = Bundle.module.url(
            forResource: fileName,
            withExtension: fileExtension
        )
        let url = try XCTUnwrap(textEmbdeddingsUrl, "Wrong file URL \(fileName).\(fileExtension)")
        let fileContent = try String(contentsOf: url)
        return fileContent.components(separatedBy: ", ").compactMap { Float($0) }
    }
}
