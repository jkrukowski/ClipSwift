import AppKit
import ArgumentParser
import ClipSwift
import Foundation

struct ClipCommand: ParsableCommand {
    @Option var modelPath: String =
        "Sources/ClipSwiftTests/Resources/models/jkrukowski/clip-vit-base-patch16/"
    @Option var textToEncode: String?
    @Option var imageFileToEncode: String?
    private var inputType: InputType?

    static let configuration = CommandConfiguration(
        commandName: "clip",
        abstract: "Run CLIP model"
    )

    mutating func validate() throws {
        switch (textToEncode, imageFileToEncode) {
        case (.none, .none):
            throw ValidationError("You must provide either a text or an image to encode.")
        case (.some(let textToEncode), .some(let imageFileToEncode)):
            inputType = .computeSimilarity(textToEncode, URL(fileURLWithPath: imageFileToEncode))
        case (.some(let textToEncode), .none):
            inputType = .encodeText(textToEncode)
        case (.none, .some(let imageFileToEncode)):
            inputType = .encodeImage(URL(fileURLWithPath: imageFileToEncode))
        }
    }

    func run() throws {
        let modelUrl = URL(fileURLWithPath: modelPath)
        let clip = try Clip(url: modelUrl)
        switch inputType {
        case .encodeText(let text):
            print(clip.encode(text: text).asArray(Float.self))
        case .encodeImage(let imageUrl):
            let image = try NSImage.from(contentsOf: imageUrl)
            try print(clip.encode(image: image).asArray(Float.self))
        case .computeSimilarity(let string, let imageUrl):
            let image = try NSImage.from(contentsOf: imageUrl)
            try print(clip.similarity(text: string, image: image).item(Float.self))
        case nil:
            throw ValidationError("You must provide either a text or an image to encode.")
        }
    }
}
