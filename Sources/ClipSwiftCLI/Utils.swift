import AppKit
import ArgumentParser
import Foundation

enum InputType: Decodable {
    case encodeText(String)
    case encodeImage(URL)
    case computeSimilarity(String, URL)
}

extension NSImage {
    static func from(contentsOf file: URL) throws -> NSImage {
        guard let image = NSImage(contentsOf: file) else {
            throw ValidationError("Invalid image file to encode.")
        }
        return image
    }
}
