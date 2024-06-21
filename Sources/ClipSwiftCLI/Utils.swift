import ArgumentParser
import Foundation
import ClipSwift

enum InputType: Decodable {
    case encodeText(String)
    case encodeImage(String)
    case computeSimilarity(text: String, imagePath: String)
}
