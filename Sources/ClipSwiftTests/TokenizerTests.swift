import XCTest

@testable import ClipSwift

final class TokenizerTests: XCTestCase {
    func testTokenizer() throws {
        let bundleUrl = Bundle.module
            .url(forResource: "merges", withExtension: "txt")?
            .deletingLastPathComponent()
        let url = try XCTUnwrap(bundleUrl, "Wrong bundle URL")
        let tokenizer = try loadTokenizer(at: url)

        XCTAssertEqual(tokenizer.tokenize(""), [49406, 49407])
        XCTAssertEqual(tokenizer.tokenize("", appendEos: false), [49406])
        XCTAssertEqual(tokenizer.tokenize("", prependBos: false), [49407])
        XCTAssertEqual(tokenizer.tokenize("", prependBos: false, appendEos: false), [])
        XCTAssertEqual(
            tokenizer.tokenize("a photo of a cat"),
            [49406, 320, 1125, 539, 320, 2368, 49407]
        )
        XCTAssertEqual(
            tokenizer.tokenize("a photo of a cat"),
            tokenizer.tokenize("    a    photo  of  a cat    ")
        )
        XCTAssertEqual(
            tokenizer.tokenize("a photo of a cat"),
            tokenizer.tokenize("A pHotO of a CaT")
        )
    }
}
