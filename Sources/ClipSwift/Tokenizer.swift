import Foundation

open class Tokenizer {
    private let bpeRanks: [Pair<String>: Int]
    private let vocab: [String: Int]
    private let splitStringPattern: Regex<AnyRegexOutput>
    private let emptyStringPattern: Regex<AnyRegexOutput>
    private let bos: String
    private let bosToken: Int
    private let eos: String
    private let eosToken: Int
    private var cache: [String: [String]]

    public init(
        bpeRanks: [Pair<String>: Int],
        vocab: [String: Int]
    ) {
        self.bpeRanks = bpeRanks
        self.vocab = vocab
        self.splitStringPattern = try! Regex(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        )
        self.emptyStringPattern = try! Regex("\\s+")
        self.bos = "<|startoftext|>"
        self.bosToken = vocab[bos]!
        self.eos = "<|endoftext|>"
        self.eosToken = vocab[eos]!
        self.cache = [:]
    }

    open func tokenize(_ text: String, prependBos: Bool = true, appendEos: Bool = true) -> [Int] {
        let cleanText = text.lowercased().replacing(emptyStringPattern, with: " ")
        let tokens = cleanText.ranges(of: splitStringPattern).map { String(cleanText[$0]) }
        let bpeTokens = tokens.flatMap { bpe($0) }
        var tokenIds = [Int]()
        if prependBos {
            tokenIds.append(bosToken)
        }
        tokenIds.append(contentsOf: bpeTokens.map { vocab[$0]! })
        if appendEos {
            tokenIds.append(eosToken)
        }
        return tokenIds
    }

    private func bpe(_ text: String) -> [String] {
        if let cached = cache[text] {
            return cached
        }
        var unigrams = text.dropLast().map { String($0) } + ["\(text.suffix(1))</w>"]
        var uniqueBigrams = uniquePairs(from: unigrams)
        while !uniqueBigrams.isEmpty {
            guard let lowestMergePair = findLowestMergePair(in: uniqueBigrams, using: bpeRanks)
            else {
                break
            }
            var newUnigrams = [String]()
            var skip = false
            for (first, second) in zip(unigrams, unigrams.dropFirst()) {
                if skip {
                    skip = false
                    continue
                }
                let pair = Pair(first: first, second: second)
                if pair == lowestMergePair {
                    newUnigrams.append(first + second)
                    skip = true
                } else {
                    newUnigrams.append(first)
                }
            }

            if !skip {
                newUnigrams.append(unigrams.last!)
            }

            unigrams = newUnigrams
            uniqueBigrams = uniquePairs(from: unigrams)
        }

        cache[text] = unigrams
        return unigrams
    }
}
