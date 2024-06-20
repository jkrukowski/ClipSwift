import Foundation
import MLX
import MLXNN

func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights: [String: MLXArray] = [:]
    for (k, v) in weights {
        if k.contains("position_ids") {
            continue
        } else if k.contains("patch_embedding.weight") {
            sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
        } else {
            sanitizedWeights[k] = v
        }
    }
    return sanitizedWeights
}

func uniquePairs(from arr: [String]) -> Set<Pair<String>> {
    Set(zip(arr, arr.dropFirst()).map { Pair($0) })
}

func findLowestMergePair(
    in keySet: Set<Pair<String>>,
    using bpeRanks: [Pair<String>: Int]
) -> Pair<String>? {
    var pair: Pair<String>?
    var index: Int?
    for key in keySet {
        guard let mergeIndex = bpeRanks[key] else {
            continue
        }
        if let currentIndex = index {
            if mergeIndex < currentIndex {
                index = mergeIndex
                pair = key
            }
        } else {
            index = mergeIndex
            pair = key
        }
    }
    guard let pair else {
        return nil
    }
    return pair
}

public func loadImageProcessor(at url: URL) throws -> ImageProcessor {
    let configData = try Data(contentsOf: url.appendingPathComponent("preprocessor_config.json"))
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    let config = try jsonDecoder.decode(ImageProcessorConfig.self, from: configData)
    return ImageProcessor(config: config)
}

public func loadModel(at url: URL) throws -> Model {
    let configData = try Data(contentsOf: url.appendingPathComponent("config.json"))
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    let config = try jsonDecoder.decode(ModelConfig.self, from: configData)
    let model = Model(config: config)
    let weightFileUrls = try FileManager.default
        .contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for url in weightFileUrls {
        let fileWeights = try MLX.loadArrays(url: url)
        weights.merge(fileWeights) { $1 }
    }
    weights = sanitize(weights: weights)
    let params = ModuleParameters.unflattened(weights)
    return try model.update(parameters: params, verify: [.noUnusedKeys])
}

public func loadTokenizer(at url: URL) throws -> Tokenizer {
    let mergesData = try String(contentsOf: url.appendingPathComponent("merges.txt"))
    let merges = mergesData.split(separator: "\n").dropFirst()
    var bpeRanks = [Pair<String>: Int]()
    for (index, line) in merges.enumerated() {
        let pair = line.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
        if pair.count != 2 {
            fatalError("Malformed data on line \(line)")
        }
        bpeRanks[Pair(first: pair[0], second: pair[1])] = index
    }
    let vocabData = try JSONSerialization.jsonObject(
        with: Data(contentsOf: url.appendingPathComponent("vocab.json")))
    guard let vocab = vocabData as? [String: Int] else {
        fatalError("Malformed vocab data")
    }
    return Tokenizer(bpeRanks: bpeRanks, vocab: vocab)
}
