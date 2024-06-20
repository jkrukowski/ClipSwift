import Foundation
import MLX

public struct VisionOutput {
    public var poolerOutput: MLXArray
    public var lastHiddenState: MLXArray
    public var hiddenStates: MLXArray?

    public init(
        poolerOutput: MLXArray,
        lastHiddenState: MLXArray,
        hiddenStates: MLXArray? = nil
    ) {
        self.poolerOutput = poolerOutput
        self.lastHiddenState = lastHiddenState
        self.hiddenStates = hiddenStates
    }
}

public struct TextOutput {
    public var poolerOutput: MLXArray
    public var lastHiddenState: MLXArray

    public init(
        poolerOutput: MLXArray,
        lastHiddenState: MLXArray
    ) {
        self.poolerOutput = poolerOutput
        self.lastHiddenState = lastHiddenState
    }
}

public protocol Configurable {
    var numHiddenLayers: Int { get }
    var hiddenSize: Int { get }
    var intermediateSize: Int { get }
    var numAttentionHeads: Int { get }
    var layerNormEps: Float { get }
}

public struct TextConfig: Codable, Configurable {
    public var numHiddenLayers: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var layerNormEps: Float
    public var maxPositionEmbeddings: Int
    public var vocabSize: Int

    public init(
        numHiddenLayers: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        layerNormEps: Float,
        maxPositionEmbeddings: Int,
        vocabSize: Int
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.layerNormEps = layerNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.vocabSize = vocabSize
    }
}

public struct VisionConfig: Codable, Configurable {
    public var numHiddenLayers: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var layerNormEps: Float
    public var numChannels: Int?
    public var imageSize: Int
    public var patchSize: Int

    public init(
        numHiddenLayers: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        layerNormEps: Float,
        numChannels: Int? = nil,
        imageSize: Int,
        patchSize: Int
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.layerNormEps = layerNormEps
        self.numChannels = numChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
    }
}

public struct ModelConfig: Codable {
    public var textConfig: TextConfig
    public var visionConfig: VisionConfig
    public var projectionDim: Int

    public init(
        textConfig: TextConfig,
        visionConfig: VisionConfig,
        projectionDim: Int
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.projectionDim = projectionDim
    }
}

public struct Pair<T> {
    public var first: T
    public var second: T

    public init(first: T, second: T) {
        self.first = first
        self.second = second
    }
}

extension Pair {
    public init(_ pair: (T, T)) {
        self.init(first: pair.0, second: pair.1)
    }
}

extension Pair: Equatable where T: Equatable {}
extension Pair: Hashable where T: Hashable {}

public enum ClipSwiftError: Error {
    case imageCoversionFailed
}
