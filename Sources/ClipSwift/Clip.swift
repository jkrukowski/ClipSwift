import Foundation
import MLX
import MLXLinalg
import MLXNN

func quickGELU(_ x: MLXArray) -> MLXArray {
    x * MLX.sigmoid(1.702 * x)
}

open class Attention: Module {
    @ModuleInfo public var qProj: Linear
    @ModuleInfo public var kProj: Linear
    @ModuleInfo public var vProj: Linear
    @ModuleInfo public var outProj: Linear
    private let numHeads: Int

    public init(
        dims: Int,
        numHeads: Int,
        queryInputDims: Int? = nil,
        keyInputDims: Int? = nil,
        valueInputDims: Int? = nil,
        valueDims: Int? = nil,
        valueOutputDims: Int? = nil,
        bias: Bool = false
    ) {
        let queryInputDims = queryInputDims ?? dims
        let keyInputDims = keyInputDims ?? dims
        let valueInputDims = valueInputDims ?? keyInputDims
        let valueDims = valueDims ?? dims
        let valueOutputDims = valueOutputDims ?? dims

        self.numHeads = numHeads
        self._qProj = .init(
            wrappedValue: Linear(queryInputDims, dims, bias: bias),
            key: "q_proj"
        )
        self._kProj = .init(
            wrappedValue: Linear(keyInputDims, dims, bias: bias),
            key: "k_proj"
        )
        self._vProj = .init(
            wrappedValue: Linear(valueInputDims, valueDims, bias: bias),
            key: "v_proj"
        )
        self._outProj = .init(
            wrappedValue: Linear(valueDims, valueOutputDims, bias: bias),
            key: "out_proj"
        )
    }

    public func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        var queries = qProj(queries)
        var keys = kProj(keys)
        var values = vProj(values)
        let B = queries.shape[0]
        let L = queries.shape[1]
        let S = keys.shape[1]
        queries = queries.reshaped([B, L, numHeads, -1]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, S, numHeads, -1]).transposed(0, 2, 3, 1)
        values = values.reshaped([B, S, numHeads, -1]).transposed(0, 2, 1, 3)
        let scale = sqrt(1.0 / Float(queries.shape.last!))
        var scores = MLX.matmul(queries * scale, keys)
        if let mask {
            scores = scores + mask.asType(scores.dtype)
        }
        scores = softmax(scores, axis: -1)
        let valuesHat = MLX.matmul(scores, values).transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return outProj(valuesHat)
    }
}

open class MLP: Module {
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear

    public init(config: any Configurable) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(quickGELU(fc1(x)))
    }
}

open class EncoderLayer: Module {
    @ModuleInfo public var selfAttn: Attention
    @ModuleInfo public var layerNorm1: LayerNorm
    @ModuleInfo public var mlp: MLP
    @ModuleInfo public var layerNorm2: LayerNorm

    public init(config: any Configurable) {
        self._selfAttn = .init(
            wrappedValue: Attention(
                dims: config.hiddenSize,
                numHeads: config.numAttentionHeads,
                bias: true
            ),
            key: "self_attn"
        )
        self._layerNorm1 = .init(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps),
            key: "layer_norm1"
        )
        self.mlp = MLP(config: config)
        self._layerNorm2 = .init(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps),
            key: "layer_norm2"
        )
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var y = layerNorm1(x)
        y = selfAttn(queries: y, keys: y, values: y, mask: mask)
        let x = x + y
        y = layerNorm2(x)
        y = mlp(y)
        return x + y
    }
}

open class TextEmbeddings: Module {
    @ModuleInfo public var tokenEmbedding: Embedding
    @ModuleInfo public var positionEmbedding: Embedding

    public init(config: TextConfig) {
        let embedDim = config.hiddenSize
        self._tokenEmbedding = .init(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: embedDim),
            key: "token_embedding"
        )
        self._positionEmbedding = .init(
            wrappedValue: Embedding(
                embeddingCount: config.maxPositionEmbeddings, dimensions: embedDim),
            key: "position_embedding"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var embeddings = tokenEmbedding(x)
        embeddings = embeddings + positionEmbedding.weight[..<x.shape[1]]
        return embeddings
    }
}

open class Encoder: Module {
    public var layers: [EncoderLayer]

    public init(config: any Configurable) {
        self.layers = (0..<config.numHiddenLayers).map { _ in EncoderLayer(config: config) }
    }
}

open class TextModel: Module {
    @ModuleInfo public var embeddings: TextEmbeddings
    @ModuleInfo public var encoder: Encoder
    @ModuleInfo public var finalLayerNorm: LayerNorm

    public init(config: TextConfig) {
        self.embeddings = TextEmbeddings(config: config)
        self.encoder = Encoder(config: config)
        self._finalLayerNorm = .init(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps),
            key: "final_layer_norm"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> TextOutput {
        let B = x.shape[0]
        let N = x.shape[1]
        let eotTokens = MLX.argMax(x, axis: -1)
        var x = embeddings(x)
        let mask = MultiHeadAttention.createAdditiveCausalMask(N, dtype: x.dtype)
        for l in encoder.layers {
            x = l(x, mask: mask)
        }
        let lastHiddenState = finalLayerNorm(x)
        let poolerOutput = lastHiddenState[MLXArray(0..<B), eotTokens]
        return TextOutput(poolerOutput: poolerOutput, lastHiddenState: lastHiddenState)
    }
}

open class VisionEmbeddings: Module {
    @ModuleInfo public var classEmbedding: MLXArray
    @ModuleInfo public var patchEmbedding: Conv2d
    @ModuleInfo public var positionEmbedding: Embedding
    private let embedDim: Int
    private let imageSize: Int
    private let patchSize: Int
    private let numPatches: Int
    private let numPositions: Int

    public init(config: VisionConfig) {
        self._classEmbedding = .init(
            wrappedValue: MLX.zeros([config.hiddenSize]),
            key: "class_embedding"
        )
        self._patchEmbedding = .init(
            wrappedValue: Conv2d(
                inputChannels: config.numChannels ?? 3,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false
            ),
            key: "patch_embedding"
        )
        self.numPatches =
            (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
        self.numPositions = self.numPatches + 1
        self.embedDim = config.hiddenSize
        self.imageSize = config.imageSize
        self.patchSize = config.patchSize
        self._positionEmbedding = .init(
            wrappedValue: Embedding(embeddingCount: numPositions, dimensions: config.hiddenSize),
            key: "position_embedding"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        var patchEmbeddings = patchEmbedding(x)
        patchEmbeddings = MLX.flattened(patchEmbeddings, start: 1, end: 2)
        let embedDim = patchEmbeddings.shape.last!
        let clsEmbeddings = MLX.broadcast(classEmbedding, to: [batchSize, 1, embedDim])
        var embeddings = MLX.concatenated([clsEmbeddings, patchEmbeddings], axis: 1)
        embeddings = embeddings + positionEmbedding.weight
        return embeddings
    }
}

open class VisionModel: Module {
    @ModuleInfo public var embeddings: VisionEmbeddings
    @ModuleInfo public var preLayerNorm: LayerNorm
    @ModuleInfo public var encoder: Encoder
    @ModuleInfo public var postLayerNorm: LayerNorm

    public init(config: VisionConfig) {
        self.embeddings = VisionEmbeddings(config: config)
        self._preLayerNorm = .init(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps),
            key: "pre_layrnorm"
        )
        self.encoder = Encoder(config: config)
        self._postLayerNorm = .init(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps),
            key: "post_layernorm"
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> VisionOutput {
        var x = embeddings(x)
        x = preLayerNorm(x)
        var encoderStates = outputHiddenStates ? [x] : []
        for l in encoder.layers {
            x = l(x)
            if outputHiddenStates {
                encoderStates.append(x)
            }
        }
        let poolerOutput = postLayerNorm(x[0..., 0, 0...])
        return VisionOutput(
            poolerOutput: poolerOutput,
            lastHiddenState: x,
            hiddenStates: outputHiddenStates ? MLX.stacked(encoderStates) : nil
        )
    }
}

open class Model: Module {
    @ModuleInfo public var textModel: TextModel
    @ModuleInfo public var visionModel: VisionModel
    @ModuleInfo public var visualProjection: Linear
    @ModuleInfo public var textProjection: Linear
    @ModuleInfo public var logitScale: MLXArray

    public init(config: ModelConfig) {
        self._textModel = .init(
            wrappedValue: TextModel(config: config.textConfig),
            key: "text_model"
        )
        self._visionModel = .init(
            wrappedValue: VisionModel(config: config.visionConfig),
            key: "vision_model"
        )
        self._visualProjection = .init(
            wrappedValue: Linear(config.visionConfig.hiddenSize, config.projectionDim, bias: false),
            key: "visual_projection"
        )
        self._textProjection = .init(
            wrappedValue: Linear(config.textConfig.hiddenSize, config.projectionDim, bias: false),
            key: "text_projection"
        )
        self._logitScale = .init(
            wrappedValue: MLX.zeros([1]),
            key: "logit_scale"
        )
    }

    public func textFeatures(_ x: MLXArray) -> MLXArray {
        textProjection(textModel(x).poolerOutput)
    }

    public func imageFeatures(_ x: MLXArray) -> MLXArray {
        visualProjection(visionModel(x).poolerOutput)
    }

    public func callAsFunction(
        tokenIds: MLXArray
    ) -> MLXArray {
        let modelOutput = textModel(tokenIds)
        let textEmbeddings = textProjection(modelOutput.poolerOutput)
        return textEmbeddings / MLXLinalg.norm(textEmbeddings, axis: -1, keepDims: true)
    }

    public func callAsFunction(
        pixelValues: MLXArray
    ) -> MLXArray {
        let modelOutput = visionModel(pixelValues)
        let imageEmbeddings = visualProjection(modelOutput.poolerOutput)
        return imageEmbeddings / MLXLinalg.norm(imageEmbeddings, axis: -1, keepDims: true)
    }

    public func similarity(
        tokenIds: MLXArray,
        pixelValues: MLXArray
    ) -> MLXArray {
        let textEmbeddings = self(tokenIds: tokenIds)
        let imageEmbeddings = self(pixelValues: pixelValues)
        return MLX.matmul(textEmbeddings, imageEmbeddings.T)
    }
}

open class Clip {
    public let model: Model
    public let tokenizer: Tokenizer
    public let imageProcessor: ImageProcessor

    public init(
        model: Model,
        tokenizer: Tokenizer,
        imageProcessor: ImageProcessor
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.imageProcessor = imageProcessor
    }

    public convenience init(url: URL) throws {
        let model = try loadModel(at: url)
        let tokenizer = try loadTokenizer(at: url)
        let imageProcessor = try loadImageProcessor(at: url)
        self.init(
            model: model,
            tokenizer: tokenizer,
            imageProcessor: imageProcessor
        )
    }

    public func encode(text: String) -> MLXArray {
        let inputIds = tokenizer.tokenize(text)
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axes: [0])
        return model(tokenIds: inputIdsTensor)
    }

    public func encode(image: Image) throws -> MLXArray {
        let imageArray =
            try imageProcessor
            .preprocess(image: image)
            .expandedDimensions(axes: [0])
        return model(pixelValues: imageArray)
    }

    public func similarity(text: String, image: Image) throws -> MLXArray {
        let inputIds = tokenizer.tokenize(text)
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axes: [0])
        let imageArray =
            try imageProcessor
            .preprocess(image: image)
            .expandedDimensions(axes: [0])
        return model.similarity(tokenIds: inputIdsTensor, pixelValues: imageArray)
    }
}
