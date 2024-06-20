import CoreML
import Foundation
import ImageIO
import MLX

#if os(macOS)
    import AppKit
    public typealias Image = NSImage
#else
    import UIKit
    public typealias Image = UIImage
#endif

extension Image {
    public var cgImage: CGImage? {
        #if os(macOS)
            return cgImage(forProposedRect: nil, context: nil, hints: nil)
        #else
            return cgImage
        #endif
    }

    public var scale: CGFloat {
        #if os(macOS)
            return 1
        #else
            return scale
        #endif
    }

    public func pngData() -> Data? {
        #if os(macOS)
            guard let cgImage else {
                return nil
            }
            let rep = NSBitmapImageRep(cgImage: cgImage)
            return rep.representation(using: .png, properties: [:])
        #else
            return pngData()
        #endif
    }

    public func jpegData(compressionQuality: CGFloat = 1.0) -> Data? {
        #if os(macOS)
            guard let cgImage else {
                return nil
            }
            let rep = NSBitmapImageRep(cgImage: cgImage)
            return rep.representation(
                using: .jpeg, properties: [.compressionFactor: compressionQuality])
        #else
            return jpegData(compressionQuality: compressionQuality)
        #endif
    }

    func centerCrop(to newSize: CGSize) -> Image? {
        guard Int(newSize.width) % 2 == 0, Int(newSize.height) % 2 == 0 else {
            fatalError("Only even crop sizes supported.")
        }
        let originalSize = size
        let top = (originalSize.height - newSize.height) / 2
        let left = (originalSize.width - newSize.width) / 2
        let rect = CGRect(x: left, y: top, width: newSize.width, height: newSize.height)
        guard let cgImage = cgImage?.cropping(to: rect) else {
            return nil
        }
        #if os(macOS)
            return Image(cgImage: cgImage, size: newSize)
        #else
            return Image(cgImage: cgImage)
        #endif
    }

    func resized(to squareSize: Int) -> Image? {
        let width = Int(size.width)
        let height = Int(size.height)
        let short = min(width, height)
        let long = max(width, height)
        if short == squareSize {
            return self
        }
        let newShort = squareSize
        let newLong = Int(Double(squareSize) * Double(long) / Double(short))
        let newWidth = width <= height ? newShort : newLong
        let newHeight = width <= height ? newLong : newShort
        let size = CGSize(width: newWidth, height: newHeight)
        return resized(to: size)
    }

    #if os(macOS)
        func resized(to newSize: CGSize) -> Image? {
            guard
                let bitmapRep = NSBitmapImageRep(
                    bitmapDataPlanes: nil,
                    pixelsWide: Int(newSize.width),
                    pixelsHigh: Int(newSize.height),
                    bitsPerSample: 8,
                    samplesPerPixel: 4,
                    hasAlpha: true,
                    isPlanar: false,
                    colorSpaceName: .calibratedRGB,
                    bytesPerRow: 0,
                    bitsPerPixel: 0
                )
            else {
                return nil
            }
            bitmapRep.size = newSize
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmapRep)
            draw(
                in: NSRect(x: 0, y: 0, width: newSize.width, height: newSize.height),
                from: .zero,
                operation: .copy,
                fraction: 1.0
            )
            NSGraphicsContext.restoreGraphicsState()

            let resizedImage = NSImage(size: newSize)
            resizedImage.addRepresentation(bitmapRep)
            return resizedImage
        }
    #else
        func resized(to size: CGSize) -> Image? {
            UIGraphicsBeginImageContextWithOptions(size, false, scale)
            draw(in: CGRect(origin: CGPoint.zero, size: size))
            let result = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            return result
        }
    #endif
}

public struct ImageProcessorConfig: Codable {
    public var size: Int
    public var cropSize: Int
    public var doCenterCrop: Bool
    public var doNormalize: Bool
    public var doResize: Bool
    public var imageMean: [Float]
    public var imageStd: [Float]
}

open class ImageProcessor {
    private let config: ImageProcessorConfig
    private let imageMean: MLXArray
    private let imageStd: MLXArray

    public init(config: ImageProcessorConfig) {
        self.config = config
        self.imageMean = MLXArray(config.imageMean)
        self.imageStd = MLXArray(config.imageStd)
    }

    public func preprocess(image: Image) throws -> MLXArray {
        var image = image
        if config.doResize {
            image = image.resized(to: config.size)!
        }
        if config.doCenterCrop {
            image = image.centerCrop(to: CGSize(width: config.cropSize, height: config.cropSize))!
        }
        var imageArray = try image.toMLXArray()
        imageArray = rescale(image: imageArray)
        if config.doNormalize {
            imageArray = normalize(image: imageArray, mean: imageMean, std: imageStd)
        }
        return imageArray
    }

    func rescale(image: MLXArray) -> MLXArray {
        image.asType(.float32) * Float(1.0 / 255.0)
    }

    func normalize(image: MLXArray, mean: MLXArray, std: MLXArray) -> MLXArray {
        (image - mean) / std
    }
}

extension Image {
    public func toMLXArray() throws -> MLXArray {
        guard let bytesPerRow = cgImage?.bytesPerRow, let provider = cgImage?.dataProvider else {
            throw ClipSwiftError.imageCoversionFailed
        }
        guard let bytes = CFDataGetBytePtr(provider.data) else {
            throw ClipSwiftError.imageCoversionFailed
        }
        let height = Int(size.height)
        let width = Int(size.width)
        var arr = [UInt8]()
        arr.reserveCapacity(width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let r = bytes[offset + 0]
                let g = bytes[offset + 1]
                let b = bytes[offset + 2]
                arr.append(r)
                arr.append(g)
                arr.append(b)
            }
        }
        let array = MLXArray(arr, [height, width, 3])
        return array
    }
}
