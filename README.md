# ClipSwift 📎

Implementation of the CLIP (Contrastive Language-Image Pre-training) model using the [mlx-swift](https://swiftpackageindex.com/ml-explore/mlx-swift) library.
It's based on [mlx_clip](https://github.com/harperreed/mlx_clip) and [mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/clip).

### Download the model

Before you start, make sure you have the model downloaded

```bash
swift run clipswift-cli download-model
```

### Build

```bash
xcodebuild build \
    -configuration Release \
    -scheme clipswift-cli \
    -destination generic/platform=macOS \
    -derivedDataPath .build/.xcodebuild/ \
    -clonedSourcePackagesDirPath .build/
```

### Run

Encode text

```bash
.build/.xcodebuild/Build/Products/Release/clipswift-cli clip \
    --text-to-encode "a photo of a dog"
```

Encode image

```bash
.build/.xcodebuild/Build/Products/Release/clipswift-cli clip \
    --image-file-to-encode "Sources/ClipSwiftTests/Resources/data/dog.jpeg"
```

Image-text similarity

```bash
.build/.xcodebuild/Build/Products/Release/clipswift-cli clip \
    --text-to-encode "a photo of a dog" \
    --image-file-to-encode "Sources/ClipSwiftTests/Resources/data/dog.jpeg"
```

### Attribution

- `dog.jpeg` is a "Happy Dog" by tedmurphy, licensed under CC BY 2.0.
