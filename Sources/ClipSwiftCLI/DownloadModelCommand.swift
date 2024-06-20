import ArgumentParser
import Foundation
import Hub

struct DownloadModelCommand: AsyncParsableCommand {
    @Option var repo: String = "jkrukowski/clip-vit-base-patch16"
    @Option var savePath: String = "Sources/ClipSwiftTests/Resources/"

    static let configuration = CommandConfiguration(
        commandName: "download-model",
        abstract: "Download model from Hugging Face"
    )

    func run() async throws {
        let hubApi = HubApi(downloadBase: URL(fileURLWithPath: savePath))
        let hubRepo = Hub.Repo(id: repo, type: .models)
        try await hubApi.snapshot(from: hubRepo) { progress in
            let fractionCompleted = String(format: "%.2f", progress.fractionCompleted * 100)
            print("Downloading a model from \(repo): \(fractionCompleted)%")
        }
    }
}
