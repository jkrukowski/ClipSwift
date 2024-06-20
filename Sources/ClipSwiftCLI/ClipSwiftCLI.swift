import ArgumentParser
import ClipSwift
import Foundation

@main
struct ClipSwiftCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clipswift-cli",
        abstract: "Clip Swift CLI",
        subcommands: [ClipCommand.self, DownloadModelCommand.self]
    )
}
