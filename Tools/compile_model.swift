import CoreML
import Foundation

// Usage: compile_model <Model.mlpackage> <Model.mlmodelc>
// Compiles an exported Core ML package into the form the input method loads at runtime.
guard CommandLine.arguments.count == 3 else { exit(2) }
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let destination = URL(fileURLWithPath: CommandLine.arguments[2])

let compiled = try MLModel.compileModel(at: source)
try? FileManager.default.removeItem(at: destination)
try FileManager.default.copyItem(at: compiled, to: destination)
print("compiled \(destination.path)")
