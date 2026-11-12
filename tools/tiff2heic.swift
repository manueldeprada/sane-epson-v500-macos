// tiff2heic in.tif out.heic [quality 0-1]
// Writes a 10-bit HEIC tagged sRGB from a 16-bit sRGB-encoded TIFF.
import CoreImage
import Foundation

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: tiff2heic in.tif out.heic [quality]\n".data(using: .utf8)!)
    exit(2)
}
let quality = args.count > 3 ? Double(args[3]) ?? 0.9 : 0.9
let space = CGColorSpace(name: CGColorSpace.sRGB)!
guard let image = CIImage(contentsOf: URL(fileURLWithPath: args[1]), options: [.colorSpace: space]) else {
    FileHandle.standardError.write("cannot read \(args[1])\n".data(using: .utf8)!)
    exit(1)
}
let context = CIContext()
do {
    try context.writeHEIF10Representation(
        of: image,
        to: URL(fileURLWithPath: args[2]),
        colorSpace: space,
        options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality]
    )
    print("wrote \(args[2]) \(Int(image.extent.width))x\(Int(image.extent.height))")
} catch {
    FileHandle.standardError.write("failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
