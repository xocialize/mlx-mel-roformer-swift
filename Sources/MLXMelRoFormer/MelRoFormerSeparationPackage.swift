import Foundation
import MLXToolKit
import MLX
import Hub
import SwiftRoFormer

/// An MLXEngine `audioSeparation` package over **Mel-Band-RoFormer** — splits a music mixture
/// into `vocals` + `instrumental` at 44.1 kHz. A thin conformance wrapper over the standalone
/// `SwiftRoFormer` engine (mel-roformer-mlx-swift); all model logic (STFT, dual-axis transformer,
/// 8-second-chunk overlap-add) lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `MelRoFormerConfiguration`, pages
/// weights in with `load()` (downloads the HF snapshot on first run via `fromPretrained`), drives
/// `run(_:)`, and reclaims with `unload()`. Returns canonical `.wav` `Audio` per stem.
///
/// **v1 produces vocals + instrumental.** The Mel-Band-RoFormer vocal checkpoints estimate the
/// vocal stem; the instrumental is its complement (`mixture - vocals`). A request for stems the
/// package does not produce is ignored (the response only contains what it can deliver).
@InferenceActor
public final class MelRoFormerSeparationPackage: ModelPackage {
    public typealias Configuration = MelRoFormerConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Both shipped checkpoints (Kim Vocal 2, ZFTurbo v1) are MIT; the Swift port is MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/mel-roformer-kim-vocal-2-mlx",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 1_500_000_000),  // kimVocal2 (228M)
                    QuantFootprint(quant: .fp16, residentBytes: 600_000_000),    // zfturboVocalsV1 (33M)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                AudioSeparationContract.descriptor(
                    name: "mel-roformer-separate",
                    summary: "Mel-Band-RoFormer vocal source separation (44.1 kHz .wav): splits a mixture into vocals + instrumental."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var separator: RoFormerSeparator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard separator == nil else { return }
        // Download (or reuse the cached) HF snapshot, then load weights. When the engine has set a
        // model-store root, point the Hub download base there (the caller holds security-scoped
        // access) so weights land in the chosen models folder, not the default cache.
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
        let cfg = configuration.variant.roformerConfiguration
        // `fromPretrained` resolves config.json + model.safetensors and throws on a key mismatch
        // (the parity gate: a wrong/incompatible checkpoint cannot load partially and emit garbage).
        let model = try await MelRoFormer.fromPretrained(configuration.variant.repo,
                                                         configuration: cfg, hub: hub)
        separator = RoFormerSeparator(model: model, configuration: cfg)
    }

    public func unload() async {
        separator = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let separator else { throw PackageError.notLoaded }
        guard request.capability == .audioSeparation,
              let req = request as? AudioSeparationRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Decode the mixture to [1, 2, N] @ 44.1 kHz stereo. AudioIO handles arbitrary input rate /
        // channel count via AVFoundation resampling, so the contract's `Audio` need not be 44.1k.
        let mixture = try Self.decodeMixture(req.audio)
        let vocals = try await separator.separate(samples: mixture)

        // Empty request => every stem this package produces (vocals + instrumental).
        let wantVocals = req.stems.isEmpty || req.stems.contains(.vocals)
        let wantInstrumental = req.stems.isEmpty || req.stems.contains(.instrumental)

        var stems: [Stem: Audio] = [:]
        if wantVocals {
            stems[.vocals] = Self.encodeStem(vocals)
        }
        if wantInstrumental {
            stems[.instrumental] = Self.encodeStem(mixture - vocals)
        }
        return AudioSeparationResponse(stems: stems)
    }

    // MARK: - Audio I/O

    /// Decode a canonical `Audio` (.wav) to a `[1, 2, N]` 44.1 kHz stereo MLXArray, reusing
    /// SwiftRoFormer's `AudioIO` (AVFoundation-backed load + resample). `AudioIO` reads from a
    /// URL, so the bytes round-trip through a temp file.
    nonisolated static func decodeMixture(_ audio: Audio) throws -> MLXArray {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let (samples, _) = try AudioIO().loadAudio(from: tmp)
        return samples
    }

    /// Wrap a `[1, 2, N]` stem MLXArray as a canonical 16-bit PCM stereo WAV `Audio`.
    nonisolated static func encodeStem(_ audio: MLXArray) -> Audio {
        let interleaved = interleaveStereo(audio)
        let wav = encodeWAV16(interleaved: interleaved, channels: 2, sampleRate: 44_100)
        return Audio(format: .wav, data: wav, sampleRate: 44_100, channels: 2)
    }

    /// `[1, 2, N]` (or `[2, N]`) → interleaved `[L0, R0, L1, R1, …]` float samples.
    nonisolated static func interleaveStereo(_ audio: MLXArray) -> [Float] {
        let n = audio.shape[audio.ndim - 1]
        let chans = audio.reshaped([2, n])
        let left = chans[0].asType(.float32).asArray(Float.self)
        let right = chans[1].asType(.float32).asArray(Float.self)
        var out = [Float](repeating: 0, count: n * 2)
        for i in 0..<n {
            out[i * 2] = left[i]
            out[i * 2 + 1] = right[i]
        }
        return out
    }

    /// Encode interleaved float samples as a 16-bit PCM WAV (broadly playable) in memory.
    nonisolated static func encodeWAV16(interleaved samples: [Float], channels: Int, sampleRate: Int) -> Data {
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = (samples.count / channels) * blockAlign

        var data = Data(capacity: 44 + dataSize)
        func ascii(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        ascii("RIFF"); u32(UInt32(36 + dataSize)); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1) // PCM
        u16(UInt16(channels)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(UInt16(bitsPerSample))
        ascii("data"); u32(UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var le = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }
}

extension MelRoFormerSeparationPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MelRoFormerSeparationPackage.self)
    }
}
