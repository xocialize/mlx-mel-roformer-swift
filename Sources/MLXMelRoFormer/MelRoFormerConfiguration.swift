import Foundation
import MLXToolKit
import SwiftRoFormer

/// A published Mel-Band-RoFormer checkpoint this package can load. Each variant pins its
/// `mlx-community` repo, the matching `RoFormerConfiguration` preset, and a resident-memory
/// footprint so the engine can admit/deny before load.
public enum MelRoFormerVariant: String, Codable, Sendable, CaseIterable {
    /// Kim Vocal 2 — 228M params, higher quality (~12.6 dB SDR). bf16.
    case kimVocal2
    /// ZFTurbo vocals v1 — 33M params, lightweight. fp16.
    case zfturboVocalsV1

    /// The published HuggingFace repo (mlx-community).
    public var repo: String {
        switch self {
        case .kimVocal2: return "mlx-community/mel-roformer-kim-vocal-2-mlx"
        case .zfturboVocalsV1: return "mlx-community/mel-roformer-zfturbo-vocals-v1-mlx"
        }
    }

    /// The core `RoFormerConfiguration` preset for this checkpoint (drives STFT + chunking).
    var roformerConfiguration: RoFormerConfiguration {
        switch self {
        case .kimVocal2: return .kimVocal2
        case .zfturboVocalsV1: return .zfturboVocalsV1
        }
    }

    /// Shipped quantization of the published weights.
    public var quant: Quant {
        switch self {
        case .kimVocal2: return .bf16
        case .zfturboVocalsV1: return .fp16
        }
    }

    /// Conservative resident-memory estimate: weights + STFT/8-second-chunk activations.
    var residentBytes: UInt64 {
        switch self {
        case .kimVocal2: return 1_500_000_000
        case .zfturboVocalsV1: return 600_000_000
        }
    }
}

/// Init-time configuration for `MelRoFormerSeparationPackage` (C9): which published checkpoint
/// to load. Per-request input/stems ride the `AudioSeparationRequest`, not here.
///
/// Conforms to `QuantConfigured` so the memory governor charges the *selected* variant's declared
/// `QuantFootprint` (the two variants are distinguished by quant: kimVocal2 bf16 vs
/// zfturboVocalsV1 fp16) instead of the largest-that-fits heuristic — mirrors how NAFNet's two
/// size variants declare per-variant footprints.
public struct MelRoFormerConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Which published Mel-Band-RoFormer checkpoint to load.
    public var variant: MelRoFormerVariant

    /// Selected variant's shipped quantization (bf16 for kimVocal2, fp16 for zfturboVocalsV1).
    /// Exposed for `QuantConfigured` so the governor matches the right per-variant footprint.
    public var quant: Quant { variant.quant }
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(variant: MelRoFormerVariant = .kimVocal2, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}
