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
    var quant: Quant {
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
public struct MelRoFormerConfiguration: PackageConfiguration, ModelStorable {
    /// Which published Mel-Band-RoFormer checkpoint to load.
    public var variant: MelRoFormerVariant
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
