import Testing
import Foundation
import MLXToolKit
@testable import MLXMelRoFormer

/// Offline conformance checks — no weights, no Metal. Live separation + weight-parity is proven
/// in the test app (the model needs the GPU and the published checkpoints).
struct MelRoFormerSeparationTests {

    @Test func manifestIsAudioSeparationAndPermissive() {
        let m = MelRoFormerSeparationPackage.manifest
        #expect(m.capabilities == [.audioSeparation])
        #expect(m.license.weightLicense == .mit)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.provenance.sourceRepo == "mlx-community/mel-roformer-kim-vocal-2-mlx")
    }

    @Test func manifestRequirementsDeclareBothCheckpoints() {
        let r = MelRoFormerSeparationPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.contains { $0.quant == .bf16 && $0.residentBytes == 1_500_000_000 })
        #expect(r.footprints.contains { $0.quant == .fp16 && $0.residentBytes == 600_000_000 })
    }

    @Test func surfaceIsTheCanonicalSeparationDescriptor() {
        let surface = MelRoFormerSeparationPackage.manifest.surfaces.first
        #expect(surface?.capability == .audioSeparation)
        #expect(surface?.parameters.first?.kind == .audio)
    }

    @Test func registrationConstructs() throws {
        let reg = MelRoFormerSeparationPackage.registration
        #expect(reg.manifest.capabilities == [.audioSeparation])
        let pkg = try reg.makePackage(MelRoFormerConfiguration())
        #expect(pkg is MelRoFormerSeparationPackage)
    }

    @Test func variantsMapToPublishedRepos() {
        #expect(MelRoFormerConfiguration().variant == .kimVocal2)
        #expect(MelRoFormerVariant.kimVocal2.repo == "mlx-community/mel-roformer-kim-vocal-2-mlx")
        #expect(MelRoFormerVariant.zfturboVocalsV1.repo == "mlx-community/mel-roformer-zfturbo-vocals-v1-mlx")
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = MelRoFormerConfiguration(variant: .zfturboVocalsV1)
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/should-not-persist")
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(MelRoFormerConfiguration.self, from: data)
        #expect(back.variant == .zfturboVocalsV1)
        #expect(back.modelsRootDirectory == nil)
    }

    @Test func wavHeaderIsValidStereo() {
        // 100 stereo frames -> 200 interleaved samples -> 400 data bytes.
        let interleaved = [Float](repeating: 0, count: 200)
        let wav = MelRoFormerSeparationPackage.encodeWAV16(interleaved: interleaved, channels: 2, sampleRate: 44_100)
        #expect(wav.count == 44 + 400)
        #expect(wav.prefix(4) == Data("RIFF".utf8))
        #expect(wav[8..<12] == Data("WAVE".utf8))
        #expect(wav[36..<40] == Data("data".utf8))
        #expect(wav[22] == 2)  // num channels (LE u16 low byte)
    }
}
