# mlx-mel-roformer-swift

The MLXEngine **`audioSeparation`** package over [Mel-Band-RoFormer](https://arxiv.org/abs/2310.01809) — vocal source separation on Apple Silicon.

A thin conformance layer that wraps the standalone inference engine
[`mel-roformer-mlx-swift`](https://github.com/xocialize/mel-roformer-mlx-swift) (product `SwiftRoFormer`)
and exposes it to [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) as a
`ModelPackage`. All model logic — STFT, dual-axis transformer, 8-second-chunk overlap-add — lives
in the core; this package only maps the canonical `AudioSeparationRequest → [Stem: Audio]` contract
onto it and owns the engine lifecycle (`load` / `run` / `unload`).

## Capability

| | |
|---|---|
| Capability | `audioSeparation` |
| Input | mixture `Audio` (.wav, any rate/channels — resampled to 44.1 kHz stereo) |
| Output | `[Stem: Audio]` — `.vocals` and `.instrumental` (.wav, 44.1 kHz stereo) |
| Stems requested empty | returns all the package produces (vocals + instrumental) |

The vocal checkpoints estimate the vocal stem; the instrumental is its complement (`mixture - vocals`).

## Checkpoints

Two parity-tested checkpoints are published on the
[`mlx-community`](https://huggingface.co/mlx-community) HuggingFace org and selected via
`MelRoFormerConfiguration.variant`:

| Variant | HuggingFace repo | Params | Precision |
|---|---|---|---|
| `.kimVocal2` (default) | `mlx-community/mel-roformer-kim-vocal-2-mlx` | 228M | bf16 |
| `.zfturboVocalsV1` | `mlx-community/mel-roformer-zfturbo-vocals-v1-mlx` | 33M | fp16 |

## Usage

```swift
import MLXServeCore
import MLXMelRoFormer

let engine = MLXServeEngine()
try await engine.register(
    MelRoFormerSeparationPackage.registration,
    configuration: MelRoFormerConfiguration(variant: .kimVocal2)
)

let mixture = Audio(format: .wav, data: wavBytes)
let response = try await engine.run(AudioSeparationRequest(audio: mixture)) as! AudioSeparationResponse
let vocals = response[.vocals]            // Audio(.wav, 44.1 kHz stereo)
let instrumental = response[.instrumental]
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-mel-roformer-swift", from: "0.1.0")`, then import `MLXMelRoFormer` (the conformant `audioSeparation` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port code MIT; weights MIT.
