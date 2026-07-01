# Efficiency Adoption Brief — `mlx-mel-roformer-swift` (Mel-Band-RoFormer, `audioSeparation`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**) + references/memory-harness.md. LIGHT **split + unload-clearCache** adoption. Closest template:
> **NAFNet** (two size variants → per-variant footprint). Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXMelRoFormer` (`MelRoFormerSeparationPackage: ModelPackage`). Capability **`audioSeparation`**
  (vocal/stem separation). Engine pinned `from: "0.3.0"`. Single component: `separator: RoFormerSeparator?`.
- **Footprints today (FLAT residentBytes only, NO transient), TWO size variants:**
  `bf16 1.5 GB` (kimVocal2, 228M) · `fp16 600 MB` (zfturboVocalsV1, 33M).
- `unload()` (~line 69) — verify it `MLX.Memory.clearCache()`s.
- **NOTE:** registry **Val 🟡** (validation incomplete). The efficiency adoption is INDEPENDENT of validation —
  do the four-lever work; **do NOT try to fix/complete the 🟡 validation** (out of scope). If the adoption
  happens to surface something blocking validation, note it and stop-and-report; don't chase it.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 1.5 / 0.6 GB (two variants), no transient | **P1 (headline)** |
| 2. Per-stage evict | ➖ N/A | single separator model | note |
| 3. mmap/lazy | 🟡 verify | confirm lazy load | note |
| 4. BudgetAware | ➖ | variant-chosen | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift.
- **P1 (headline):** split BOTH variants. `residentBytes` = per-variant weights floor; `peakActivationBytes`
  = the per-**chunk** transient (RoFormer processes in chunks → chunk-bounded; measure at the default chunk).
  Use **`FootprintConfigured`** (`residentBytesHint` + `peakActivationBytesHint`) for the two variants if they
  share a quant, or per-`QuantFootprint` if the variants are distinguished by quant (bf16 vs fp16, as declared
  today) — mirror how NAFNet handled signage-vs-width64. Adopt `QuantConfigured`.
- **`unload()` must `MLX.Memory.clearCache()`** after niling `separator`.

## Measurement
Declare `residentBytes` from the measured weight floor (solid) + a **FLAGGED** `peakActivationBytes` from the
smoke per variant (in-app phys ~2.5–2.9× higher). Audio smoke at the default chunk is tractable; flag pending
in-app phys.

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P1 split per variant (chunk size = the activation driver); `unload()` clearCache.
- [ ] Smoke green per variant (valid separation); split recorded; activation flagged. (Val 🟡 left as-is.)
- [ ] Registry: mel-roformer row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split per variant, the chunk-bounded transient, any validation observation (don't fix), drift, effort,
commit SHA. STAY IN SCOPE — four-lever adoption + this brief + registry row only; verify `git show --stat`;
stop-and-report if bigger.
