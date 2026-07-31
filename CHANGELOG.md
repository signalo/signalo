# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please make sure to add your changes to the appropriate categories:

- `Added`: for new functionality
- `Changed`: for changes in existing functionality
- `Deprecated`: for soon-to-be removed functionality
- `Removed`: for removed functionality
- `Fixed`: for fixed bugs
- `Performance`: for performance-relevant changes
- `Security`: for security-relevant changes
- `Other`: for everything else

## [Unreleased]

### Added

- Added turns and radians phase units to `Nco`: `from_turns`, `from_radians`, and `from_turns_per_sample` constructors; `set_turns_per_sample` and `set_radians_per_sample` setters; `turns_per_sample`, `radians_per_sample`, `phase_turns`, and `phase_radians` getters
- Added `Nco` phase-step conversion functions: `phase_step_from_turns_per_sample`, `phase_step_from_radians_per_sample`, `turns_per_sample_from_phase_step`, `radians_per_sample_from_phase_step`, `phase_word_from_turns`, `phase_word_from_radians`, `turns_from_phase_word`, and `radians_from_phase_word`
- Added `Config::from_turns_per_sample` and `Config::from_radians_per_sample` constructors for creating NCO configs from turns-per-sample and radians-per-sample rates
- Added `Nco::phasor_then_step` (behind `complex` feature), returning the current phasor and then advancing one sample — the correct order for derotating a stream
- Added `math::phase::phasor_from_radians` (behind `complex` feature), converting an angle to a phase word and looking its phasor up in one call
- Exposed `math::phase::MAX_ABS_ERROR`, the absolute-error bound of `sin`, `cos`, `sin_cos` and `phasor`, and documented that those are quarter-wave-table approximations rather than exact, so a caller can size a tolerance against the bound instead of copying a literal
- Moved every phase-word, phase-step and frequency conversion from `Nco` to `math::phase` as free functions: `phase_word_from_turns`, `phase_word_from_radians`, `turns_from_phase_word`, `radians_from_phase_word`, `phase_step_from_turns_per_sample`, `phase_step_from_radians_per_sample`, `phase_step_from_frequency`, `turns_per_sample_from_phase_step`, `radians_per_sample_from_phase_step` and `frequency_from_phase_step`. `math::phase` owns the phase-word representation, and none of these involve the oscillator's scalar type, so calling them through `Nco` needed a turbofish for a type parameter they never used. The `Nco` associated functions remain and delegate, so existing callers are unaffected and agree bit for bit

### Changed

- `Nco::phase_step_from_frequency` now delegates to the shared turns-per-sample conversion (behavior unchanged; tested as bit-for-bit equivalent)

### Deprecated

- n/a

### Removed

- n/a

### Fixed

- n/a

### Performance

- n/a

### Security

- n/a

### Other

- n/a

## [0.10.0] - 2026-07-29

### Added

- Added complex sample support to IIR biquad filters: `Biquad<T, K>` and `BiquadCascade<T, CS, SS, K>` accept an explicit coefficient type `K`, enabling `Biquad<Complex32, f32>` (requires `complex` feature)
- Added `KahanSum<T>`, a compensated accumulator for long-running sums that repeatedly add small deltas
- Add `KahanIntegrate<T>`, a compensated cumulative-sum filter backed by `KahanSum<T>`
- Added `fir::design::windowed_sinc` module with per-window submodules for slice-filling low-pass, high-pass, band-pass, and band-stop FIR tap generation
- Added `KaiserSinc::highpass_with_beta`, `KaiserSinc::bandpass_with_beta`, and `KaiserSinc::bandstop_with_beta` constructors (with `_hz` variants) for Kaiser-windowed filters with custom `β`
- Added Kaiser FIR order-design helpers in `filters::fir::design`
- Added Kaiser-windowed low-pass convenience helpers in `windowed_sinc::kaiser`
- Added FIR polyphase prototype packing helpers:
  - Added `packed_len` for computing the rectangular packed polyphase coefficient storage length given a number of phases and taps per phase
  - Added `taps_per_phase_for_prototype_len` for computing taps per phase from a dense prototype length
  - Added `packed_len_for_prototype_len` for computing packed polyphase storage length from a dense prototype length and number of phases
  - Added `pack_prototype_taps` for copying and reordering a dense FIR prototype into phase-major rectangular polyphase coefficient storage
  - Added `pack_prototype_taps_in_place` for in-place reordering of a padded dense prototype buffer into phase-major polyphase storage without a second allocation
- Added [`RingBuffer::fill_with`](crate::storage::RingBuffer::fill_with), a trait method that replaces the entire ring buffer contents with `capacity()` values produced by a closure
- Added [`WithConfig`](crate::traits::WithConfig) implementation for [`ConvolveVec`](crate::filters::fir::convolve::ConvolveVec), so heap-allocated convolution filters can be constructed from config without manually allocating and zero-filling a tap buffer
- Added `polyphase::fractional_delay` module with fractional-delay polyphase FIR filter design helper functions

### Changed

- `Biquad<T>` generalized to `Biquad<T, K = T>` with separate coefficient type; `Config<T>` renamed to `Config<K>` (breaks explicit `Config<f32>` references)
- `BiquadCascade<T, CS, SS>` generalized to `BiquadCascade<T, CS, SS, K = T>` and its type aliases (`BiquadCascadeArray`, `BiquadCascadeVec`, `BiquadCascadeRefMut`) gained a `K` parameter
- Relaxed `State<T>` / `Biquad<T>` default bounds from `Num` to `Zero` for state initialization
- `Kaiser::Config::beta_for_attenuation` now delegates to `filters::fir::design::kaiser_beta`; the boundary at exactly 50 dB now uses the mid-attenuation formula (matching SciPy) instead of the high-attenuation formula
- `Convolve::reset` now fills the delay line in place via [`RingBuffer::fill_with`](crate::storage::RingBuffer::fill_with) instead of reconstructing from config; the `Reset` impl no longer requires `WithConfig`, making it available on `ConvolveVec` and `ConvolveRefMut`
- Moved `Reset` from per-alias impls (e.g. `PolyphaseFirArray`, `PolyphaseDecimatorArray`) onto the generic types, using `fill_with(T::zero)` in place instead of rebuilding via `with_config`.
- Relaxed the `Interpolator` and `RationalResampler` trait bounds from `WithConfig` to `PolyphaseFir: Reset`, delegating through the wrapped FIR.

### Fixed

- Widened the root-raised-cosine singularity guard from `4ε` to `√ε` so the closed-form limit is substituted across the full cancellation region instead of only on an exact hit; rolloff `α = 0.25001` at 4 samples per symbol no longer loses 11 bits of precision.

## [0.9.0] - 2026-07-15

### Added

- Added `BiquadCascade` and `Butterworth` low-pass, high-pass, band-pass, and band-stop biquad filters
- Added `Allpass`, `Comb`, `DcBlocker`, and `FirstOrder` IIR filters
- Added `ZeroCrossing` signal filter
- Added `Envelope` filter with asymmetric attack and release
- Added "libm" feature for `#![no_std]` float support
- Added FIR window functions: `Blackman`, `BlackmanHarris`, `FlatTop`, `Hamming`, `Hann`, `Kaiser`, `Rectangular`, and `Triangular`
- Added FIR filter implementations: differentiator (Fornberg), Lagrange fractional-delay, moving-sum, and windowed-sinc (lowpass, highpass, bandpass, bandstop)
- Added `crate::storage` module with `AsSlice` and `RingBuffer` traits for abstracting over array-based, vec-based, and borrowed storage backends
- Added `AsSlice` impls for `[T; N]`, `Vec<T>`, and `&mut [T]`
- Added `*Array` (stack-allocated) type aliases for all applicable storage-generic filters, sinks, and window functions
- Added `*Vec` (heap-allocated, requires `alloc` feature) type aliases for all applicable storage-generic filters, sinks, and window functions
- Added `*RefMut` (borrowed) type aliases for all applicable storage-generic filters, sinks, and window functions
- Added `from_parts()` constructors to all storage-generic types for constructing with pre-initialized storage
- Added `MultirateFilter` trait for streaming rate-changing filters with independent input and output progress
- Added `PolyphaseFilterBank` coefficient container with array, vec, and borrowed storage aliases
- Added `PolyphaseFir` polyphase FIR executor with shared delay line
- Added `PolyphaseInterpolator` streaming multirate interpolator
- Added `PolyphaseDecimator` streaming multirate decimator
- Added `RationalResampler` streaming multirate resampler with configurable interpolation/decimation ratio
- Added "complex" feature (opt-in `dep:num-complex`) for `Complex<T>` IQ sample support
- Made `Convolve` generic over coefficient type `K` to support complex input with real taps (fixes #166)
- Added `crate::math::phase` module with 32-bit wrapping phase-word trigonometry: `sin`, `cos`, `sin_cos`, and `phasor` (behind `complex` feature)
- Added `fir::design` module with raised-cosine, root-raised-cosine, and GMSK Gaussian pulse-tap generators
- Added `Normalization` tap-normalization enum (`None`, `UnitEnergy`, `UnitPeak`, `PassbandGain`)
- Added `Erf` trait and `erf()` free function (behind `libm` feature)
- Added `math::bessel_i0` for modified Bessel function of the first kind, order 0
- Added `math::safe_normalise_divisor` for validated denominator guarding
- Added `sources::oscillator::nco` module with fixed-point numerically-controlled oscillator (`Nco`) using wrapping `u32` phase-word state, `(T, T)` sin/cos output, and `Complex<T>` phasor (behind `complex` feature)

### Changed

- Made "std" a default-enabled crate feature
- Re-organized `filters` module from flat listing into grouped sub-modules (`iir`, `fir`, `rank`, `classify`, `estimate`, `ops`, `util`, `wavelet`)
- Split `Comb` filter into `FeedbackComb` (`iir::comb`) and `FeedforwardComb` (`fir::comb`)
- Renamed `Cache` wrapper filter to `Last`
- Renamed `UnitSystem` to `Uom` and moved to `util::uom`
- Renamed `observe` module to `estimate`
- `Convolve::normalized()` now normalizes all non-zero coefficient sums (was `sum > 0` only), with `debug_assert!` guards against non-finite sums
- Made all filters, sinks, and window functions generic over their storage backends — old concrete `Type<T, const N: usize>` signatures replaced with generic `Type<T, S>` plus `TypeArray<T, N>` / `TypeVec<T>` aliases
- `BiquadCascade` is now generic over its config and state section storage backends
- `FeedbackComb` switched from manual `[T; D]` delay-line array to generic `RingBuffer`-based delay line
- `Config` type added to `Mean` and `MeanVariance` filters (was bare struct without config)
- `Histogram` now generic over bin storage backend (array, vec, or custom `AsSlice<u32>`)
- Bumped MSRV from "1.68.2" to "1.93.0" (driven by `circular-buffer` 2.0.0)
- `alloc` feature now enables `circular-buffer/alloc`
- `Mean` and `MeanVariance` `Debug` output now includes the `config` field
- Updated dependencies:
  - `circular-buffer` from `1.0.0` to `2.0.0`

## [0.8.0] - 2026-06-10

### Added

- Added `Thresholds<T>` validated type to Schmitt trigger, guaranteeing `low <= high`
- Added zero-window-size assertions (panic on `N = 0`) to `Max`, `Min`, `Median`, `Mean`, `MeanVariance`, `Hampel`, and `Convolve`
- Added `Median::window_iter()` for iterating populated window values
- Added `#[doc(hidden)]` to `StateMut::state_mut()` to discourage direct state manipulation
- Added Complexity documentation sections to all filters, pipes, sinks, and sources
- Added `Chirp`, `Pulse`, `Sawtooth`, `Sine`, `Square`, and `Triangle` oscillator signal sources
- Added `Impulse`, `Step`, and `Noise` signal sources
- Added `PeakHold` and `Rms` signal sinks
- Added `Histogram` and `Percentile` signal sinks
- Added `Correlation` signal sink

### Changed

- Made `StateMut::state_mut()` safe (was `unsafe fn`)
- Changed Schmitt trigger `Config.thresholds` type from `[T; 2]` to `Thresholds<T>`

### Fixed

- Fixed Hampel filter: replaced incorrect min/max-based MAD with full-window median absolute deviation computation
- Fixed `Mean` filter: recompute sum from scratch on each call to prevent floating-point drift
- Fixed `MeanVariance` filter: replaced delegation to two internal `Mean` instances with a direct `sum`/`sum_sq` accumulator for correct variance computation
- Fixed `Median` filter: `max()` returned wrong value after certain insertion patterns
- Fixed `Integrate` filter trait bound (`Sub` → `Add`)
- Fixed `TimeInvariant` supertrait: now extends `ContinuousTime` instead of `DiscreteTime`
- Fixed `ShiftInvariant` supertrait: now extends `DiscreteTime` instead of `ContinuousTime`
- Fixed `Max`/`Min` timestamp recovery on `usize::MAX` overflow

## [0.7.0] - 2026-01-18

### Added

- Added sliding window filters for moving minimum/maximum/bounds.

### Changed

- Merged `signalo_…` crates into the `signalo` umbrella crate.
- Replaced own `CircularBuffer` with `circular_buffer` crate.
- Updated dependencies:
  - `dimensioned` from `0.7` to `0.8`
  - `guts` from `0.1.1` to `0.2.0`
  - `replace_with` from `0.1.5` to `0.1.8`

### Removed

- Removed the `signalo_traits` crate (merging it into the `signalo` umbrella crate).
- Removed the `signalo_filters` crate (merging it into the `signalo` umbrella crate).
- Removed the `signalo_pipes` crate (merging it into the `signalo` umbrella crate).
- Removed the `signalo_sources` crate (merging it into the `signalo` umbrella crate).
- Removed the `signalo_sinks` crate (merging it into the `signalo` umbrella crate).

### Fixed

- Fixed `Mean` filter incorrectly doubling the first input value (#126).
- Fixed variance calculation in `MeanVariance` filter.

### Performance

- Reduced redundant clones in Kalman filter.
- Used `take()` pattern in `MeanVariance` sink to reduce clones.

## [0.6.0] - 2021-06-28

See commit log.

## [0.5.0] - 2018-10-19

See commit log.

## [0.4.0] - 2018-10-07

See commit log.

## [0.3.1] - 2018-09-27

See commit log.

## [0.3.0] - 2018-09-26

See commit log.

## [0.2.0] - 2018-08-28

See commit log.

## [0.1.5] - 2018-08-02

See commit log.

## [0.1.3] - 2018-06-26

See commit log.

## [0.1.2] - 2018-05-18

See commit log.

## [0.1.1] - 2018-05-11

See commit log.

## [0.1.0] - 2018-05-02

Initial release.
