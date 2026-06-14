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

- Added `BiquadCascade` and `Butterworth` low-pass, high-pass, band-pass, and band-stop biquad filters
- Added `Allpass`, `Comb`, `DcBlocker`, and `FirstOrder` IIR filters
- Added `ZeroCrossing` signal filter
- Added `Envelope` filter with asymmetric attack and release

### Changed

- Made "std" a default-enabled crate feature

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
