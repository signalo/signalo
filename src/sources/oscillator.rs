// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Oscillators for generating periodic waveforms.
//!
//! Provides stable recursive generators for sine, cosine, and other periodic signals.

use num_traits::float::FloatCore;

#[macro_use]
pub(crate) mod macros;

pub mod sine;

pub mod nco;

#[cfg(any(feature = "libm", feature = "std"))]
pub mod chirp;

pub mod square;

pub mod pulse;

pub mod triangle;

pub mod sawtooth;

/// Maps a fractional sample position `input` to a wrapped phase in the oscillator's phase space,
/// as `(phase0 + increment * input).fract()`.
///
/// For non-negative `increment` and non-negative resulting phase this reproduces the phase used by
/// the oscillators' `Source::source` path. It does not renormalize negative phase: with a negative
/// `increment` (or a negative accumulated phase) the result lies in `(-1, 0]`, matching `.fract()`
/// rather than `Source::source`'s positive-only wrap.
pub(crate) fn sample_phase<T: FloatCore>(phase0: T, increment: T, input: T) -> T {
    (phase0 + increment * input).fract()
}
