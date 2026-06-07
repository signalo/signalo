// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Oscillators for generating periodic waveforms.

#[macro_use]
pub(crate) mod macros;

/// Sine wave oscillator using stable recursive generation.
///
/// Generates sinusoidal output using a recursive oscillator structure that maintains
/// amplitude stability over extended generation periods without accumulated phase drift.
pub mod sine;

/// Chirp (frequency sweep) source for frequency analysis.
///
/// Generates a finite-duration source that sweeps frequency linearly from a start
/// frequency to an end frequency over a specified number of samples.
#[cfg(feature = "std")]
pub mod chirp;

/// Square wave oscillator with configurable duty cycle.
///
/// Generates square wave output with a fixed 50% duty cycle.
/// Amplitude and frequency are configurable.
pub mod square;

/// Pulse wave oscillator with configurable duty cycle.
///
/// Generates pulse wave output with a configurable duty cycle, allowing asymmetric
/// waveforms where the positive portion of the period can be varied from 0% to 100%.
pub mod pulse;

/// Triangle wave oscillator with linear ramps.
///
/// Generates triangle wave output with equal rise and fall times,
/// producing linear ramps between positive and negative peaks.
pub mod triangle;

/// Sawtooth wave oscillator with linear ramps.
///
/// Generates sawtooth wave output that rises linearly from negative to positive peaks
/// over each period before resetting.
pub mod sawtooth;
