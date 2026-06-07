// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Signal generator implementations.

pub use crate::traits;

/// Cache source that stores and repeats the most recent generated value.
///
/// Wraps another source and caches its last output, allowing repeated access without
/// requesting new values from the underlying source.
pub mod cache;

/// Sequential chaining of two sources in series.
///
/// Combines two sources such that the second source begins generating values after the first
/// source is exhausted, useful for concatenating signal sequences.
pub mod chain;

/// Constant DC source generating a single repeated value.
///
/// Produces an infinite stream of identical values, useful as a baseline signal or for
/// testing and placeholder generation.
pub mod constant;

/// Cyclic buffering source repeating a fixed sequence.
///
/// Cycles through a predefined array of values repeatedly, useful for generating periodic
/// waveforms and test patterns.
pub mod cycle;

/// Wrapper converting Iterator to Source trait implementation.
///
/// Adapts Rust iterators into the Source trait, enabling any iterator to be used in
/// signal processing pipelines without modification.
pub mod from_iter;

/// Unit impulse signal source.
///
/// Generates a single-sample pulse at time zero with zero amplitude thereafter, representing
/// the discrete-time unit impulse (Dirac delta function). Fundamental for impulse response
/// measurement and filter characterization.
pub mod impulse;

/// Linear increment/ramp generator with configurable step.
///
/// Generates linearly increasing or decreasing values with each call, useful for creating
/// ramps, sweeps, and testing linear behavior.
pub mod increment;

/// Wrapper converting IntoIterator to Source trait implementation.
///
/// Adapts types implementing IntoIterator into the Source trait, enabling flexible
/// integration with Rust's iterator ecosystem.
pub mod into_iter;

/// Noise source using xorshift32 pseudorandom number generator.
///
/// Generates deterministic sequences of pseudorandom u32 values with configurable seeding,
/// suitable for DSP and embedded applications requiring fast, simple PRNG behavior.
pub mod noise;

/// Oscillators for generating periodic waveforms.
///
/// Provides stable recursive generators for sine, cosine, and other periodic signals.
pub mod oscillator;

/// Padding source that extends sequences with padding values.
///
/// Appends repeated padding values to another source's output, useful for extending signals
/// with silence, edge values, or zero-padding.
pub mod pad;

/// Peek-ahead wrapper allowing inspection without consuming from the source.
///
/// Provides read-only access to upcoming values without advancing the source, enabling
/// lookahead and conditional processing.
pub mod peek;

/// Repetition source repeating a value N times.
///
/// Generates a fixed number of copies of a single value, useful for pulse generation and
/// discrete signal construction.
pub mod repeat;

/// Skip source that discards first N values from inner source.
///
/// Ignores a specified number of values from the underlying source before beginning generation,
/// useful for seeking and alignment in signal streams.
pub mod skip;

/// Unit step signal source.
///
/// Generates an infinite stream of a constant amplitude value, representing the discrete-time
/// unit step function (Heaviside function). Useful for step response analysis and system testing.
pub mod step;

/// Take source limiting output to first N values.
///
/// Generates at most N values from the underlying source before terminating, useful for
/// windowing and finite sample extraction.
pub mod take;

/// Unit-aware source support with dimensional types.
///
/// Enables type-safe signal generation with physical units, ensuring dimensional consistency
/// throughout source chains.
pub mod unit_system;
