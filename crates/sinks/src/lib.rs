// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of filters used in 'signalo' umbrella crate.

#![warn(missing_docs)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub use signalo_traits as traits;

/// Sink computing min/max bounds of an input signal.
///
/// Tracks both the minimum and maximum values seen across all inputs,
/// useful for signal range analysis and normalization.
pub mod bounds;

/// Sink collecting all values into a vector.
///
/// Accumulates all input values into a heap-allocated vector, useful for buffering signals
/// and later analysis. Requires the `std` or `alloc` feature.
pub mod collect;

/// Sink computing cumulative sum/integration over input values.
///
/// Accumulates inputs through running addition, effectively computing a discrete integral
/// over the signal, useful for energy and area calculations.
pub mod integrate;

/// Sink returning only the last received value.
///
/// Discards all intermediate values and produces only the final input sample, useful for
/// extracting end states and terminal values.
pub mod last;

/// Sink tracking maximum value seen.
///
/// Maintains the largest value encountered across all inputs, useful for finding peaks
/// and signal amplitude analysis.
pub mod max;

/// Sink computing arithmetic mean of input values.
///
/// Calculates the average of all input samples, useful for signal level estimation and
/// statistical analysis.
pub mod mean;

/// Sink computing mean and variance simultaneously.
///
/// Efficiently computes both mean and variance in a single pass, useful for statistical
/// analysis and adaptive filtering without two passes over data.
pub mod mean_variance;

/// Sink tracking minimum value seen.
///
/// Maintains the smallest value encountered across all inputs, useful for finding troughs
/// and baseline detection.
pub mod min;

/// Sink computing comprehensive descriptive statistics.
///
/// Computes multiple statistics (min, max, mean, variance, etc.) simultaneously from input
/// values, providing complete statistical characterization.
pub mod statistics;

/// Unit-aware signal analysis support with dimensional types.
///
/// Enables type-safe signal analysis with physical units, ensuring dimensional consistency
/// in statistical operations and measurements.
pub mod unit_system;
