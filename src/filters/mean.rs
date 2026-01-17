// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Mean (aka "average") filters.

/// Exponential mean and exponential mean-variance filters.
///
/// Implements exponential smoothing (also called exponentially weighted moving average or EWMA),
/// which gives more weight to recent values. Useful for adaptive filtering and real-time signal smoothing.
pub mod exp;

#[allow(clippy::module_inception)]
/// Basic moving average filter.
///
/// Computes the arithmetic mean of values within a sliding window. Classic noise reduction filter
/// with linear phase response.
pub mod mean;

#[allow(clippy::module_name_repetitions)]
/// Mean with simultaneous variance computation.
///
/// Efficiently computes both mean and variance in a single pass over the data,
/// useful for statistical analysis and adaptive filtering applications.
pub mod mean_variance;
