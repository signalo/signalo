// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter implementations for signal transformation.

pub use crate::traits;

/// Min/max moving window filters that track boundary values.
///
/// Provides efficient computation of the minimum and maximum values within a sliding window
/// using optimized data structures. Useful for envelope detection and peak/valley analysis.
pub mod bounds;

/// Cache filter that stores and returns the most recent output value.
///
/// Wraps another filter and caches its last output, allowing repeated access to the most recent
/// result without recomputation.
pub mod cache;

/// Classification filters mapping signals to discrete states.
///
/// Includes threshold detection, debouncing, Schmitt triggers, and peak/slope detection for
/// converting continuous signals into discrete events or classifications.
pub mod classify;

/// Convolution-based filters including polynomial smoothing.
///
/// Implements convolution operations and Savitzky-Golay polynomial smoothing filters for
/// noise reduction while preserving signal features and derivatives.
pub mod convolve;

/// Delay filter storing a circular buffer of tap values.
///
/// Provides time-domain delay and tap access, enabling multi-tap filtering, delay-based
/// operations, and finite impulse response (FIR) filter implementations.
pub mod delay;

/// Numerical differentiation (first derivative) filter.
///
/// Computes the discrete first derivative of input signals, useful for edge detection,
/// rate analysis, and derivative-based signal processing.
pub mod differentiate;

/// Outlier-resistant Hampel median filter.
///
/// Implements the Hampel filter for robust noise reduction using median and median absolute
/// deviation (MAD) statistics to identify and replace outliers.
pub mod hampel;

/// Pass-through identity filter that returns input unchanged.
///
/// A trivial filter useful for pipeline composition, type conversions, and as a placeholder
/// in generic code.
pub mod identity;

/// Numerical integration (cumulative sum) filter.
///
/// Computes the discrete integral or cumulative sum of input signals, useful for slope
/// extraction, position tracking, and accumulation operations.
pub mod integrate;

/// Mean filters including arithmetic and exponential mean with variance tracking.
///
/// Implements moving average and exponential smoothing filters with optional variance
/// computation for statistical analysis and adaptive filtering.
pub mod mean;

/// Moving median filter with O(n) complexity.
///
/// Efficiently computes the median value within a sliding window, useful for robust
/// noise reduction while preserving edges and discontinuities.
pub mod median;

/// Observer filters including Kalman and Alpha-Beta filters.
///
/// State-space observers for optimal recursive estimation of signals with noise, supporting
/// both fixed-gain and adaptive estimation strategies.
pub mod observe;

/// Arithmetic operation filters for signal manipulation.
///
/// Provides filters for basic arithmetic operations: addition, multiplication, division,
/// remainder, negation, and squaring applied element-wise to signals.
pub mod ops;

/// Unit-aware signal processing support with dimensional types.
///
/// Enables type-safe signal processing with physical units, ensuring dimensional consistency
/// across filter chains and preventing unit-related errors.
pub mod unit_system;

/// Wavelet analysis and synthesis filters.
///
/// Implements discrete wavelet transforms and related filters for multi-resolution signal
/// analysis, compression, and feature extraction.
pub mod wavelet;
