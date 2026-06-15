// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Finite Impulse Response (FIR) filters; no feedback, always stable.
//!
//! FIR filters compute each output sample as a weighted sum of current and past input
//! samples only. There is no feedback path: past outputs do not influence the current
//! output. This makes FIR filters inherently stable and capable of exact linear phase
//! (constant group delay), at the cost of higher order than an equivalent IIR design.
//!
//! # When to use which filter
//!
//! | Filter                  | Purpose                                                |
//! | ----------------------- | ------------------------------------------------------ |
//! | `convolve::Convolve`    | General-purpose FIR via arbitrary coefficient kernels  |
//! | `differentiate`         | First derivative / rate-of-change estimation           |
//! | `comb::FeedforwardComb` | Delay-line feedforward comb (no resonant feedback)    |
//! | `mean`                  | Sliding-window arithmetic mean for noise reduction     |
//! | `mean_variance`         | Sliding-window mean and variance (two statistics)      |
//! | `savitzky_golay`        | Polynomial smoothing / differentiation (Savitzky-Golay)|
//!
//! - **Convolve** is the general-purpose workhorse: supply any coefficient kernel
//!   and it performs convolution. Use when no specialized filter fits.
//! - **`FeedforwardComb`** adds a delayed copy of the input (feedforward). Unlike the IIR
//!   `comb` filter, it has no feedback resonance; output is a finite-length echo.
//! - **Mean** and **`mean_variance`** are sliding-window statistics, useful for basic
//!   smoothing, noise estimation, and signal quality monitoring.
//! - **Differentiate** computes a running estimate of the first derivative, useful for
//!   edge detection, velocity estimation, and trend analysis.
//! - **Savitzky-Golay** fits a low-degree polynomial over a sliding window, providing
//!   smoothing that preserves higher moments of the signal (peak height/width) better
//!   than a simple moving average.
//!
//! # See also
//!
//! - [`super::iir`]: recursive (feedback) alternatives for when filter order matters more than
//!   phase linearity, or when resonant/ringing behavior is desired.
//! - [`super::rank::median`]: edge-preserving smoothing; a non-linear alternative to the
//!   linear smoothing provided by `mean` and `savitzky_golay`.
//! - [`super::wavelet`]: multi-resolution decomposition built on FIR convolution.

pub mod comb;
pub mod convolve;
pub mod differentiate;
pub mod mean;
pub mod mean_variance;
pub mod savitzky_golay;
