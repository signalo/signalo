// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet analysis and synthesis filters; multi-resolution signal decomposition.
//!
//! Wavelet transforms decompose a signal into components at different frequency scales,
//! providing simultaneous time and frequency localization. Unlike the Fourier transform
//! (which loses all time information), wavelets reveal *when* frequency content changes;
//! essential for analyzing non-stationary signals with transients, edges, or bursts.
//!
//! # The `Decomposition` type
//!
//! Both analysis and synthesis operate on [`Decomposition`]`<T>`, a pair of:
//! - **low** (approximation): the smoothed, low-frequency component of the signal.
//! - **high** (detail): the residual high-frequency component.
//!
//! # When to use which filter
//!
//! | Filter                       | Purpose                                                    |
//! | ---------------------------- | ---------------------------------------------------------- |
//! | `analyze::Analyze`           | Forward wavelet transform; decompose into low/high bands  |
//! | `synthesize::Synthesize`     | Inverse wavelet transform; reconstruct from low/high bands|
//! | `daubechies`                 | Daubechies wavelet family (coefficient generation)         |
//!
//! - **Analyze** performs one level of the forward discrete wavelet transform (DWT):
//!   it convolves the signal with low-pass and high-pass decomposition filters, then
//!   downsamples. The result is a `Decomposition` containing the approximation and
//!   detail coefficients.
//! - **Synthesize** is the inverse: upsamples the approximation and detail
//!   coefficients, convolves with reconstruction filters, and sums to recover the
//!   original signal (up to numerical precision).
//! - **Daubechies** provides the coefficient sets for the Daubechies family of
//!   compactly-supported orthogonal wavelets (e.g. `db2`, `db4`, `db8`). These
//!   coefficients are consumed by `Analyze` and `Synthesize` as the filter banks.
//!
//! # See also
//!
//! - [`super::fir::convolve`]: wavelet transforms are built on FIR convolution; the analysis
//!   and synthesis filter banks are applied via convolution with the wavelet and
//!   scaling coefficients.
//! - [`super::fir`]: general-purpose FIR filters for convolution and smoothing, which can
//!   serve as building blocks for custom filter banks.

pub mod analyze;

pub mod daubechies;

pub mod synthesize;

/// Result of a wavelet analysis (or input of a wavelet synthesis).
#[derive(Clone, Debug)]
pub struct Decomposition<T> {
    /// Low-frequency component.
    pub low: T,
    /// High-frequency component.
    pub high: T,
}
