// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet filters.

/// Wavelet analysis (decomposition) filter.
///
/// Decomposes signals into low-frequency (approximation) and high-frequency (detail) components
/// using wavelet transforms for multi-resolution analysis.
pub mod analyze;

/// Daubechies wavelet implementation.
///
/// Provides Daubechies wavelets with configurable order, commonly used for signal analysis
/// and compression due to their compact support and smoothness properties.
pub mod daubechies;

/// Wavelet synthesis (reconstruction) filter.
///
/// Reconstructs signals from low-frequency and high-frequency components, inverse operation
/// of analysis enabling signal recovery and manipulation in wavelet domain.
pub mod synthesize;

/// Result of a wavelet analysis (or input of a wavelet synthesis).
#[derive(Clone, Debug)]
pub struct Decomposition<T> {
    /// Low-frequency component.
    pub low: T,
    /// High-frequency component.
    pub high: T,
}
