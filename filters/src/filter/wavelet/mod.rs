// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet filters.

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
