// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! FIR window functions for spectral analysis and filter design.
//!
//! # Important: streaming vs. frame mode
//!
//! Each window is a stateful streaming filter that multiplies input samples by
//! a window coefficient `w[k mod N]`. This is a **periodic mask**: the same
//! sequence `w[0], w[1], …, w[N−1]` repeats every N calls, indefinitely.
//!
//! This differs from the typical use of window functions for frame-based
//! FFT pre-processing or FIR filter design:
//!
//! - Windows that are zero at both endpoints (Hann, Blackman, Blackman-Harris,
//!   Flat-top) will **zero the output** every N samples — the output train has
//!   hard zeros at periodic intervals.
//! - The result acts as an AM-modulated comb, not a conventional apodisation
//!   window.
//!
//! For frame-based windowing, construct the weights via `Config::new()` on
//! each window type and apply them manually to your frame, or use the
//! [`windowed_sinc`](super::convolve::windowed_sinc) module for window-based
//! FIR filter design.
//!

pub mod blackman;
pub mod blackman_harris;
pub mod flat_top;
pub mod hamming;
pub mod hann;
pub mod kaiser;
pub mod rectangular;
pub mod triangular;
