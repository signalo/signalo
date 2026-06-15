// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter implementations for signal transformation.

pub use crate::traits;

/// Classification filters mapping signals to discrete states.
pub mod classify;

/// State estimation filters (Kalman, Alpha-Beta).
pub mod estimate;

/// Finite Impulse Response (FIR) filters.
pub mod fir;

/// Infinite Impulse Response (IIR) filters with feedback.
pub mod iir;

/// Arithmetic operation filters for signal manipulation.
pub mod ops;

/// Rank-order and order-statistic filters.
pub mod rank;

/// Utility and wrapper filters.
pub mod util;

/// Wavelet analysis and synthesis filters.
pub mod wavelet;
