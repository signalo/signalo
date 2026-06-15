// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter implementations for signal transformation.
//!
//! Contains a variety of filter types including moving averages, median filters, differentiation,
//! integration, convolution, and state observers (Kalman, Alpha-Beta).

pub use crate::traits;

pub mod classify;

pub mod estimate;

pub mod fir;

pub mod iir;

pub mod ops;

pub mod rank;

pub mod util;

pub mod wavelet;
