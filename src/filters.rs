// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Composable signal-processing filters.
//!
//! All filters implement the [`Filter<T>`](crate::traits::Filter) trait
//! and can be chained together in pipelines.

pub use crate::traits;

pub mod classify;

pub mod estimate;

pub mod fir;

pub mod iir;

pub mod ops;

pub mod rank;

pub mod util;

pub mod wavelet;
