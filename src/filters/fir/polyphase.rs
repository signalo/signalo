// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR filter banks, executors, and multirate filters.
//!
//! [`filter_bank`] contains the coefficient storage and selected-phase execution
//! primitive. [`fir`] adds sample history. The [`interpolator`], [`decimator`], and
//! [`rational_resampler`] modules build streaming
//! [`MultirateFilter`](crate::traits::MultirateFilter) adapters on top of those
//! primitives.

pub mod decimator;
pub mod filter_bank;
pub mod fir;
pub mod interpolator;
pub mod rational_resampler;
