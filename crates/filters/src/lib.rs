// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of filters used in 'signalo' umbrella crate.

#![warn(missing_docs)]
#![warn(
    // Coarse:
    clippy::all,
    // clippy::restriction,
    clippy::pedantic,
    // clippy::nursery,
    clippy::cargo,
    clippy::perf,
    clippy::style,
    clippy::correctness,
    // Fine:
    clippy::use_self,
    clippy::unimplemented,
    clippy::todo,
    clippy::else_if_without_else,
    clippy::unneeded_field_pattern,
    clippy::unwrap_used,
    clippy::wrong_self_convention,
)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub use signalo_traits as traits;

pub mod cache;
pub mod classify;
pub mod convolve;
pub mod delay;
pub mod differentiate;
pub mod hampel;
pub mod identity;
pub mod integrate;
pub mod mean;
pub mod median;
pub mod observe;
pub mod ops;
pub mod unit_system;
pub mod wavelet;
