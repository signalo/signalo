// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of filters used in 'signalo' umbrella crate.

#![warn(missing_docs)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

#[cfg(all(not(test), not(feature = "std")))]
extern crate core as std;

#[cfg(all(test, feature = "std"))]
extern crate std;

pub extern crate num_traits;

#[cfg(feature = "dimensioned")]
pub extern crate dimensioned;

#[cfg(test)]
#[macro_use]
extern crate nearly_eq;

pub extern crate signalo_traits;

pub use signalo_traits as traits;

/// The crate's prelude.
pub mod prelude {}

pub mod bounds;
pub mod collect;
pub mod integrate;
pub mod last;
pub mod max;
pub mod mean;
pub mod mean_variance;
pub mod min;
pub mod statistics;
pub mod unit_system;
