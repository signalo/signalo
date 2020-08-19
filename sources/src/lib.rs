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

pub mod cache;
pub mod chain;
pub mod constant;
pub mod cycle;
pub mod from_iter;
pub mod increment;
pub mod into_iter;
pub mod pad;
pub mod peek;
pub mod repeat;
pub mod skip;
pub mod take;
pub mod unit_system;
