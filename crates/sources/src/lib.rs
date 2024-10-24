// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of filters used in 'signalo' umbrella crate.

#![warn(missing_docs)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub use signalo_traits as traits;

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
