// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of filters used in 'signalo' umbrella crate.

#![cfg_attr(feature = "nightly", feature(try_from))]

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]

#![warn(missing_docs)]

extern crate num_traits;
extern crate num_integer;

extern crate arraydeque;

#[cfg(test)]
#[macro_use]
extern crate nearly_eq;

extern crate signalo_traits;

pub mod source;
pub mod sink;
pub mod filter;
pub mod traits;

/// The crate's prelude.
pub mod prelude {
    pub use signalo_traits::source::Source;
    pub use signalo_traits::sink::Sink;
    pub use signalo_traits::filter::Filter;
}
