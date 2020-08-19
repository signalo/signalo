// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The 'signalo' umbrella crate.

#![warn(missing_docs)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

#[cfg(all(not(test), not(feature = "std")))]
extern crate core as std;

#[cfg(all(test, feature = "std"))]
extern crate std;

pub extern crate signalo_filters;
pub extern crate signalo_pipes;
pub extern crate signalo_traits;

pub use signalo_filters as filters;
pub use signalo_pipes as pipes;
pub use signalo_traits as traits;

/// The crate's prelude.
pub mod prelude {
    pub use signalo_filters::prelude as filters;
    pub use signalo_pipes::prelude as pipes;
    pub use signalo_traits::prelude as traits;
}
