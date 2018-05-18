// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filters that map a signal onto two discrete values (on, off, e.g.).

#![macro_use]

mod debounce;
mod schmitt;
mod threshold;

pub use self::debounce::Debounce;
pub use self::schmitt::Schmitt;
pub use self::threshold::Threshold;