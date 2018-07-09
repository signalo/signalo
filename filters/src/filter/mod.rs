// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementations of `trait Filter`.

mod identity;
mod differentiate;

pub mod ops;
pub mod mean;
pub mod median;
pub mod observe;
pub mod convolve;
pub mod classify;

pub use self::identity::Identity;
pub use self::differentiate::Differentiate;
pub use self::convolve::Convolve;
