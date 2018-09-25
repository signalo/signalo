// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementations of `trait Filter`.

pub mod classify;
pub mod convolve;
pub mod delay;
pub mod differentiate;
pub mod hampel;
pub mod identity;
pub mod mean;
pub mod median;
pub mod observe;
pub mod ops;
pub mod wavelet;
