// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementations of `trait Sink`.

mod sum;
mod last;
mod mean_variance;

pub use self::sum::Sum;
pub use self::last::Last;
pub use self::mean_variance::MeanVariance;
