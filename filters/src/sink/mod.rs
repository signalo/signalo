// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementations of `trait Sink`.

mod bounds;
mod integrate;
mod last;
mod max;
mod mean;
mod mean_variance;
mod min;

pub use self::bounds::Bounds;
pub use self::integrate::Integrate;
pub use self::last::Last;
pub use self::max::Max;
pub use self::mean::Mean;
pub use self::mean_variance::MeanVariance;
pub use self::min::Min;
