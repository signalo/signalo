// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementations of `trait Source`.

mod pad;
mod from_iter;
mod take;
mod skip;
mod chain;
mod cycle;
mod repeat;
mod constant;
mod increment;

pub use self::pad::{PadConstant, PadEdge};
pub use self::from_iter::FromIter;
pub use self::take::Take;
pub use self::skip::Skip;
pub use self::chain::Chain;
pub use self::cycle::Cycle;
pub use self::repeat::Repeat;
pub use self::constant::Constant;
pub use self::increment::Increment;
