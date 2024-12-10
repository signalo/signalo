// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Mean (aka "average") filters.

pub mod exp;

#[allow(clippy::module_inception)]
pub mod mean;

#[allow(clippy::module_name_repetitions)]
pub mod mean_variance;
