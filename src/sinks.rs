// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Signal consumer/accumulator implementations.

pub use crate::traits;

pub mod bounds;

pub mod collect;

pub mod correlation;

pub mod integrate;

pub mod last;

pub mod max;

pub mod mean;

pub mod mean_variance;

pub mod min;

pub mod statistics;

pub mod unit_system;

pub mod rms;

pub mod peak_hold;

pub mod histogram;

pub mod percentile;
