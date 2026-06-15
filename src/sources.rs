// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Composable signal generator sources.
//!
//! All sources implement the [`Source<T>`](crate::traits::Source) trait
//! and can be chained together in pipelines.

pub use crate::traits;

pub mod cache;

pub mod chain;

pub mod constant;

pub mod cycle;

pub mod from_iter;

pub mod impulse;

pub mod increment;

pub mod into_iter;

pub mod noise;

pub mod oscillator;

pub mod pad;

pub mod peek;

pub mod repeat;

pub mod skip;

pub mod step;

pub mod take;

pub mod unit_system;
