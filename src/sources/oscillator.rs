// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Oscillators for generating periodic waveforms.
//!
//! Provides stable recursive generators for sine, cosine, and other periodic signals.

#[macro_use]
pub(crate) mod macros;

pub mod sine;

#[cfg(any(feature = "libm", feature = "std"))]
pub mod chirp;

pub mod square;

pub mod pulse;

pub mod triangle;

pub mod sawtooth;
