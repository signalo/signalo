// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic operation filters; element-wise signal manipulation.
//!
//! Unlike other filter groups, these filters have no internal state and no windowing.
//! Each operates on a per-sample basis, transforming input values through simple
//! arithmetic. They implement the [`Filter`](crate::traits::Filter)`<T>` trait, making
//! them composable in filter pipelines for inline computation without buffering.
//!
//! # Binary operations
//!
//! | Filter       | Operation                        | Typical use                       |
//! | ------------ | -------------------------------- | --------------------------------- |
//! | `add::Add`   | `output = input + constant`      | DC offset / bias addition         |
//! | `sub::Sub`   | `output = input − constant`      | DC offset removal                 |
//! | `mul::Mul`   | `output = input × constant`      | Gain scaling / amplification      |
//! | `div::Div`   | `output = input ÷ constant`      | Attenuation / normalization       |
//! | `rem::Rem`   | `output = input % constant`      | Wrapping, phase modulo, remainder |
//!
//! # Unary operations
//!
//! | Filter           | Operation               | Typical use                     |
//! | ---------------- | ----------------------- | ------------------------------- |
//! | `neg::Neg`       | `output = −input`       | Signal inversion / polarity flip|
//! | `square::Square` | `output = input²`       | Power / energy computation      |
//!
//! # Composition
//!
//! Operations are designed to be stacked. For example, `Add` → `Mul` first adds a DC
//! offset then scales the result. Because there is no internal state, order is
//! commutative for independent offsets and gains.
//!
//! # See also
//!
//! - [`super::util`]: wrapper and infrastructure filters (delay lines, identity, last-value
//!   caching) that also implement `Filter<T>` but serve structural rather than arithmetic
//!   roles.

pub mod add;

pub mod div;

pub mod mul;

pub mod rem;

pub mod sub;

pub mod neg;

pub mod square;
