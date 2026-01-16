// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic operation filters.

/// Addition filter for combining signals element-wise.
///
/// Adds a constant value to each input signal sample.
pub mod add;

/// Division filter for signal scaling.
///
/// Divides each input signal sample by a constant divisor.
pub mod div;

/// Multiplication filter for signal scaling.
///
/// Multiplies each input signal sample by a constant factor.
pub mod mul;

/// Remainder filter for modulo operations.
///
/// Computes the remainder of each input signal sample divided by a divisor.
pub mod rem;

/// Subtraction filter for signal offset and difference operations.
///
/// Subtracts a constant value from each input signal sample.
pub mod sub;

/// Negation filter that inverts signal amplitude.
///
/// Negates each input value, equivalent to multiplying by -1.
pub mod neg;

/// Squaring filter for non-linear signal transformation.
///
/// Squares each input value, useful for power calculation and envelope extraction.
pub mod square;
