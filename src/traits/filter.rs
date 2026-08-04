// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Signal transformers that accept an input value and produce an output.
//!
//! Filters are the primary building blocks for signal processing pipelines. They transform
//! input signals according to their internal state and logic, supporting complex operations
//! like filtering, decimation, and statistical analysis.

/// Filters accept values of a signal, and produce transformed values.
/// Furthermore, the result of concatenating all the output data is the same as the result
/// of applying the filter over the concatenation of the input data.
pub trait Filter<Input>: Sized {
    /// The filter's output type.
    type Output;

    /// Processes the input value, returning a corresponding output.
    fn filter(&mut self, input: Input) -> Self::Output;
}
