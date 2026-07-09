// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Rate-changing signal transformers with independent input and output progress.
//!
//! Multirate filters are the slice-based counterpart to
//! [`Filter`](super::Filter). They support systems where the number of output
//! samples is not necessarily equal to the number of input samples.

/// Filters that consume and produce variable numbers of samples.
///
/// The result of concatenating all produced output data is the same as the
/// result of applying the filter over the concatenation of the input data.
///
/// `process` accepts arbitrary input and output slice sizes. Implementations
/// return `(input_consumed, output_produced)`, where `input_consumed` is less
/// than or equal to `input.len()` and `output_produced` is less than or equal to
/// `output.len()`.
///
/// If the output slice fills before all input can be consumed, callers can pass
/// the unconsumed input to the next call. If input is exhausted before the next
/// output is available, callers can continue later with more input. If an output
/// becomes available when no output space remains, implementations must retain
/// it internally and produce it on a later call rather than dropping it.
pub trait MultirateFilter<Input>: Sized {
    /// The filter's output sample type.
    type Output;

    /// Processes input samples into output samples.
    ///
    /// Returns `(input_consumed, output_produced)`.
    fn process(&mut self, input: &[Input], output: &mut [Self::Output]) -> (usize, usize);
}
