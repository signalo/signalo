// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filters of a digital signal.

/// Filters accept values of a signal, and produce transformed values.
/// Furthermore, the result of concatenating all the output data is the same as the result
/// of applying the filter over the concatenation of the input data.
pub trait Filter<Input>: Sized {
    /// The filter's output type.
    type Output;

    /// Processes the input value, returning a corresponding output.
    fn filter(&mut self, input: Input) -> Self::Output;
}

impl<F, T, U> Filter<T> for F
where
    F: FnMut(T) -> U,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self(input)
    }
}
