// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Add;

use num_traits::Zero;

use signalo_traits::sink::Sink;

/// A sink that computes the integrate of all received values of a signal.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_filters;
/// #
/// # fn main() {
/// use signalo_filters::prelude::Source;
/// use signalo_filters::prelude::Sink;
///
/// use signalo_filters::source::Increment;
/// let increment: Increment<_> = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::source::Take;
/// let mut take: Take<_> = Take::new(increment, 3);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::sink::Integrate;
/// let mut integrate = Integrate::new();
/// while let Some(value) = take.source() {
///     integrate.sink(value);
/// }
/// assert_eq!(integrate.finalize(), 3);
/// # }
///```
#[derive(Default, Clone, Debug)]
pub struct Integrate<T> {
    state: T,
}

impl<T> Integrate<T>
where
    T: Zero,
{
    /// Creates a new `Integrate` sink.
    #[inline]
    pub fn new() -> Self {
        Integrate { state: T::zero() }
    }
}

impl<T> Sink<T> for Integrate<T>
where
    T: Copy + Add<T, Output=T>,
{
    type Output = T;

    #[inline]
    fn sink(&mut self, input: T) {
        self.state = self.state + input;
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let mut sink = Integrate::new();
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, 196);
    }
}
