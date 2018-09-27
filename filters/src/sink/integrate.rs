// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration sinks.

use num_traits::Num;

use signalo_traits::sink::Sink;

/// A sink that computes the integrate of all received values of a signal.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_filters;
/// #
/// # fn main() {
/// use signalo_filters::traits::Source;
/// use signalo_filters::traits::Sink;
///
/// use signalo_filters::source::increment::Increment;
/// let increment: Increment<_> = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::source::take::Take;
/// let mut take: Take<_> = Take::new(increment, 3);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::sink::integrate::Integrate;
/// let mut integrate = Integrate::default();
/// while let Some(value) = take.source() {
///     integrate.sink(value);
/// }
/// assert_eq!(integrate.finalize(), Some(3));
/// # }
///```
#[derive(Clone, Default, Debug)]
pub struct Integrate<T> {
    sum: Option<T>,
}

impl<T> Sink<T> for Integrate<T>
where
    T: Clone + Num,
{
    type Output = Option<T>;

    #[inline]
    fn sink(&mut self, input: T) {
        let sum = self.sum.clone().unwrap_or(T::zero());
        self.sum = Some(sum + input);
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let mut sink = Integrate::default();
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, Some(196));
    }
}
