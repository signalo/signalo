// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter unit pipes.

use std::ops::BitOr;

use signalo_traits::filter::Filter;
use signalo_traits::sink::Sink;
use signalo_traits::source::Source;

use filter::pipe::Pipe;

/// A `UnitPipe` is a simple container wrapping a `Filter`
///
/// ```plain
/// ════════════
///  ╭────────╮
///  │ Filter │
///  ╰────────╯
/// ════════════
/// └─┬────────┘
///   └ UnitPipe
/// ```
#[derive(Default, Clone, Debug)]
pub struct UnitPipe<T> {
    filter: T,
}

impl<T> UnitPipe<T> {
    /// Creates a new unit pipe wrapping `filter`.
    #[inline]
    pub fn new(filter: T) -> Self {
        Self { filter }
    }
}

impl<T, Rhs> BitOr<Rhs> for UnitPipe<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, I> Filter<I> for UnitPipe<T>
where
    T: Filter<I>,
{
    type Output = T::Output;

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.filter.filter(input)
    }
}

impl<T> Source for UnitPipe<T>
where
    T: Filter<()>,
{
    type Output = T::Output;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        Some(self.filter(()))
    }
}

impl<T, I> Sink<I> for UnitPipe<T>
where
    T: Filter<I, Output = ()>,
{
    type Output = T::Output;

    #[inline]
    fn sink(&mut self, input: I) {
        self.filter(input)
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        ()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Value = usize;

    struct DummyFilterAdd;

    impl Filter<Value> for DummyFilterAdd {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input + 1
        }
    }

    #[test]
    fn test() {
        let input = vec![
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let filter = DummyFilterAdd;
        let pipe = UnitPipe::new(filter);
        let subject: Vec<_> = input
            .iter()
            .scan(pipe, |pipe, &input| Some(pipe.filter(input)))
            .collect();
        let expected = vec![
            1, 2, 8, 3, 6, 9, 17, 4, 20, 7, 15, 10, 10, 18, 18, 5, 13, 21, 21, 8,
        ];
        assert_eq!(subject, expected);
    }
}
