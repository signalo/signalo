// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::BitOr;

use signalo_traits::filter::Filter;
use signalo_traits::sink::Sink;

/// A `Pipe` is a simple container joining a pair of a `Filter` and a `Sink`
///
/// ```plain
/// ╠════════════ + ════════════
/// ║ ╭────────╮  +  ╭────────╮
/// ║ │ Filter │  +  │  Sink  │
/// ║ ╰────────╯  +  ╰────────╯
/// ╠════════════ + ════════════
/// └─┬───────────────────────┘
///   └ Pipe
/// ```
#[derive(Default, Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U>
{
    /// Creates a new pipe connecting `lhs` and `rhs`.
    #[inline]
    pub fn new(lhs: T, rhs: U) -> Self {
        Self { lhs, rhs }
    }
}

impl<T, U> From<(T, U)> for Pipe<T, U> {
    #[inline]
    fn from(parts: (T, U)) -> Self {
        let (lhs, rhs) = parts;
        Self::new(lhs, rhs)
    }
}

impl<T, U, Rhs> BitOr<Rhs> for Pipe<T, U> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, U, I> Sink<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Sink<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn sink(&mut self, input: I) {
        self.rhs.sink(self.lhs.filter(input))
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.rhs.finalize()
    }
}

impl<T, U, I> Filter<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Sink<T::Output>,
{
    type Output = ();

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.rhs.sink(self.lhs.filter(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Value = usize;

    struct DummySink {
        sum: usize
    }

    impl Sink<Value> for DummySink {
        type Output = Value;

        #[inline]
        fn sink(&mut self, input: Value) {
            self.sum += input;
        }

        #[inline]
        fn finalize(self) -> Self::Output {
            self.sum
        }
    }

    struct DummyFilter;

    impl Filter<Value> for DummyFilter {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input + 1
        }
    }

    #[test]
    fn test() {
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let filter = DummyFilter;
        let sink = DummySink { sum: 0 };
        let mut pipe = Pipe::new(filter, sink);
        for i in input {
            pipe.sink(i);
        }
        let subject = pipe.finalize();
        let expected = 216;
        assert_eq!(subject, expected);
    }
}
