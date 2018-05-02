// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::BitOr;

use signalo_traits::filter::Filter;
use signalo_traits::source::Source;

/// A `Pipe` is a simple container joining a pair of a `Source` and a `Filter`
///
/// ```plain
/// ╠════════════ + ════════════
/// ║ ╭────────╮  +  ╭────────╮
/// ║ │ Source │  +  │ Filter │
/// ║ ╰────────╯  +  ╰────────╯
/// ╠════════════ + ════════════
/// └─┬───────────────────────┘
///   └ Pipe
/// ```
#[derive(Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U>
{
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

impl<T, U> Source for Pipe<T, U>
where
    T: Source,
    U: Filter<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        self.lhs.source().map(|input| self.rhs.filter(input))
    }
}

impl<T, U> Filter<()> for Pipe<T, U>
where
    T: Source,
    U: Filter<T::Output>,
{
    type Output = Option<U::Output>;

    #[inline]
    fn filter(&mut self, _input: ()) -> Self::Output {
        self.lhs.source().map(|input| self.rhs.filter(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Value = usize;
    const VALUE: Value = 42;

    struct DummySource;

    impl Source for DummySource {
        type Output = Value;

        #[inline]
        fn source(&mut self) -> Option<Self::Output> {
            Some(VALUE)
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
    fn source() {
        const COUNT: usize = 3;
        let pipe = Pipe::new(DummySource, DummyFilter);
        let subject: Vec<_> = (0..COUNT).scan(pipe, |pipe, _| {
            pipe.source()
        }).collect();
        let expected = vec![VALUE + 1; COUNT];
        assert_eq!(subject, expected);
    }
}
