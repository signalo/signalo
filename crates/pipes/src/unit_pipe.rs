// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter unit pipes.

use std::ops::BitOr;

use signalo_traits::{Filter, Finalize, Sink, Source};

use crate::pipe::Pipe;

/// A `UnitPipe` is a simple container wrapping a `Filter`/`Source`/`Sink`/`Finalize` impl.
#[derive(Default, Clone, Debug)]
pub struct UnitPipe<T> {
    inner: T,
}

impl<T> UnitPipe<T> {
    /// Creates a new unit pipe wrapping `inner`.
    #[inline]
    pub fn new(inner: T) -> Self {
        Self { inner }
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
        self.inner.filter(input)
    }
}

impl<T> Source for UnitPipe<T>
where
    T: Source,
{
    type Output = T::Output;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        self.inner.source()
    }
}

impl<T, I> Sink<I> for UnitPipe<T>
where
    T: Sink<I>,
{
    #[inline]
    fn sink(&mut self, input: I) {
        self.inner.sink(input)
    }
}

impl<T> Finalize for UnitPipe<T>
where
    T: Finalize,
{
    type Output = T::Output;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.inner.finalize()
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
