// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter unit pipes.

use core::ops::BitOr;

use crate::traits::{Filter, Finalize, Sink, Source};

use super::pipe::Pipe;

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
        self.inner.sink(input);
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
    use std::vec;
    use std::vec::Vec;

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
        let input = [
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

    struct DummyFilterMul;

    impl Filter<Value> for DummyFilterMul {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input * 2
        }
    }

    #[test]
    fn test_bitor_operator() {
        let input = [1, 2, 3, 4, 5];
        let filter_add = UnitPipe::new(DummyFilterAdd);
        let filter_mul = DummyFilterMul;
        let mut pipe = filter_add | filter_mul;
        let subject: Vec<_> = input.iter().map(|&input| pipe.filter(input)).collect();
        let expected = vec![4, 6, 8, 10, 12];
        assert_eq!(subject, expected);
    }

    struct DummySource {
        values: Vec<Value>,
        index: usize,
    }

    impl Source for DummySource {
        type Output = Value;

        fn source(&mut self) -> Option<Self::Output> {
            if self.index < self.values.len() {
                let value = self.values[self.index];
                self.index += 1;
                Some(value)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_source() {
        let source = DummySource {
            values: vec![1, 2, 3, 4, 5],
            index: 0,
        };
        let mut pipe = UnitPipe::new(source);

        let mut results = vec![];
        while let Some(value) = pipe.source() {
            results.push(value);
        }

        let expected = vec![1, 2, 3, 4, 5];
        assert_eq!(results, expected);
    }

    struct DummySink {
        sum: Value,
    }

    impl Sink<Value> for DummySink {
        fn sink(&mut self, input: Value) {
            self.sum += input;
        }
    }

    impl Finalize for DummySink {
        type Output = Value;

        fn finalize(self) -> Self::Output {
            self.sum
        }
    }

    #[test]
    fn test_sink() {
        let sink = DummySink { sum: 0 };
        let mut pipe = UnitPipe::new(sink);

        let inputs = [1, 2, 3, 4, 5];
        for &input in &inputs {
            pipe.sink(input);
        }

        let result = pipe.finalize();
        assert_eq!(result, 15);
    }

    #[test]
    fn test_default() {
        #[derive(Default)]
        struct DefaultFilter;

        impl Filter<Value> for DefaultFilter {
            type Output = Value;

            fn filter(&mut self, input: Value) -> Self::Output {
                input
            }
        }

        let _pipe: UnitPipe<DefaultFilter> = UnitPipe::default();
    }
}
