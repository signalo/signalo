// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filter pipes.

use core::ops::BitOr;

use crate::traits::{Filter, Finalize, Sink, Source};

/// A `Pipe` is a simple container joining a pair of `Filter`/`Source`/`Sink`/`Finalize` impls.
#[derive(Default, Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U> {
    /// Creates a new pipe connecting `lhs` and `rhs`.
    #[inline]
    pub fn new(lhs: T, rhs: U) -> Self {
        Self { lhs, rhs }
    }
}

impl<T, U, Rhs> BitOr<Rhs> for Pipe<T, U> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, U, I> Filter<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Filter<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.rhs.filter(self.lhs.filter(input))
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

impl<T, U, I> Sink<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Sink<T::Output>,
{
    #[inline]
    fn sink(&mut self, input: I) {
        self.rhs.sink(self.lhs.filter(input));
    }
}

impl<T, U> Finalize for Pipe<T, U>
where
    U: Finalize,
{
    type Output = U::Output;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.rhs.finalize()
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

    struct DummyFilterMul;

    impl Filter<Value> for DummyFilterMul {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input * 2
        }
    }

    #[test]
    fn test() {
        let input = [
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let filter_add = DummyFilterAdd;
        let filter_mul = DummyFilterMul;
        let pipe = Pipe::new(filter_add, filter_mul);
        let subject: Vec<_> = input
            .iter()
            .scan(pipe, |pipe, &input| Some(pipe.filter(input)))
            .collect();
        let expected = vec![
            2, 4, 16, 6, 12, 18, 34, 8, 40, 14, 30, 20, 20, 36, 36, 10, 26, 42, 42, 16,
        ];
        assert_eq!(subject, expected);
    }

    #[test]
    fn test_bitor_operator() {
        let input = [1, 2, 3, 4, 5];
        let pipe1 = Pipe::new(DummyFilterAdd, DummyFilterMul);
        let mut pipe = pipe1;
        let subject: Vec<_> = input.iter().map(|&input| pipe.filter(input)).collect();
        let expected = vec![4, 6, 8, 10, 12];
        assert_eq!(subject, expected);
    }

    #[test]
    fn test_chained_bitor() {
        let input = [1, 2, 3];
        let pipe1 = Pipe::new(DummyFilterAdd, DummyFilterMul);
        let mut pipe = pipe1 | DummyFilterAdd;
        let subject: Vec<_> = input.iter().map(|&input| pipe.filter(input)).collect();
        // (1 + 1) * 2 + 1 = 5
        // (2 + 1) * 2 + 1 = 7
        // (3 + 1) * 2 + 1 = 9
        let expected = vec![5, 7, 9];
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
    fn test_source_pipe() {
        let source = DummySource {
            values: vec![1, 2, 3, 4, 5],
            index: 0,
        };
        let filter = DummyFilterMul;
        let mut pipe = Pipe::new(source, filter);

        let mut results = vec![];
        while let Some(value) = pipe.source() {
            results.push(value);
        }

        let expected = vec![2, 4, 6, 8, 10];
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
    fn test_sink_pipe() {
        let filter = DummyFilterAdd;
        let sink = DummySink { sum: 0 };
        let mut pipe = Pipe::new(filter, sink);

        let inputs = [1, 2, 3, 4, 5];
        for &input in &inputs {
            pipe.sink(input);
        }

        let result = pipe.finalize();
        // (1+1) + (2+1) + (3+1) + (4+1) + (5+1) = 2 + 3 + 4 + 5 + 6 = 20
        assert_eq!(result, 20);
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

        let _pipe: Pipe<DefaultFilter, DefaultFilter> = Pipe::default();
    }
}
