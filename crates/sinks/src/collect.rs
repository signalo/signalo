// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Collection sinks.

#![cfg(feature = "std")]

use signalo_traits::{Filter, Finalize, Sink};

/// A sink that computes the integrate of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Collect<U> {
    collected: U,
}

impl<T> Filter<T> for Collect<Vec<T>>
where
    T: Clone,
{
    type Output = Vec<T>;

    // FIXME: add documentation pointing out the performance overhead
    // due to `self.collected.clone()`.
    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        self.sink(input);
        self.collected.clone()
    }
}

impl<T> Sink<T> for Collect<Vec<T>> {
    #[inline]
    fn sink(&mut self, input: T) {
        self.collected.push(input);
    }
}

impl<U> Finalize for Collect<U> {
    type Output = U;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.collected
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
        let expected = input.clone();
        let mut sink = Collect::default();
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, expected);
    }
}
