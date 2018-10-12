// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration sinks.

use num_traits::Num;

use signalo_traits::{Filter, Finalize, Sink};

/// A sink that computes the integrate of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Integrate<T> {
    sum: Option<T>,
}

impl<T> Filter<T> for Integrate<T>
where
    T: Clone + Num,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let sum = self.sum.clone().unwrap_or(T::zero()) + input;

        self.sum = Some(sum.clone());

        sum
    }
}

impl<T> Sink<T> for Integrate<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for Integrate<T> {
    type Output = Option<T>;

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
