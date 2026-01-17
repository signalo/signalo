// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! "Min value" sinks.

use crate::traits::{Filter, Finalize, Sink};

/// A sink that computes the min and max of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Min<T> {
    min: Option<T>,
}

impl<T> Filter<T> for Min<T>
where
    T: Clone + PartialOrd,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let min = if let Some(ref min) = self.min {
            if &input < min {
                input
            } else {
                min.clone()
            }
        } else {
            input
        };
        self.min = Some(min.clone());
        min
    }
}

impl<T> Sink<T> for Min<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for Min<T> {
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.min
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let mut sink = Min::default();
        for input in input {
            sink.sink(input);
        }
        let min = sink.finalize().unwrap();
        assert_eq!(min, 0);
    }
}
