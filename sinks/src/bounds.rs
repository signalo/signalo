// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Bound sinks.

use signalo_traits::{Filter, Finalize, Sink};

use {max::Max, min::Min};

/// Output of `Bounds` filter.
pub struct Output<T> {
    /// Smallest value.
    pub min: T,
    /// Largest value.
    pub max: T,
}

/// A sink that computes the min and max of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Bounds<T> {
    min: Min<T>,
    max: Max<T>,
}

impl<T> Filter<T> for Bounds<T>
where
    T: Clone,
    Min<T>: Filter<T, Output = T>,
    Max<T>: Filter<T, Output = T>,
{
    type Output = Output<T>;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let min = self.min.filter(input.clone());
        let max = self.max.filter(input);
        Output { min, max }
    }
}

impl<T> Sink<T> for Bounds<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for Bounds<T>
where
    Min<T>: Finalize<Output = Option<T>>,
    Max<T>: Finalize<Output = Option<T>>,
{
    type Output = Option<Output<T>>;

    #[inline]
    fn finalize(self) -> Self::Output {
        let min = self.min.finalize();
        let max = self.max.finalize();
        match (min, max) {
            (Some(min), Some(max)) => Some(Output { min, max }),
            (None, None) => None,
            _ => unreachable!(),
        }
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
        let mut bounds: Bounds<usize> = Bounds::default();
        for input in input {
            bounds.sink(input);
        }
        let Output { min, max } = bounds.finalize().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 20);
    }
}
