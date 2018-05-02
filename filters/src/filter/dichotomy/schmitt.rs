// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::{PartialOrd, PartialEq};

use signalo_traits::filter::Filter;
use traits::Stateful;

/// A [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger) filter.
#[derive(Clone, Debug)]
pub struct Schmitt<T, U> {
    /// [low, high] input thresholds.
    thresholds: [T; 2],
    /// [off, on] outputs.
    outputs: [U; 2],
    /// Current internal state.
    state: bool,
}

impl<T, U> Schmitt<T, U> where U: Clone {
    /// Creates a new `Schmitt` filter with given `thresholds` and `outputs`.
    #[inline]
    pub fn new(thresholds: [T; 2], outputs: [U; 2]) -> Self {
        Schmitt { thresholds, outputs, state: false }
    }
}

impl<T, U> Filter<T> for Schmitt<T, U>
where
    T: PartialOrd<T>,
    U: Clone + PartialEq<U>,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state = match self.state {
            false => input > self.thresholds[1],
            true => input >= self.thresholds[0],
        };
        self.outputs[self.state as usize].clone()
    }
}

impl<T, U> Stateful for Schmitt<T, U> {
    #[inline]
    fn reset(&mut self) {
        self.state = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool() {
        let filter = Schmitt::new([5, 10], [false, true]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![false, false, false, false, false, false, true, false, true, true, true, true, true, true, true, false, true, true, true, true]);
    }

    #[test]
    fn fixed_point() {
        let filter = Schmitt::new([5, 10], [0, 1]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn floating_point() {
        let filter = Schmitt::new([5.0, 10.0], [0.0, 1.0]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
