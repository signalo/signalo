// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use signalo_traits::filter::Filter;

/// A threshold filter.
#[derive(Clone, Debug)]
pub struct Threshold<T, U> {
    /// input threshold.
    threshold: T,
    /// [off, on] outputs.
    outputs: [U; 2],
}

impl<T, U> Threshold<T, U>
where
    U: Clone
{
    /// Creates a new `Threshold` filter with given `threshold` and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(threshold: T, outputs: [U; 2]) -> Self {
        Threshold { threshold, outputs }
    }
}

impl<T, U> Filter<T> for Threshold<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let index = (input >= self.threshold) as usize;
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Threshold::new(10, u8::classes());
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]);
    }
}
