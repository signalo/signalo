// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialEq;

use num_traits::Zero;

use signalo_traits::filter::Filter;
use traits::Stateful;

#[derive(Clone, Debug)]
pub struct Debounce<T, U> {
    /// Threshold of how long input must remain same to be accepted
    threshold: usize,
    /// [off, on] output
    output: [U; 2],
    /// Value to debounce
    predicate: T,
    /// Counter of how long input was the same
    counter: usize,
}

impl<T, U> Debounce<T, U>
where
    T: Copy + Zero
{
    #[inline]
    pub fn new(threshold: usize, predicate: T, output: [U; 2]) -> Self {
        Debounce { threshold, output, predicate, counter: 0 }
    }
}

impl<T, U> Filter<T> for Debounce<T, U>
where
    T: Copy + PartialEq<T>,
    U: Copy,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        if input == self.predicate {
            self.counter = (self.counter + 1).min(self.threshold);
        } else {
            self.counter = 0;
        }
        self.output[(self.counter >= self.threshold) as usize]
    }
}

impl<T, U> Stateful for Debounce<T, U> {
    #[inline]
    fn reset(&mut self) {
        self.counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Debounce::new(3, 1, [0, 1]);
        let input = vec![0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
