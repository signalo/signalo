// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use signalo_traits::filter::Filter;

#[derive(Clone, Debug)]
pub struct Threshold<T, U> {
    /// input threshold
    threshold: T,
    /// [off, on] output
    output: [U; 2],
}

impl<T, U> Threshold<T, U> where U: Clone {
    #[inline]
    pub fn new(threshold: T, output: [U; 2]) -> Self {
        Threshold { threshold, output }
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
        self.output[(input >= self.threshold) as usize].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool() {
        let filter = Threshold::new(10, [false, true]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![false, false, false, false, false, false, true, false, true, false, true, false, false, true, true, false, true, true, true, false]);
    }

    #[test]
    fn fixed_point() {
        let filter = Threshold::new(10, [0, 1]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn floating_point() {
        let filter = Threshold::new(10.0, [0.0, 1.0]);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
