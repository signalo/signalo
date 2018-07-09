// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Sub;
use std::mem;

use num_traits::Zero;

use signalo_traits::filter::Filter;

/// A filter that produces the derivative of the signal.
#[derive(Default, Clone, Debug)]
pub struct Differentiate<T> {
    state: Option<T>,
}

impl<T> Filter<T> for Differentiate<T>
where
    T: Copy + Sub<T, Output = T> + Zero,
{
    type Output = <T as Sub<T>>::Output;

    fn filter(&mut self, input: T) -> Self::Output {
        let output = match self.state {
            None => {
                T::zero()
            },
            Some(state) => {
                input - state
            },
        };
        self.state = Some(input);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Differentiate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 1, 6, -5, 3, 3, 8, -13, 16, -13, 8, -5, 0, 8, 0, -13, 8, 8, 0, -13]);
    }

    #[test]
    fn floating_point() {
        let filter = Differentiate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -13.0, 16.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0, 8.0, 8.0, 0.0, -13.0]);
    }
}
