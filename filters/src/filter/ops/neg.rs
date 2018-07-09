// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Neg as StdNeg;

use signalo_traits::filter::Filter;

/// A filter performing a negation operation over a signal.
#[derive(Default, Clone, Debug)]
pub struct Neg;

impl<T> Filter<T> for Neg
where
    T: Copy + StdNeg
{
    type Output = <T as StdNeg>::Output;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        -input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, -1.000, -7.000, -2.000, -5.000, -8.000, -16.000, -3.000, -19.000, -6.000,
            -14.000, -9.000, -9.000, -17.000, -17.000, -4.000, -12.000, -20.000, -20.000, -7.000,
            -7.000, -15.000, -15.000, -10.000, -23.000, -10.000, -111.000, -18.000, -18.000,
            -18.000, -106.000, -5.000, -26.000, -13.000, -13.000, -21.000, -21.000, -21.000,
            -34.000, -8.000, -109.000, -8.000, -29.000, -16.000, -16.000, -16.000, -104.000,
            -11.000, -24.000, -24.000
        ]
    }

    #[test]
    fn neg() {
        let filter = Neg::default();
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
