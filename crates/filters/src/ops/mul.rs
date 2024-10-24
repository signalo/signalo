// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic multiplication filters.

use core::ops::Mul as StdMul;

use signalo_traits::Filter;

/// A filter performing `*` on a pair of values.
#[derive(Default, Clone, Debug)]
pub struct Mul;

impl<T, U> Filter<(T, U)> for Mul
where
    T: StdMul<U>,
{
    type Output = <T as StdMul<U>>::Output;

    fn filter(&mut self, input: (T, U)) -> Self::Output {
        let (lhs, rhs) = input;
        lhs * rhs
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
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 4.200, 29.400, 8.400, 21.000, 33.600, 67.200, 12.600, 79.800, 25.200, 58.800,
            37.800, 37.800, 71.400, 71.400, 16.800, 50.400, 84.000, 84.000, 29.400, 29.400, 63.000,
            63.000, 42.000, 96.600, 42.000, 466.200, 75.600, 75.600, 75.600, 445.200, 21.000,
            109.200, 54.600, 54.600, 88.200, 88.200, 88.200, 142.800, 33.600, 457.800, 33.600,
            121.800, 67.200, 67.200, 67.200, 436.800, 46.200, 100.800, 100.800,
        ]
    }

    #[test]
    fn test() {
        let filter = Mul::default();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter((input, 4.2))))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
