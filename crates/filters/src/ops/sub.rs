// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic subtraction filters.

use core::ops::Sub as StdSub;

use signalo_traits::Filter;

/// A filter performing `-` on a pair of values.
#[derive(Default, Clone, Debug)]
pub struct Sub;

impl<T, U> Filter<(T, U)> for Sub
where
    T: StdSub<U>,
{
    type Output = <T as StdSub<U>>::Output;

    fn filter(&mut self, input: (T, U)) -> Self::Output {
        let (lhs, rhs) = input;
        lhs - rhs
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
            -4.200, -3.200, 2.800, -2.200, 0.800, 3.800, 11.800, -1.200, 14.800, 1.800, 9.800,
            4.800, 4.800, 12.800, 12.800, -0.200, 7.800, 15.800, 15.800, 2.800, 2.800, 10.800,
            10.800, 5.800, 18.800, 5.800, 106.800, 13.800, 13.800, 13.800, 101.800, 0.800, 21.800,
            8.800, 8.800, 16.800, 16.800, 16.800, 29.800, 3.800, 104.800, 3.800, 24.800, 11.800,
            11.800, 11.800, 99.800, 6.800, 19.800, 19.800,
        ]
    }

    #[test]
    fn test() {
        let filter = Sub::default();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter((input, 4.2))))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
