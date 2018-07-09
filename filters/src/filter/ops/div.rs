// This Source Code Form is divject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Div as StdDiv;

use signalo_traits::filter::Filter;

/// A filter performing `/` on a pair of values.
#[derive(Default, Clone, Debug)]
pub struct Div;

impl<T, U> Filter<(T, U)> for Div
where
    T: Copy + StdDiv<U>,
{
    type Output = <T as StdDiv<U>>::Output;

    #[inline]
    fn filter(&mut self, input: (T, U)) -> Self::Output {
        let (lhs, rhs) = input;
        lhs / rhs
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
            0.000, 0.238, 1.667, 0.476, 1.190, 1.905, 3.810, 0.714, 4.524, 1.429, 3.333, 2.143,
            2.143, 4.048, 4.048, 0.952, 2.857, 4.762, 4.762, 1.667, 1.667, 3.571, 3.571, 2.381,
            5.476, 2.381, 26.429, 4.286, 4.286, 4.286, 25.238, 1.190, 6.190, 3.095, 3.095, 5.000,
            5.000, 5.000, 8.095, 1.905, 25.952, 1.905, 6.905, 3.810, 3.810, 3.810, 24.762, 2.619,
            5.714, 5.714
        ]
    }

    #[test]
    fn div() {
        let filter = Div::default();
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter((input, 4.2)))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
