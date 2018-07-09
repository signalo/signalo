// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Add as StdAdd;

use signalo_traits::filter::Filter;

/// A filter performing `+` on a pair of values.
#[derive(Default, Clone, Debug)]
pub struct Add;

impl<T, U> Filter<(T, U)> for Add
where
    T: Copy + StdAdd<U>,
{
    type Output = <T as StdAdd<U>>::Output;

    #[inline]
    fn filter(&mut self, input: (T, U)) -> Self::Output {
        let (lhs, rhs) = input;
        lhs + rhs
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
            4.200, 5.200, 11.200, 6.200, 9.200, 12.200, 20.200, 7.200, 23.200, 10.200, 18.200,
            13.200, 13.200, 21.200, 21.200, 8.200, 16.200, 24.200, 24.200, 11.200, 11.200,
            19.200, 19.200, 14.200, 27.200, 14.200, 115.200, 22.200, 22.200, 22.200, 110.200,
            9.200, 30.200, 17.200, 17.200, 25.200, 25.200, 25.200, 38.200, 12.200, 113.200,
            12.200, 33.200, 20.200, 20.200, 20.200, 108.200, 15.200, 28.200, 28.200
        ]
    }

    #[test]
    fn div() {
        let filter = Add::default();
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter((input, 4.2)))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
