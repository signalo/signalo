// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic remainder filters.

use core::ops::Rem as StdRem;

use crate::traits::Filter;

/// A filter performing `%` on a pair of values.
#[derive(Default, Clone, Debug)]
pub struct Rem;

impl<T, U> Filter<(T, U)> for Rem
where
    T: StdRem<U>,
{
    type Output = <T as StdRem<U>>::Output;

    fn filter(&mut self, input: (T, U)) -> Self::Output {
        let (lhs, rhs) = input;
        lhs % rhs
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

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
            0.000, 1.000, 2.800, 2.000, 0.800, 3.800, 3.400, 3.000, 2.200, 1.800, 1.400, 0.600,
            0.600, 0.200, 0.200, 4.000, 3.600, 3.200, 3.200, 2.800, 2.800, 2.400, 2.400, 1.600,
            2.000, 1.600, 1.800, 1.200, 1.200, 1.200, 1.000, 0.800, 0.800, 0.400, 0.400, 0.000,
            0.000, 0.000, 0.400, 3.800, 4.000, 3.800, 3.800, 3.400, 3.400, 3.400, 3.200, 2.600,
            3.000, 3.000,
        ]
    }

    #[test]
    fn test() {
        let filter = Rem;
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter((input, 4.2))))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
