// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Arithmetic square filters.

use core::ops::Mul;

use signalo_traits::Filter;

/// A filter performing a squaring operation over a signal.
#[derive(Default, Clone, Debug)]
pub struct Square;

impl<T> Filter<T> for Square
where
    T: Clone + Mul<T>,
{
    type Output = <T as Mul<T>>::Output;

    fn filter(&mut self, input: T) -> Self::Output {
        input.clone() * input
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
            0.000, 1.000, 49.000, 4.000, 25.000, 64.000, 256.000, 9.000, 361.000, 36.000, 196.000,
            81.000, 81.000, 289.000, 289.000, 16.000, 144.000, 400.000, 400.000, 49.000, 49.000,
            225.000, 225.000, 100.000, 529.000, 100.000, 12321.000, 324.000, 324.000, 324.000,
            11236.000, 25.000, 676.000, 169.000, 169.000, 441.000, 441.000, 441.000, 1156.000,
            64.000, 11881.000, 64.000, 841.000, 256.000, 256.000, 256.000, 10816.000, 121.000,
            576.000, 576.000,
        ]
    }

    #[test]
    fn test() {
        let filter = Square::default();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
