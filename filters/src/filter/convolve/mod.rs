// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use std::ops::{Add, Mul, Div};
use std::fmt;

use arraydeque::{Array, ArrayDeque, Wrapping};
use num_traits::Zero;

use signalo_traits::filter::Filter;
use traits::Stateful;

/// A convolution filter.
#[derive(Clone)]
pub struct Convolve<A>
where
    A: Array,
    A::Item: Copy,
{
    coefficients: A,
    state: ArrayDeque<A, Wrapping>,
}

impl<T, A> Convolve<A>
where
    T: Copy + PartialOrd + Zero,
    A: Array<Item=T>,
{
    /// Creates a new `Convolve` filter with given `coefficients`.
    #[inline]
    pub fn new(coefficients: A) -> Self {
        Convolve { coefficients, state: ArrayDeque::new() }
    }

    /// Returns the filter's coefficients.
    #[inline]
    pub fn coefficients(&self) -> &A {
        &self.coefficients
    }
}

impl<T, A> Convolve<A>
where
    T: Copy + PartialOrd + Zero + Div<T, Output = T>,
    A: Array<Item=T>,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing them.
    #[inline]
    pub fn normalized(mut coefficients: A) -> Self {
        let sum = coefficients.as_slice().iter().fold(T::zero(), |sum, coeff| {
            sum + (*coeff)
        });
        if !sum.is_zero() {
            for coeff in coefficients.as_mut_slice() {
                *coeff = *coeff / sum;
            }
        }
        Convolve { coefficients, state: ArrayDeque::new() }
    }
}

impl<T, A> fmt::Debug for Convolve<A>
where
    T: Copy + fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Convolve")
            .field("coefficients", &self.coefficients)
            .field("state", &self.state)
            .finish()
    }
}

impl<T, A> Filter<T> for Convolve<A>
where
    T: Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>,
    A: Array<Item = T>,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        loop {
            if self.state.push_back(input).is_some() {
                break;
            }
        }

        let state_iter = self.state.iter();
        let coeff_iter = self.coefficients.as_slice().iter().rev();

        let output = state_iter.zip(coeff_iter).fold(T::zero(), |sum, (state, coeff)| {
            sum + ((*state) * (*coeff))
        });

        output
    }
}

impl<T, A> Stateful for Convolve<A>
where
    T: Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>,
    A: Array<Item = T>,
{
    fn reset(&mut self) {
        self.state.clear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0,
            8.0, 8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0,
            88.0, -101.0, 21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0,
            -13.0, 0.0, 0.0, 88.0, -93.0, 13.0, 0.0
        ]
    }

    #[test]
    fn floating_point() {
        // Effectively calculates the derivative:
        let filter = Convolve::new([1.000, -1.000]);
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
