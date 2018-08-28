// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use std::fmt;

use arraydeque::{ArrayDeque, Wrapping};
use generic_array::{ArrayLength, GenericArray};

use num_traits::Num;

use signalo_traits::filter::Filter;

use traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

pub mod savitzky_golay;

/// A convolution filter's internal state.
#[derive(Clone)]
pub struct State<T, N>
where
    N: ArrayLength<T>,
{
    /// The filter's taps (i.e. buffered input).
    pub taps: ArrayDeque<GenericArray<T, N>, Wrapping>,
}

impl<T, N> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
    N: ArrayLength<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State").field("taps", &self.taps).finish()
    }
}

/// A convolution filter.
#[derive(Clone)]
pub struct Convolve<T, N>
where
    N: ArrayLength<T>,
{
    coefficients: GenericArray<T, N>,
    state: State<T, N>,
}

impl<T, N> Convolve<T, N>
where
    N: ArrayLength<T>,
{
    /// Creates a new `Convolve` filter with given `coefficients`.
    #[inline]
    pub fn new(coefficients: GenericArray<T, N>) -> Self {
        let state = Self::initial_state(());
        Convolve {
            coefficients,
            state,
        }
    }

    /// Returns the filter's coefficients.
    #[inline]
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
}

impl<T, N> Convolve<T, N>
where
    T: Clone + PartialOrd + Num,
    N: ArrayLength<T>,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing them.
    #[inline]
    pub fn normalized(mut coefficients: GenericArray<T, N>) -> Self {
        let sum = coefficients
            .as_slice()
            .iter()
            .fold(T::zero(), |sum, coeff| sum + coeff.clone());
        if !sum.is_zero() {
            for coeff in coefficients.as_mut_slice() {
                *coeff = coeff.clone() / sum.clone();
            }
        }
        let state = Self::initial_state(());
        Convolve {
            coefficients,
            state,
        }
    }
}

impl<T, N> Stateful for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    type State = State<T, N>;
}

unsafe impl<T, N> StatefulUnsafe for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, N> InitialState<()> for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    fn initial_state(_: ()) -> Self::State {
        let taps = ArrayDeque::new();
        State { taps }
    }
}

impl<T, N> Resettable for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, N> Filter<T> for Convolve<T, N>
where
    T: Clone + Num,
    N: ArrayLength<T>,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        loop {
            if self.state.taps.push_back(input.clone()).is_some() {
                break;
            }
        }

        let state_iter = self.state.taps.iter();
        let coeff_iter = self.coefficients.as_slice().iter().rev();

        let output = state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + (state.clone() * coeff.clone())
            });

        output
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
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0,
            8.0, 8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0,
            88.0, -101.0, 21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0, -13.0,
            0.0, 0.0, 88.0, -93.0, 13.0, 0.0,
        ]
    }

    #[test]
    fn test() {
        // Effectively calculates the derivative:
        let filter = Convolve::new(arr![f32; 1.000, -1.000]);
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
