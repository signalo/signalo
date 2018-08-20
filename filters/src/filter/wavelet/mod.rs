// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet filters.

use std::fmt;

use arraydeque::{Array, ArrayDeque, Wrapping};
use num_traits::{Num, Zero};

use signalo_traits::filter::Filter;
use signalo_traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

pub mod daubechies;

/// A wavelet filter's internal state.
#[derive(Clone)]
pub struct State<A>
where
    A: Array,
    A::Item: Copy,
{
    /// The filter's taps (i.e. buffered input).
    pub taps: ArrayDeque<A, Wrapping>,
}

impl<T, A> fmt::Debug for State<A>
where
    T: Copy + fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State").field("taps", &self.taps).finish()
    }
}

/// A wavelet filter.
#[derive(Clone)]
pub struct Wavelet<A>
where
    A: Array,
    A::Item: Copy,
{
    scale: A,
    translate: A,
    state: State<A>,
}

impl<T, A> Wavelet<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    /// Creates a new `Wavelet` filter with given `coefficients`.
    #[inline]
    pub fn new(mut scale: A, mut translate: A) -> Self {
        // // In order to avoid cache inefficiencies due to reverse-order
        // // iteration of the kernel (`coefficients`) during convolution
        // // we reverse the kernel once during construction, instead:
        // scale.reverse();
        // translate.reverse();
        let state = Self::initial_state(());
        Wavelet {
            scale,
            translate,
            state,
        }
    }

    /// Returns the filter's scaling coefficients.
    #[inline]
    pub fn scale(&self) -> &A {
        &self.scale
    }

    /// Returns the filter's translation coefficients.
    #[inline]
    pub fn translate(&self) -> &A {
        &self.translate
    }
}

impl<T, A> fmt::Debug for Wavelet<A>
where
    T: Copy + fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Wavelet")
            .field("scale", &self.scale)
            .field("translate", &self.translate)
            .field("state", &self.state)
            .finish()
    }
}

impl<T, A> Stateful for Wavelet<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    type State = State<A>;
}

unsafe impl<T, A> StatefulUnsafe for Wavelet<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, A> InitialState<()> for Wavelet<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    fn initial_state(_: ()) -> Self::State {
        let taps = ArrayDeque::new();
        State { taps }
    }
}

impl<T, A> Resettable for Wavelet<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, A> Filter<T> for Wavelet<A>
where
    T: Copy + Num,
    A: Array<Item = T>,
{
    type Output = (T, T);

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        // FIXME: padding should not be implementated within filter:
        loop {
            if self.state.taps.push_back(input).is_some() {
                break;
            }
        }

        let state_iter = self.state.taps.iter();
        let scale_coeff_iter = self.scale.as_slice().iter().rev();
        let translate_coeff_iter = self.translate.as_slice().iter().rev();
        let coeff_iter = scale_coeff_iter.zip(translate_coeff_iter);

        let output = state_iter.zip(coeff_iter).fold(
            (T::zero(), T::zero()),
            |(sum, diff), (state, (scale, translate))| {
                let sum = sum + ((*state) * (*scale));
                let diff = diff + ((*state) * (*translate));
                (sum, diff)
            },
        );

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

    fn get_sums() -> Vec<f32> {
        vec![
            0.000, 0.707, 5.657, 6.364, 4.950, 9.192, 16.971, 20.506, 22.627, 17.678, 14.142,
            16.263, 12.728, 18.385, 24.042, 14.849, 11.314, 22.627, 28.284, 19.092, 9.899, 15.556,
            21.213, 17.678, 23.335, 23.335, 85.560, 205.768, 203.647, 89.095, 87.681, 78.489,
            21.920, 27.577, 18.385, 24.042, 29.698, 29.698, 38.891, 29.698, 82.731, 82.731, 26.163,
            31.820, 22.627, 22.627, 84.853, 81.317, 24.749, 33.941,
        ]
    }

    fn get_differences() -> Vec<f32> {
        vec![
            0.000, 0.707, 4.243, -3.536, 2.121, 2.121, 5.657, -2.121, 4.243, -9.192, 5.657, -3.536,
            0.000, 5.657, 0.000, -9.192, 5.657, 5.657, 0.000, -9.192, 0.000, 5.657, 0.000, -3.536,
            9.192, -9.192, 71.418, 48.790, -50.912, -63.640, 62.225, -71.418, 14.849, -9.192,
            0.000, 5.657, 0.000, 0.000, 9.192, -18.385, 71.418, -71.418, 14.849, -9.192, 0.000,
            0.000, 62.225, -65.761, 9.192, 0.000,
        ]
    }

    #[test]
    fn sums() {
        // Effectively calculates the haar transform:
        let filter = Wavelet::new([0.707106781, 0.707106781], [0.707106781, -0.707106781]);
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).0))
            .collect();

        assert_nearly_eq!(output, get_sums(), 0.001);
    }

    #[test]
    fn differences() {
        // Effectively calculates the haar transform:
        let filter = Wavelet::new([0.707106781, 0.707106781], [0.707106781, -0.707106781]);
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).1))
            .collect();

        assert_nearly_eq!(output, get_differences(), 0.001);
    }
}
