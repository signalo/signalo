// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cross-correlation sinks.
//!
//! Computes the normalized cross-correlation coefficient between two signals
//! over a fixed-size sliding window.

use circular_buffer::CircularBuffer;
use num_traits::{cast::NumCast, Num};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Finalize, Reset, Sink,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The cross-correlation accumulator state.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    buffer_x: CircularBuffer<N, T>,
    buffer_y: CircularBuffer<N, T>,
    len: usize,
}

/// A sink that computes the normalized cross-correlation coefficient between two input signals.
///
/// Takes tuples of `(T, T)` and computes the dot product of the last N x and y samples,
/// normalized by N to produce a correlation coefficient at lag 0.
#[derive(Clone, Debug)]
pub struct Correlation<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> ConfigTrait for Correlation<T, N> {
    type Config = ();
}

impl<T, const N: usize> StateTrait for Correlation<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for Correlation<T, N>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(_config: Self::Config) -> Self::Output {
        Self {
            state: State {
                buffer_x: CircularBuffer::new(),
                buffer_y: CircularBuffer::new(),
                len: 0,
            },
        }
    }
}

impl<T, const N: usize> Default for Correlation<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(())
    }
}

impl<T, const N: usize> ConfigRef for Correlation<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &()
    }
}

impl<T, const N: usize> ConfigClone for Correlation<T, N> {
    fn config(&self) -> Self::Config {}
}

impl<T, const N: usize> StateMut for Correlation<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Correlation<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Correlation<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { state: guts }
    }
}

impl<T, const N: usize> IntoGuts for Correlation<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Correlation<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Correlation<T, N> where Self: Reset {}

impl<T, const N: usize> Sink<(T, T)> for Correlation<T, N>
where
    T: Clone + Num,
{
    #[inline]
    fn sink(&mut self, input: (T, T)) {
        let (x, y) = input;
        self.state.buffer_x.push_back(x);
        self.state.buffer_y.push_back(y);
        if self.state.len < N {
            self.state.len += 1;
        }
    }
}

impl<T, const N: usize> Filter<(T, T)> for Correlation<T, N>
where
    T: Clone + Num + NumCast,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: (T, T)) -> Self::Output {
        self.sink(input);
        let dot_product = self
            .state
            .buffer_x
            .iter()
            .zip(self.state.buffer_y.iter())
            .fold(T::zero(), |sum, (x, y)| sum + (x.clone() * y.clone()));

        let count = NumCast::from(self.state.len).unwrap_or(T::zero());

        if count.is_zero() {
            T::zero()
        } else {
            dot_product / count
        }
    }
}

impl<T, const N: usize> Finalize for Correlation<T, N>
where
    T: Clone + Num + NumCast,
{
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        if self.state.len == 0 {
            return None;
        }
        let dot_product = self
            .state
            .buffer_x
            .iter()
            .zip(self.state.buffer_y.iter())
            .fold(T::zero(), |sum, (x, y)| sum + (x.clone() * y.clone()));

        let count = NumCast::from(self.state.len).unwrap_or(T::zero());

        if count.is_zero() {
            Some(T::zero())
        } else {
            Some(dot_product / count)
        }
    }
}

#[cfg(test)]
mod tests {
    use nearly_eq::assert_nearly_eq;

    use super::*;

    #[test]
    fn autocorrelation_constant_signal() {
        // Input: [(1.0, 1.0)] repeated, N=1
        // Expected: dot_product = 1.0 * 1.0 = 1.0, divided by 1 = 1.0
        let mut sink: Correlation<f32, 1> = Correlation::default();
        sink.sink((1.0, 1.0));
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(1.0));
    }

    #[test]
    fn uncorrelated_signals() {
        // Input: [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)] with N=3
        // dot_product = 1.0*0.0 + (-1.0)*0.0 + 1.0*0.0 = 0.0
        // correlation = 0.0 / 3 = 0.0
        let mut sink: Correlation<f32, 3> = Correlation::default();
        sink.sink((1.0, 0.0));
        sink.sink((-1.0, 0.0));
        sink.sink((1.0, 0.0));
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(0.0), 0.0001);
    }

    #[test]
    fn perfectly_correlated() {
        // Input: [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)] with N=3
        // dot_product = 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
        // correlation = 29 / 3 ≈ 9.667
        let mut sink: Correlation<f32, 3> = Correlation::default();
        sink.sink((2.0, 2.0));
        sink.sink((3.0, 3.0));
        sink.sink((4.0, 4.0));
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(29.0 / 3.0), 0.0001);
    }

    #[test]
    fn filter_interface() {
        // Test Filter interface returns same result as Finalize
        // Input: [(1.0, 1.0), (2.0, 2.0)] with N=2
        let mut sink: Correlation<f32, 2> = Correlation::default();
        let out1 = sink.filter((1.0, 1.0));
        let out2 = sink.filter((2.0, 2.0));

        // After first input: dot = 1*1 = 1, count = 1, out = 1.0
        assert_nearly_eq!(out1, 1.0);
        // After second input: dot = 1*1 + 2*2 = 5, count = 2, out = 2.5
        assert_nearly_eq!(out2, 2.5);

        let finalize_result = sink.finalize();
        assert_nearly_eq!(finalize_result, Some(2.5), 0.0001);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_nan_propagation() {
        let mut sink: Correlation<f32, 4> = Correlation::default();
        let result = sink.filter((f32::NAN, 1.0));
        assert!(result.is_nan());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_large_values() {
        let mut sink: Correlation<f32, 2> = Correlation::default();
        let result = sink.filter((1e10, 1e10));
        assert!(result.is_finite());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_inf_propagation() {
        let mut sink: Correlation<f32, 4> = Correlation::default();
        let result = sink.filter((f32::INFINITY, 1.0));
        assert!(result.is_infinite());
    }

    #[test]
    fn test_reset() {
        let mut sink: Correlation<f32, 2> = Correlation::default();
        sink.sink((1.0, 1.0));
        sink.sink((2.0, 2.0));
        let sink = sink.reset();
        let mut sink = sink;
        assert_eq!(sink.finalize(), None);
    }

    #[test]
    fn test_n1_window() {
        let mut sink: Correlation<f32, 1> = Correlation::default();
        let out1 = sink.filter((3.0, 4.0));
        assert_eq!(out1, 12.0);
        let out2 = sink.filter((5.0, 6.0));
        assert_eq!(out2, 30.0);
    }

    #[test]
    fn test_integer_type() {
        let mut sink: Correlation<i32, 2> = Correlation::default();
        let out1 = sink.filter((1, 2));
        let out2 = sink.filter((3, 4));
        // dot = 1*2 + 3*4 = 14, count = 2, result = 7
        assert_eq!(out2, 7);
    }

    #[test]
    fn test_state_mut() {
        let mut sink: Correlation<f32, 4> = Correlation::default();
        unsafe {
            let state = sink.state_mut();
            state.buffer_x.push_back(1.0);
            state.buffer_y.push_back(1.0);
            state.len = 1;
        }
        let result = sink.filter((2.0, 2.0));
        assert_eq!(result, (1.0 * 1.0 + 2.0 * 2.0) / 2.0);
    }

    #[test]
    fn empty_sink() {
        // Empty sink should return None
        let sink: Correlation<f32, 4> = Correlation::default();
        let result = sink.finalize();
        assert_eq!(result, None);
    }

    #[test]
    fn test() {
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let mut sink: Correlation<f32, 20> = Correlation::default();
        for value in input {
            sink.sink((value, value));
        }
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(137.5));
    }
}
