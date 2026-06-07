// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Root Mean Square (RMS) sinks.
//!
//! Computes the mean square value over a fixed-size sliding window.
//! Returns mean square value. Apply `sqrt()` (requires `std`) for true RMS.

use circular_buffer::CircularBuffer;
use num_traits::{cast::NumCast, Num};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Finalize, Reset, Sink,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The RMS accumulator state.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    buffer: CircularBuffer<N, T>,
    sum_sq: T,
    len: usize,
}

/// A sink that computes the root mean square over a sliding window of N samples.
///
/// Returns the mean square value. Apply `sqrt()` for the actual RMS value.
#[derive(Clone, Debug)]
pub struct Rms<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> ConfigTrait for Rms<T, N> {
    type Config = ();
}

impl<T, const N: usize> StateTrait for Rms<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for Rms<T, N>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(_config: Self::Config) -> Self::Output {
        Self {
            state: State {
                buffer: CircularBuffer::new(),
                sum_sq: T::zero(),
                len: 0,
            },
        }
    }
}

impl<T, const N: usize> Default for Rms<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(())
    }
}

impl<T, const N: usize> ConfigRef for Rms<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &()
    }
}

impl<T, const N: usize> ConfigClone for Rms<T, N> {
    fn config(&self) -> Self::Config {}
}

impl<T, const N: usize> StateMut for Rms<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Rms<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Rms<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { state: guts }
    }
}

impl<T, const N: usize> IntoGuts for Rms<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Rms<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Rms<T, N> where Self: Reset {}

impl<T, const N: usize> Sink<T> for Rms<T, N>
where
    T: Clone + Num,
{
    #[inline]
    fn sink(&mut self, input: T) {
        if let Some(evicted) = self.state.buffer.push_back(input.clone()) {
            let evicted_sq = evicted.clone() * evicted;
            self.state.sum_sq = self.state.sum_sq.clone() - evicted_sq;
        }
        let input_sq = input.clone() * input;
        self.state.sum_sq = self.state.sum_sq.clone() + input_sq;
        if self.state.len < N {
            self.state.len += 1;
        }
    }
}

impl<T, const N: usize> Filter<T> for Rms<T, N>
where
    T: Clone + Num + NumCast,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        self.sink(input);
        let count = NumCast::from(self.state.len).unwrap_or(T::zero());
        if count.is_zero() {
            T::zero()
        } else {
            self.state.sum_sq.clone() / count
        }
    }
}

impl<T, const N: usize> Finalize for Rms<T, N>
where
    T: Clone + Num + NumCast,
{
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        if self.state.len == 0 {
            return None;
        }
        let count = NumCast::from(self.state.len).unwrap_or(T::zero());
        Some(self.state.sum_sq / count)
    }
}

#[cfg(test)]
mod tests {
    use nearly_eq::assert_nearly_eq;

    use super::*;

    #[test]
    fn empty() {
        let sink: Rms<f32, 4> = Rms::default();
        let result = sink.finalize();
        assert_eq!(result, None);
    }

    #[test]
    fn all_ones() {
        // Input: [1.0, 1.0, 1.0, 1.0]
        // Mean of squares: (1 + 1 + 1 + 1) / 4 = 1.0
        let mut sink: Rms<f32, 4> = Rms::default();
        sink.sink(1.0);
        sink.sink(1.0);
        sink.sink(1.0);
        sink.sink(1.0);
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(1.0));
    }

    #[test]
    fn three_four_window_two() {
        // Input: [3.0, 4.0] with N=2
        // Mean of squares: (9 + 16) / 2 = 25 / 2 = 12.5
        let mut sink: Rms<f32, 2> = Rms::default();
        sink.sink(3.0);
        sink.sink(4.0);
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(12.5));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_nan_propagation() {
        let mut sink: Rms<f32, 4> = Rms::default();
        let result = sink.filter(f32::NAN);
        assert!(result.is_nan());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_large_values() {
        let mut sink: Rms<f32, 2> = Rms::default();
        let result = sink.filter(1e10);
        assert!(result.is_finite());
        let result = sink.filter(1e10);
        assert!(result.is_finite());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_inf_propagation() {
        let mut sink: Rms<f32, 4> = Rms::default();
        let result = sink.filter(f32::INFINITY);
        assert!(result.is_infinite());
        let result = sink.filter(f32::NEG_INFINITY);
        assert!(result.is_infinite());
    }

    #[test]
    fn test_reset() {
        let mut sink: Rms<f32, 2> = Rms::default();
        sink.sink(1.0);
        sink.sink(2.0);
        sink.sink(3.0);
        let sink = sink.reset();
        let mut sink = sink;
        assert_eq!(sink.finalize(), None);
    }

    #[test]
    fn test_n1_window() {
        let mut sink: Rms<f32, 1> = Rms::default();
        let result = sink.filter(5.0);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_integer_type() {
        let mut sink: Rms<i32, 3> = Rms::default();
        sink.sink(1);
        sink.sink(2);
        sink.sink(3);
        let result = sink.finalize();
        // mean_sq = (1 + 4 + 9) / 3 = 14 / 3 = 4 (integer division)
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_state_mut() {
        let mut sink: Rms<f32, 4> = Rms::default();
        unsafe {
            let state = sink.state_mut();
            state.sum_sq = 10.0;
            state.len = 1;
        }
        let result = sink.filter(5.0);
        assert_eq!(result, (10.0 + 25.0) / 2.0);
    }

    #[test]
    fn sliding_window() {
        // Input: [1.0, 2.0, 3.0, 4.0] with N=2
        // After 1.0: [1.0] -> mean_sq = 1.0 / 1 = 1.0
        // After 2.0: [1.0, 2.0] -> mean_sq = (1 + 4) / 2 = 2.5
        // After 3.0: [2.0, 3.0] (1.0 removed) -> mean_sq = (4 + 9) / 2 = 6.5
        // After 4.0: [3.0, 4.0] (2.0 removed) -> mean_sq = (9 + 16) / 2 = 12.5
        let mut sink: Rms<f32, 2> = Rms::default();
        let out1 = sink.filter(1.0);
        let out2 = sink.filter(2.0);
        let out3 = sink.filter(3.0);
        let out4 = sink.filter(4.0);

        assert_nearly_eq!(out1, 1.0);
        assert_nearly_eq!(out2, 2.5);
        assert_nearly_eq!(out3, 6.5);
        assert_nearly_eq!(out4, 12.5);
    }

    #[test]
    fn test() {
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let mut sink: Rms<f32, 20> = Rms::default();
        for value in input {
            sink.sink(value);
        }
        let result = sink.finalize();
        assert_nearly_eq!(result, Some(137.5));
    }
}
