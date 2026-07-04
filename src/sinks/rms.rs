// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Root Mean Square (RMS) sinks.
//!
//! Computes the mean square value over a fixed-size sliding window. Returns mean square value.
//! Apply `sqrt()` (requires `std`) for true RMS.
//!
//! Computes the mean square value over a fixed-size sliding window.
//! Returns mean square value. Apply `sqrt()` (requires `std`) for true RMS.

use circular_buffer::FixedCircularBuffer;
use num_traits::{cast::NumCast, Num};

use crate::storage::RingBuffer;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Finalize, Reset, Sink,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The RMS accumulator state.
///
/// Holds the ring buffer `R` together with the running sum-of-squares and the
/// current fill count. The fill count starts at zero and increases up to the
/// buffer's capacity, tracking how many samples have been ingested so far.
#[derive(Clone, Debug)]
pub struct State<T, R> {
    /// The ring buffer holding the most recent samples.
    pub buffer: R,
    /// The running sum of squared samples.
    pub sum_sq: T,
    /// The number of samples currently in the buffer (≤ `buffer.capacity()`).
    pub len: usize,
}

/// A sink that computes the root mean square over a sliding window.
///
/// Returns the **mean square** value (i.e. `Σxᵢ² / N`). Apply `sqrt()` to
/// obtain the true RMS. The window size is determined by the capacity of the
/// ring buffer `R`.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`RmsArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`RmsVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone, Debug)]
pub struct Rms<T, R> {
    state: State<T, R>,
}

/// A RMS sink backed by a const-generic [`FixedCircularBuffer`].
///
/// This alias is the `no_std`-friendly, zero-allocation form. Both `T` and
/// the window size `N` are fixed at compile time.
pub type RmsArray<T, const N: usize> = Rms<T, FixedCircularBuffer<T, N>>;

/// A RMS sink backed by a heap-allocated [`HeapCircularBuffer`].
///
/// Requires the `alloc` feature. Use [`Rms::from_parts`] to construct this
/// variant, since the buffer capacity must be known at runtime.
#[cfg(feature = "alloc")]
pub type RmsVec<T> = Rms<T, HeapCircularBuffer<T>>;

impl<T, R> Rms<T, R>
where
    T: Num,
    R: RingBuffer<T>,
{
    /// Creates an [`Rms`] sink from an already-constructed ring buffer.
    ///
    /// Use this constructor when the buffer is not `Default`-constructible,
    /// e.g. for [`RmsVec`] whose capacity must be known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.capacity()` is zero.
    pub fn from_parts(buffer: R) -> Self {
        assert!(buffer.capacity() > 0, "Rms: window size must be > 0");
        Self {
            state: State {
                buffer,
                sum_sq: T::zero(),
                len: 0,
            },
        }
    }
}

impl<T, R> ConfigTrait for Rms<T, R> {
    type Config = ();
}

impl<T, R> StateTrait for Rms<T, R> {
    type State = State<T, R>;
}

impl<T, const N: usize> WithConfig for RmsArray<T, N>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(_config: Self::Config) -> Self::Output {
        Self {
            state: State {
                buffer: FixedCircularBuffer::new(),
                sum_sq: T::zero(),
                len: 0,
            },
        }
    }
}

impl<T, const N: usize> Default for RmsArray<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(())
    }
}

impl<T, R> ConfigRef for Rms<T, R> {
    fn config_ref(&self) -> &Self::Config {
        &()
    }
}

impl<T, R> ConfigClone for Rms<T, R> {
    fn config(&self) -> Self::Config {}
}

impl<T, R> StateMut for Rms<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Rms<T, R> {
    type Guts = State<T, R>;
}

impl<T, R> FromGuts for Rms<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { state: guts }
    }
}

impl<T, R> IntoGuts for Rms<T, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for RmsArray<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for RmsArray<T, N> where Self: Reset {}

impl<T, R> Sink<T> for Rms<T, R>
where
    T: Clone + Num,
    R: RingBuffer<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        if let Some(evicted) = self.state.buffer.push_back(input.clone()) {
            let evicted_sq = evicted.clone() * evicted;
            self.state.sum_sq = self.state.sum_sq.clone() - evicted_sq;
        }
        let input_sq = input.clone() * input;
        self.state.sum_sq = self.state.sum_sq.clone() + input_sq;
        let cap = self.state.buffer.capacity();
        if self.state.len < cap {
            self.state.len += 1;
        }
    }
}

impl<T, R> Filter<T> for Rms<T, R>
where
    T: Clone + Num + NumCast,
    R: RingBuffer<T>,
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

impl<T, R> Finalize for Rms<T, R>
where
    T: Clone + Num + NumCast,
    R: RingBuffer<T>,
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
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn empty() {
        let sink: RmsArray<f32, 4> = RmsArray::default();
        let result = sink.finalize();
        assert_eq!(result, None);
    }

    #[test]
    fn all_ones() {
        // Input: [1.0, 1.0, 1.0, 1.0]
        // Mean of squares: (1 + 1 + 1 + 1) / 4 = 1.0
        let mut sink: RmsArray<f32, 4> = RmsArray::default();
        sink.sink(1.0);
        sink.sink(1.0);
        sink.sink(1.0);
        sink.sink(1.0);
        let result = sink.finalize();
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn three_four_window_two() {
        // Input: [3.0, 4.0] with N=2
        // Mean of squares: (9 + 16) / 2 = 25 / 2 = 12.5
        let mut sink: RmsArray<f32, 2> = RmsArray::default();
        sink.sink(3.0);
        sink.sink(4.0);
        let result = sink.finalize();
        assert_eq!(result, Some(12.5));
    }

    #[test]
    fn test_nan_propagation() {
        let mut sink: RmsArray<f32, 4> = RmsArray::default();
        let result = sink.filter(f32::NAN);
        assert!(result.is_nan());
    }

    #[test]
    fn test_large_values() {
        let mut sink: RmsArray<f32, 2> = RmsArray::default();
        let result = sink.filter(1e10);
        assert!(result.is_finite());
        let result = sink.filter(1e10);
        assert!(result.is_finite());
    }

    #[test]
    fn test_inf_propagation() {
        let mut sink: RmsArray<f32, 4> = RmsArray::default();
        let result = sink.filter(f32::INFINITY);
        assert!(result.is_infinite());
        let result = sink.filter(f32::NEG_INFINITY);
        assert!(result.is_infinite());
    }

    #[test]
    fn test_reset() {
        let mut sink: RmsArray<f32, 2> = RmsArray::default();
        sink.sink(1.0);
        sink.sink(2.0);
        sink.sink(3.0);
        let sink = sink.reset();
        let sink = sink;
        assert_eq!(sink.finalize(), None);
    }

    #[test]
    fn test_n1_window() {
        let mut sink: RmsArray<f32, 1> = RmsArray::default();
        let result = sink.filter(5.0);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_integer_type() {
        let mut sink: RmsArray<i32, 3> = RmsArray::default();
        sink.sink(1);
        sink.sink(2);
        sink.sink(3);
        let result = sink.finalize();
        // mean_sq = (1 + 4 + 9) / 3 = 14 / 3 = 4 (integer division)
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_state_mut() {
        let mut sink: RmsArray<f32, 4> = RmsArray::default();
        let state = sink.state_mut();
        state.sum_sq = 10.0;
        state.len = 1;
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
        let mut sink: RmsArray<f32, 2> = RmsArray::default();
        let out1 = sink.filter(1.0);
        let out2 = sink.filter(2.0);
        let out3 = sink.filter(3.0);
        let out4 = sink.filter(4.0);

        assert_abs_diff_eq!(out1, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out2, 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(out3, 6.5, epsilon = 1e-6);
        assert_abs_diff_eq!(out4, 12.5, epsilon = 1e-6);
    }

    #[test]
    fn test() {
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let mut sink: RmsArray<f32, 20> = RmsArray::default();
        for value in input {
            sink.sink(value);
        }
        let result = sink.finalize();
        assert_eq!(result, Some(137.5));
    }
}
