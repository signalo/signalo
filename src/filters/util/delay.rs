// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Delay filters.

use core::fmt;
use core::marker::PhantomData;

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::Num;

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

use crate::storage::RingBuffer;
use crate::traits::Filter;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Reset, State as StateTrait, StateMut,
};

/// The delay filter's state.
#[derive(Clone)]
pub struct State<R> {
    /// The current taps buffer.
    pub taps: R,
}

impl<R> fmt::Debug for State<R>
where
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State").field("taps", &self.taps).finish()
    }
}

/// A delay filter that delays the input signal by a fixed number of samples.
///
/// The filter uses a ring-buffer `R` as its tap storage. On each call to
/// [`Filter::filter`], the newest sample is pushed into the buffer and the
/// oldest sample is returned. Until the buffer is full the filter re-inserts
/// the current input in a loop until the buffer wraps and an evicted element
/// is available.
///
/// # Complexity
///
/// - **Time per sample:** O(N) only on the first N calls (buffer fill); O(1) thereafter;
///   one `push_back` to the circular buffer.
/// - **Space:** O(N); circular buffer of N delayed samples.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`DelayArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`DelayVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone)]
pub struct Delay<T, R> {
    state: State<R>,
    _pd: PhantomData<T>,
}

/// A delay filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The tap
/// ring-buffer lives entirely on the stack.
pub type DelayArray<T, const N: usize> = Delay<T, FixedCircularBuffer<T, N>>;

/// A delay filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`Delay::from_parts`] to construct this
/// variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type DelayVec<T> = Delay<T, HeapCircularBuffer<T>>;

/// A delay filter that borrows a [`CircularBuffer`] tap buffer.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. Construct via [`Delay::from_parts`], passing
/// a `&mut CircularBuffer<T>` for the tap buffer.
pub type DelayRefMut<'a, T> = Delay<T, &'a mut CircularBuffer<T>>;

impl<T, R> Delay<T, R>
where
    R: RingBuffer<T>,
{
    /// Creates a [`Delay`] filter from an already-constructed `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`DelayVec`] whose capacity must be known at runtime.
    ///
    /// The `taps` buffer is taken as-is with their current contents.
    /// If it contains pre-existing samples, those values are treated as past
    /// input history and will be output before any newly filtered samples.
    ///
    /// # Expected storage state
    ///
    /// For an idiomatic cold-start (where the first `N` outputs are re-inserted
    /// input), pass an empty buffer.
    pub fn from_parts(taps: R) -> Self {
        Self {
            state: State { taps },
            _pd: PhantomData,
        }
    }
}

impl<T, const N: usize> Default for DelayArray<T, N>
where
    T: Num,
{
    fn default() -> Self {
        let state = {
            let taps = FixedCircularBuffer::default();
            State { taps }
        };
        Self {
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, R> fmt::Debug for Delay<T, R>
where
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Delay").field("state", &self.state).finish()
    }
}

impl<T, R> StateTrait for Delay<T, R> {
    type State = State<R>;
}

impl<T, R> StateMut for Delay<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Delay<T, R> {
    type Guts = State<R>;
}

impl<T, R> FromGuts for Delay<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self {
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, R> IntoGuts for Delay<T, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for DelayArray<T, N>
where
    T: Num,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

impl<T, R> Filter<T> for Delay<T, R>
where
    T: Clone + Num,
    R: RingBuffer<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // Note: push_back may return None until the circular buffer is full.
        // Once full, it returns Some with the evicted old value.
        // This loop is guaranteed to terminate when the buffer is full.
        loop {
            if let Some(delayed) = self.state.taps.push_back(input.clone()) {
                return delayed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[allow(clippy::wildcard_imports)]
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
            0.0, 0.0, 0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0,
            17.0, 4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0,
            18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0,
            16.0, 16.0, 16.0, 104.0, 11.0,
        ]
    }

    #[test]
    fn test() {
        let filter: DelayArray<f32, 2> = DelayArray::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn test_state_mut() {
        let mut filter: DelayArray<i32, 2> = DelayArray::default();
        filter.filter(1);
        filter.filter(2);

        let state = filter.state_mut();
        // Verify we can access the state
        assert!(!state.taps.is_empty());

        // Continue filtering
        let output = filter.filter(3);
        assert_eq!(output, 1);
    }

    #[test]
    fn test_from_into_guts() {
        let filter: DelayArray<i32, 2> = DelayArray::default();
        let guts = filter.into_guts();
        let _new_filter: DelayArray<i32, 2> = FromGuts::from_guts(guts);
    }

    #[test]
    fn test_reset() {
        let mut filter: DelayArray<i32, 2> = DelayArray::default();
        filter.filter(10);
        filter.filter(20);

        let reset_filter = filter.reset();
        let mut filter_mut = reset_filter;

        // After reset, should behave like new filter
        filter_mut.filter(1);
        filter_mut.filter(2);
        let output = filter_mut.filter(3);
        assert_eq!(output, 1);
    }
}
