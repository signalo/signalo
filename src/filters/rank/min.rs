// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving minimum filters.

use core::fmt;

use circular_buffer::FixedCircularBuffer;
use num_traits::Num;

use crate::storage::RingBuffer;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The min filter's state.
///
/// Generic over the ring-buffer backend `R` that stores the monotonic-deque
/// window. The element type stored in the deque is `(T, usize)` — a
/// `(value, timestamp)` pair. Use [`MinArray`] for stack-allocated storage
/// or [`MinVec`] for heap-allocated storage.
#[derive(Clone)]
pub struct State<T, R> {
    /// The discrete timestamp of the latest input.
    pub time: usize,
    /// The current taps buffer (monotonic deque of `(value, timestamp)` pairs).
    pub taps: R,
    /// Marker to associate the value type `T` with the state without storing
    /// it directly (the element type `(T, usize)` is carried by `R`).
    _pd: core::marker::PhantomData<T>,
}

impl<T, R> fmt::Debug for State<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("time", &self.time)
            .field("taps", &self.taps)
            .finish()
    }
}

/// A min filter producing the moving minimum over a given signal.
///
/// The filter maintains a monotone increasing deque of `(value, timestamp)`
/// pairs, evicting out-of-window entries from the front and larger-than-input
/// entries from the back on each call to [`Filter::filter`]. This gives
/// amortised O(1) time per sample.
///
/// # Storage
///
/// The tap ring-buffer backend is selected by the `R` type parameter.
/// Prefer the concrete aliases for common use:
///
/// - [`MinArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`MinVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Complexity
///
/// - **Time per sample:** O(N) amortised O(1); same monotone-deque argument as `Max`.
/// - **Space:** O(N); deque holds at most N `(value, timestamp)` pairs.
#[derive(Clone)]
pub struct Min<T, R> {
    state: State<T, R>,
}

/// A min filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The tap
/// ring-buffer lives entirely on the stack.
pub type MinArray<T, const N: usize> = Min<T, FixedCircularBuffer<(T, usize), N>>;

/// A min filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`Min::from_parts`] to construct
/// this variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type MinVec<T> = Min<T, HeapCircularBuffer<(T, usize)>>;

impl<T, const N: usize> Default for MinArray<T, N> {
    fn default() -> Self {
        assert!(N > 0, "Min: window size N must be > 0");
        Self {
            state: State {
                time: 0,
                taps: FixedCircularBuffer::default(),
                _pd: core::marker::PhantomData,
            },
        }
    }
}

impl<T, R> Min<T, R>
where
    R: RingBuffer<(T, usize)>,
{
    /// Creates a [`Min`] filter from an already-constructed `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`MinVec`] whose capacity must be known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `taps.capacity()` is zero.
    pub fn from_parts(taps: R) -> Self {
        assert!(
            taps.capacity() > 0,
            "Min: window size (taps capacity) must be > 0"
        );
        Self {
            state: State {
                time: 0,
                taps,
                _pd: core::marker::PhantomData,
            },
        }
    }
}

impl<T, R> fmt::Debug for Min<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Min").field("state", &self.state).finish()
    }
}

impl<T, R> StateTrait for Min<T, R> {
    type State = State<T, R>;
}

impl<T, R> StateMut for Min<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Min<T, R> {
    type Guts = State<T, R>;
}

impl<T, R> FromGuts for Min<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, R> IntoGuts for Min<T, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for MinArray<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MinArray<T, N> where Self: Reset {}

impl<T, R> Filter<T> for Min<T, R>
where
    T: Clone + Num + PartialOrd,
    R: RingBuffer<(T, usize)>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let current_time = self.state.time;
        let n = self.state.taps.capacity();

        // pop all items that have left the moving window, from the front:
        while self
            .state
            .taps
            .front()
            .is_some_and(|(_, time)| time + n <= current_time)
        {
            let _ = self.state.taps.pop_front();
        }

        // pop all items larger than the input, from the back:
        while self
            .state
            .taps
            .back()
            .map_or(false, |(value, _)| &input < value)
        {
            let _ = self.state.taps.pop_back();
        }

        // push the input, to the back:
        self.state.taps.push_back((input, current_time));

        if self.state.time < usize::MAX {
            self.state.time += 1;
        } else {
            // Time has overflown, so we need to adjust our state accordingly:
            let offset = self.state.time - n;
            for (_, time) in self.state.taps.iter_mut() {
                *time -= offset;
            }
            self.state.time = n + 1;
        }

        self.state
            .taps
            .front()
            .expect("min taps must be non-empty; push_back guarantees at least one element")
            .0
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use super::*;

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _: MinArray<f32, 0> = MinArray::default();
    }

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
            0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 5.0, 3.0, 3.0, 3.0, 6.0, 6.0, 9.0, 9.0, 9.0, 4.0, 4.0,
            4.0, 12.0, 7.0, 7.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 5.0,
            5.0, 5.0, 13.0, 13.0, 13.0, 21.0, 21.0, 8.0, 8.0, 8.0, 8.0, 8.0, 16.0, 16.0, 16.0,
            11.0, 11.0, 11.0,
        ]
    }

    #[test]
    fn overflow_monotonicity() {
        const N: usize = 3;
        let mut filter: MinArray<usize, N> = MinArray::default();
        // Pump N values so the deque has realistic content, then wind time forward.
        filter.filter(0);
        filter.filter(0);
        filter.filter(0);
        filter.state_mut().time = usize::MAX;
        // Forcibly clear the deque so stale timestamps don't trigger overflow
        // in the expiry check before the recovery branch can run.
        filter.state_mut().taps.clear();
        filter.filter(10); // time == usize::MAX → triggers overflow branch
        filter.filter(20);
        // Window = [10, 20]; min = 10
        assert_eq!(filter.filter(15), 10);
    }

    #[test]
    fn test() {
        const N: usize = 3;

        let filter: MinArray<f32, N> = MinArray::default();

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, get_output());
    }
}
