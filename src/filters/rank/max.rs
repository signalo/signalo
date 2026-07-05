// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving maximum filters.

use circular_buffer::FixedCircularBuffer;
use core::fmt;
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

/// The max filter's state.
///
/// Generic over the ring-buffer backend `R` that stores the monotonic-deque
/// window. The element type stored in the deque is `(T, usize)` — a
/// `(value, timestamp)` pair. Use [`MaxArray`] for stack-allocated storage
/// or [`MaxVec`] for heap-allocated storage.
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

/// A max filter producing the moving maximum over a given signal.
///
/// The filter maintains a monotone decreasing deque of `(value, timestamp)`
/// pairs, evicting out-of-window entries from the front and smaller-than-input
/// entries from the back on each call to [`Filter::filter`]. This gives
/// amortised O(1) time per sample.
///
/// # Storage
///
/// The tap ring-buffer backend is selected by the `R` type parameter.
/// Prefer the concrete aliases for common use:
///
/// - [`MaxArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`MaxVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Complexity
///
/// - **Time per sample:** O(N) amortised O(1); each element is pushed and popped at most once
///   across N calls (monotone deque); O(N) only on the rare `usize::MAX` timestamp recovery.
/// - **Space:** O(N); deque holds at most N `(value, timestamp)` pairs.
#[derive(Clone)]
pub struct Max<T, R> {
    state: State<T, R>,
}

/// A max filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The tap
/// ring-buffer lives entirely on the stack.
pub type MaxArray<T, const N: usize> = Max<T, FixedCircularBuffer<(T, usize), N>>;

/// A max filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`Max::from_parts`] to construct
/// this variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type MaxVec<T> = Max<T, HeapCircularBuffer<(T, usize)>>;

impl<T, const N: usize> Default for MaxArray<T, N> {
    fn default() -> Self {
        assert!(N > 0, "Max: window size N must be > 0");
        Self {
            state: State {
                time: 0,
                taps: FixedCircularBuffer::default(),
                _pd: core::marker::PhantomData,
            },
        }
    }
}

impl<T, R> Max<T, R>
where
    T: PartialOrd,
    R: RingBuffer<(T, usize)>,
{
    /// Creates a [`Max`] filter from an already-constructed `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`MaxVec`] whose capacity must be known at runtime.
    ///
    /// The `taps` deque is taken as-is with their current contents. For correct results,
    /// the deque must be empty or monotonically decreasing (`value_i >= value_{i+1}`)
    /// with strictly increasing timestamps.
    ///
    /// # Expected storage state
    ///
    /// For an idiomatic cold-start, pass an empty buffer.
    ///
    /// # Panics
    ///
    /// Panics if `taps.capacity()` is zero.
    pub fn from_parts(taps: R) -> Self {
        assert!(
            taps.capacity() > 0,
            "Max: window size (taps capacity) must be > 0"
        );

        debug_assert!(
            {
                let mut prev: Option<&(T, usize)> = None;
                let mut ok = true;
                for current in taps.iter() {
                    if let Some((prev_val, prev_ts)) = prev {
                        if prev_val < &current.0 || *prev_ts >= current.1 {
                            ok = false;
                            break;
                        }
                    }
                    prev = Some(current);
                }
                ok
            },
            "Max: the taps deque must be empty or monotonically decreasing with strictly increasing timestamps"
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

impl<T, R> fmt::Debug for Max<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Max").field("state", &self.state).finish()
    }
}

impl<T, R> StateTrait for Max<T, R> {
    type State = State<T, R>;
}

impl<T, R> StateMut for Max<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Max<T, R> {
    type Guts = State<T, R>;
}

impl<T, R> FromGuts for Max<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, R> IntoGuts for Max<T, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for MaxArray<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MaxArray<T, N> where Self: Reset {}

impl<T, R> Filter<T> for Max<T, R>
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
            .map_or(false, |(value, _)| &input > value)
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
            .expect("max taps must be non-empty; push_back guarantees at least one element")
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
        let _: MaxArray<f32, 0> = MaxArray::default();
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
            0.0, 1.0, 7.0, 7.0, 7.0, 8.0, 16.0, 16.0, 19.0, 19.0, 19.0, 14.0, 14.0, 17.0, 17.0,
            17.0, 17.0, 20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 23.0, 23.0, 111.0, 111.0, 111.0,
            18.0, 106.0, 106.0, 106.0, 26.0, 26.0, 21.0, 21.0, 21.0, 34.0, 34.0, 109.0, 109.0,
            109.0, 29.0, 29.0, 16.0, 104.0, 104.0, 104.0, 24.0,
        ]
    }

    #[test]
    fn overflow_monotonicity() {
        const N: usize = 3;
        let mut filter: MaxArray<usize, N> = MaxArray::default();
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
        filter.filter(15);
        // Window should contain [10, 20, 15]; max = 20
        assert_eq!(filter.filter(1), 20);
    }

    #[test]
    fn test() {
        const N: usize = 3;

        let filter: MaxArray<f32, N> = MaxArray::default();

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, get_output());
    }
}
