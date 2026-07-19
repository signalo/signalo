// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving bounds filters.

use core::fmt;

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
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

/// The bounds filter's state.
#[derive(Clone)]
pub struct State<T, R> {
    /// The internal `min` filter.
    pub min: super::min::Min<T, R>,
    /// The internal `max` filter.
    pub max: super::max::Max<T, R>,
}

impl<T, const N: usize> Default for State<T, circular_buffer::FixedCircularBuffer<(T, usize), N>> {
    fn default() -> Self {
        Self {
            min: super::min::MinArray::default(),
            max: super::max::MaxArray::default(),
        }
    }
}

impl<T, R> fmt::Debug for State<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

/// A bounds filter producing the moving bounds over a given signal.
///
/// # Complexity
///
/// - **Time per sample:** O(N) amortized; each sample triggers monotonic deque maintenance on
///   both the min and max windows of length N.
/// - **Space:** O(N); two deques of at most N elements each.
#[derive(Clone)]
pub struct Bounds<T, R> {
    state: State<T, R>,
}

/// A bounds filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
pub type BoundsArray<T, const N: usize> = Bounds<T, FixedCircularBuffer<(T, usize), N>>;

/// A bounds filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
#[cfg(feature = "alloc")]
pub type BoundsVec<T> = Bounds<T, HeapCircularBuffer<(T, usize)>>;

/// A bounds filter that borrows a [`CircularBuffer`] tap buffer.
pub type BoundsRefMut<'a, T> = Bounds<T, &'a mut CircularBuffer<(T, usize)>>;

impl<T, const N: usize> Default for BoundsArray<T, N> {
    fn default() -> Self {
        Self {
            state: State::default(),
        }
    }
}

impl<T, R> fmt::Debug for Bounds<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Bounds")
            .field("state", &self.state)
            .finish()
    }
}

impl<T, R> StateTrait for Bounds<T, R> {
    type State = State<T, R>;
}

impl<T, R> StateMut for Bounds<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Bounds<T, R> {
    type Guts = State<T, R>;
}

impl<T, R> FromGuts for Bounds<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, R> IntoGuts for Bounds<T, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for BoundsArray<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BoundsArray<T, N> where Self: Reset {}

impl<T, R> Filter<T> for Bounds<T, R>
where
    T: Clone + Num + PartialOrd,
    R: RingBuffer<(T, usize)>,
{
    type Output = (T, T);

    fn filter(&mut self, input: T) -> Self::Output {
        let min = self.state.min.filter(input.clone());
        let max = self.state.max.filter(input);
        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<(f32, f32)> {
        vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 7.0),
            (1.0, 7.0),
            (2.0, 7.0),
            (2.0, 8.0),
            (5.0, 16.0),
            (3.0, 16.0),
            (3.0, 19.0),
            (3.0, 19.0),
            (6.0, 19.0),
            (6.0, 14.0),
            (9.0, 14.0),
            (9.0, 17.0),
            (9.0, 17.0),
            (4.0, 17.0),
            (4.0, 17.0),
            (4.0, 20.0),
            (12.0, 20.0),
            (7.0, 20.0),
            (7.0, 20.0),
            (7.0, 15.0),
            (7.0, 15.0),
            (10.0, 15.0),
            (10.0, 23.0),
            (10.0, 23.0),
            (10.0, 111.0),
            (10.0, 111.0),
            (18.0, 111.0),
            (18.0, 18.0),
            (18.0, 106.0),
            (5.0, 106.0),
            (5.0, 106.0),
            (5.0, 26.0),
            (13.0, 26.0),
            (13.0, 21.0),
            (13.0, 21.0),
            (21.0, 21.0),
            (21.0, 34.0),
            (8.0, 34.0),
            (8.0, 109.0),
            (8.0, 109.0),
            (8.0, 109.0),
            (8.0, 29.0),
            (16.0, 29.0),
            (16.0, 16.0),
            (16.0, 104.0),
            (11.0, 104.0),
            (11.0, 104.0),
            (11.0, 24.0),
        ]
    }

    #[test]
    fn test() {
        const N: usize = 3;

        let filter: BoundsArray<f32, N> = BoundsArray::default();

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, get_output());
    }

    #[test]
    fn test_default_state() {
        use crate::traits::guts::FromGuts;

        const N: usize = 5;

        let state: State<f32, FixedCircularBuffer<(f32, usize), N>> = State::default();
        let bounds = Bounds::from_guts(state);

        // Filter should work correctly after default initialization
        let mut bounds = bounds;
        let output = bounds.filter(10.0);
        assert_eq!(output, (10.0, 10.0));
    }

    #[test]
    fn test_state_mut() {
        const N: usize = 3;

        let mut filter: BoundsArray<f32, N> = BoundsArray::default();
        filter.filter(10.0);
        filter.filter(20.0);

        let state = filter.state_mut();
        // Verify we can access the internal min and max filters
        let _ = &state.min;
        let _ = &state.max;
    }

    #[test]
    fn test_from_into_guts() {
        use crate::traits::guts::{FromGuts, IntoGuts};

        const N: usize = 3;

        let mut filter: BoundsArray<f32, N> = BoundsArray::default();
        filter.filter(5.0);
        filter.filter(15.0);
        filter.filter(10.0);

        let guts = filter.into_guts();
        let filter2 = Bounds::from_guts(guts);

        // The reconstructed filter should produce the same results
        let mut filter2 = filter2;
        let output = filter2.filter(8.0);
        assert_eq!(output, (8.0, 15.0));
    }

    #[test]
    fn test_reset() {
        const N: usize = 3;

        let mut filter: BoundsArray<f32, N> = BoundsArray::default();
        filter.filter(100.0);
        filter.filter(200.0);
        filter.filter(50.0);

        // Reset the filter
        let mut reset_filter = filter.reset();

        // After reset, the bounds should start fresh
        reset_filter.filter(10.0);
        let output = reset_filter.filter(20.0);
        assert_eq!(output, (10.0, 20.0));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bounds_vec_emits_min_max() {
        use crate::filters::rank::{max::MaxVec, min::MinVec};
        use crate::traits::guts::FromGuts;

        let ring_min = HeapCircularBuffer::<(i32, usize)>::with_capacity(3);
        let ring_max = HeapCircularBuffer::<(i32, usize)>::with_capacity(3);
        let min: MinVec<i32> = MinVec::from_parts(ring_min);
        let max: MaxVec<i32> = MaxVec::from_parts(ring_max);
        let state = State { min, max };
        let mut bounds: BoundsVec<i32> = Bounds::from_guts(state);
        bounds.filter(5);
        bounds.filter(3);
        bounds.filter(7);
        let result = bounds.filter(1337);
        assert_eq!(result, (3, 1337));
    }

    #[test]
    fn bounds_ref_mut_emits_min_max() {
        use crate::filters::rank::{max::MaxRefMut, min::MinRefMut};
        use crate::traits::guts::FromGuts;

        let mut ring_min = FixedCircularBuffer::<(i32, usize), 3>::new();
        let mut ring_max = FixedCircularBuffer::<(i32, usize), 3>::new();
        let min = MinRefMut::from_parts(&mut ring_min);
        let max = MaxRefMut::from_parts(&mut ring_max);
        let state = State { min, max };
        let mut bounds = BoundsRefMut::from_guts(state);
        bounds.filter(5);
        bounds.filter(3);
        bounds.filter(7);
        let result = bounds.filter(1337);
        assert_eq!(result, (3, 1337));
    }
}
