// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cross-correlation sinks for signal correlation analysis.
//!
//! Computes the normalized cross-correlation coefficient between two input signals over
//! a fixed-size sliding window. Takes tuple inputs `(T, T)` and produces correlation values.
//!
//! Computes the normalized cross-correlation coefficient between two signals
//! over a fixed-size sliding window.

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
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

/// The cross-correlation accumulator state.
///
/// Holds two ring buffers `Rx` and `Ry` (for the x and y signal windows
/// respectively) together with the current fill count. Both buffers must
/// have the same capacity.
#[derive(Clone, Debug)]
pub struct State<T, Rx, Ry> {
    /// The ring buffer holding the most recent x samples.
    pub buffer_x: Rx,
    /// The ring buffer holding the most recent y samples.
    pub buffer_y: Ry,
    /// The number of samples currently in each buffer (≤ `buffer_x.capacity()`).
    pub len: usize,
    _pd: core::marker::PhantomData<T>,
}

/// A sink that computes the normalized cross-correlation coefficient between two input signals.
///
/// Takes tuples of `(T, T)` and computes the dot product of the last N x and y samples,
/// normalized by N to produce a correlation coefficient at lag 0.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`CorrelationArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`CorrelationVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone, Debug)]
pub struct Correlation<T, Rx, Ry> {
    state: State<T, Rx, Ry>,
}

/// A cross-correlation sink backed by two const-generic [`FixedCircularBuffer`]s.
///
/// This alias is the `no_std`-friendly, zero-allocation form. Both `T` and
/// the window size `N` are fixed at compile time.
pub type CorrelationArray<T, const N: usize> =
    Correlation<T, FixedCircularBuffer<T, N>, FixedCircularBuffer<T, N>>;

/// A cross-correlation sink backed by two heap-allocated [`HeapCircularBuffer`]s.
///
/// Requires the `alloc` feature. Use [`Correlation::from_parts`] to construct
/// this variant, since the buffer capacities must be known at runtime.
#[cfg(feature = "alloc")]
pub type CorrelationVec<T> = Correlation<T, HeapCircularBuffer<T>, HeapCircularBuffer<T>>;

/// A cross-correlation sink that borrows two [`CircularBuffer`]s.
///
/// This alias allows sharing a caller-owned pair of ring buffers without
/// taking ownership of them. Construct via [`Correlation::from_parts`],
/// passing `&mut CircularBuffer<T>` for both signal buffers.
pub type CorrelationRefMut<'a, T> =
    Correlation<T, &'a mut CircularBuffer<T>, &'a mut CircularBuffer<T>>;

impl<T, Rx, Ry> Correlation<T, Rx, Ry>
where
    Rx: RingBuffer<T>,
    Ry: RingBuffer<T>,
{
    /// Creates a [`Correlation`] sink from two already-constructed ring buffers.
    ///
    /// Use this constructor when the buffers are not `Default`-constructible,
    /// e.g. for [`CorrelationVec`] whose capacity must be known at runtime.
    ///
    /// Both buffers are taken as-is with their current contents. If they
    /// contain pre-existing samples, the correlation computation will reflect
    /// those values from the first call.
    ///
    /// # Expected storage state
    ///
    /// For an idiomatic cold-start, pass two empty buffers.
    ///
    /// # Panics
    ///
    /// Panics if the two buffers do not have the same capacity, or if that
    /// capacity is zero.
    pub fn from_parts(bx: Rx, by: Ry) -> Self {
        assert!(bx.capacity() > 0, "Correlation: window size must be > 0");
        assert_eq!(
            bx.capacity(),
            by.capacity(),
            "Correlation: buffer_x capacity ({}) must equal buffer_y capacity ({})",
            bx.capacity(),
            by.capacity(),
        );
        Self {
            state: State {
                buffer_x: bx,
                buffer_y: by,
                len: 0,
                _pd: core::marker::PhantomData,
            },
        }
    }
}

impl<T, Rx, Ry> ConfigTrait for Correlation<T, Rx, Ry> {
    type Config = ();
}

impl<T, Rx, Ry> StateTrait for Correlation<T, Rx, Ry> {
    type State = State<T, Rx, Ry>;
}

impl<T, const N: usize> WithConfig for CorrelationArray<T, N>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(_config: Self::Config) -> Self::Output {
        Self {
            state: State {
                buffer_x: FixedCircularBuffer::new(),
                buffer_y: FixedCircularBuffer::new(),
                len: 0,
                _pd: core::marker::PhantomData,
            },
        }
    }
}

impl<T, const N: usize> Default for CorrelationArray<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(())
    }
}

impl<T, Rx, Ry> ConfigRef for Correlation<T, Rx, Ry> {
    fn config_ref(&self) -> &Self::Config {
        &()
    }
}

impl<T, Rx, Ry> ConfigClone for Correlation<T, Rx, Ry> {
    fn config(&self) -> Self::Config {}
}

impl<T, Rx, Ry> StateMut for Correlation<T, Rx, Ry> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, Rx, Ry> HasGuts for Correlation<T, Rx, Ry> {
    type Guts = State<T, Rx, Ry>;
}

impl<T, Rx, Ry> FromGuts for Correlation<T, Rx, Ry> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { state: guts }
    }
}

impl<T, Rx, Ry> IntoGuts for Correlation<T, Rx, Ry> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for CorrelationArray<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for CorrelationArray<T, N> where Self: Reset {}

impl<T, Rx, Ry> Sink<(T, T)> for Correlation<T, Rx, Ry>
where
    T: Clone + Num,
    Rx: RingBuffer<T>,
    Ry: RingBuffer<T>,
{
    #[inline]
    fn sink(&mut self, input: (T, T)) {
        let (x, y) = input;
        self.state.buffer_x.push_back(x);
        self.state.buffer_y.push_back(y);
        let cap = self.state.buffer_x.capacity();
        if self.state.len < cap {
            self.state.len += 1;
        }
    }
}

impl<T, Rx, Ry> Filter<(T, T)> for Correlation<T, Rx, Ry>
where
    T: Clone + Num + NumCast,
    Rx: RingBuffer<T>,
    Ry: RingBuffer<T>,
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

impl<T, Rx, Ry> Finalize for Correlation<T, Rx, Ry>
where
    T: Clone + Num + NumCast,
    Rx: RingBuffer<T>,
    Ry: RingBuffer<T>,
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
mod tests;
