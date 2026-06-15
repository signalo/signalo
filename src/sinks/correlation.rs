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
mod tests;
