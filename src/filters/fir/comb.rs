// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Feedforward-only comb filter implementation.
//!
//! A feedforward comb filter uses delayed versions of the input to create
//! a resonant filtering effect.
//!
//! Difference equation: `y[n] = x[n] + ff·x[n−D]`
//!
//! where D is the delay in samples. The feedforward path is always stable (FIR).

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::Num;

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

use crate::storage::RingBuffer;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The feedforward comb filter's configuration.
///
/// Contains the feedforward coefficient that controls the resonance
/// characteristics of the comb filter.
///
/// The feedforward path is always stable (FIR).
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Feedforward coefficient (multiplies x[n-D]).
    pub feedforward: T,
}

impl<T> Default for Config<T>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self {
            feedforward: T::zero(),
        }
    }
}

/// The feedforward comb filter's state.
///
/// Contains a ring-buffer `R` for the input delay line (feedforward component).
///
/// The `input_delay` is a ring-buffer that starts empty and returns `None`
/// for the first D pushes, naturally representing zero input history without pre-filling.
#[derive(Clone)]
pub struct State<R> {
    /// Input delay line for feedforward component.
    pub input_delay: R,
}

impl<R> core::fmt::Debug for State<R>
where
    R: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("State")
            .field("input_delay", &self.input_delay)
            .finish()
    }
}

/// A feedforward comb filter generic over delay-line storage `R`.
///
/// The delay length is determined by the capacity of the ring-buffer `R`.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`FeedforwardCombArray<T, D>`] — stack-allocated, `no_std`-friendly; delay `D` must be >= 1.
/// - [`FeedforwardCombVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone, Debug)]
pub struct FeedforwardComb<T, R> {
    config: Config<T>,
    state: State<R>,
}

/// A feedforward comb filter backed by a const-generic [`FixedCircularBuffer`] delay line.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The delay length `D` must be
/// at least 1; `FeedforwardCombArray<T, 0>` is rejected at compile time via [`WithConfig`].
pub type FeedforwardCombArray<T, const D: usize> = FeedforwardComb<T, FixedCircularBuffer<T, D>>;

/// A feedforward comb filter backed by a heap-allocated [`HeapCircularBuffer`] delay line.
///
/// Requires the `alloc` feature. Use [`FeedforwardComb::from_parts`] to construct this
/// variant, since the delay buffer capacity must be known at runtime.
#[cfg(feature = "alloc")]
pub type FeedforwardCombVec<T> = FeedforwardComb<T, HeapCircularBuffer<T>>;

/// A feedforward comb filter that borrows a [`CircularBuffer`] delay line.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. Construct via [`FeedforwardComb::from_parts`], passing
/// a `&mut CircularBuffer<T>` for the delay line.
pub type FeedforwardCombRefMut<'a, T> = FeedforwardComb<T, &'a mut CircularBuffer<T>>;

impl<T, R> FeedforwardComb<T, R>
where
    R: RingBuffer<T>,
{
    /// Creates a [`FeedforwardComb`] filter from an already-constructed `config` and
    /// `input_delay` ring-buffer.
    ///
    /// Use this constructor when the delay storage is not `Default`-constructible,
    /// e.g. for [`FeedforwardCombVec`] whose capacity must be known at runtime.
    ///
    /// The `input_delay` buffer is taken as-is with their current contents. If it contains
    /// pre-existing samples, those values are treated as past input history and the
    /// filter's first `D` outputs will include the delayed term `ff·x[n−D]`
    /// immediately.
    ///
    /// # Expected storage state
    ///
    /// For an idiomatic cold-start (where the first `D` outputs are just
    /// `x[n]` with no delayed term), pass an empty buffer.
    pub fn from_parts(config: Config<T>, input_delay: R) -> Self {
        Self {
            config,
            state: State { input_delay },
        }
    }
}

impl<T, const D: usize> Default for FeedforwardCombArray<T, D>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, R> ConfigTrait for FeedforwardComb<T, R> {
    type Config = Config<T>;
}

impl<T, R> StateTrait for FeedforwardComb<T, R> {
    type State = State<R>;
}

impl<T, const D: usize> WithConfig for FeedforwardCombArray<T, D>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        const {
            assert!(
                D >= 1,
                "FeedforwardComb<T, D>: delay length D must be at least 1"
            );
        };
        let state = {
            let input_delay = FixedCircularBuffer::default();
            State { input_delay }
        };
        Self { config, state }
    }
}

impl<T, R> ConfigRef for FeedforwardComb<T, R> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, R> ConfigClone for FeedforwardComb<T, R>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, R> StateMut for FeedforwardComb<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for FeedforwardComb<T, R> {
    type Guts = (Config<T>, State<R>);
}

impl<T, R> FromGuts for FeedforwardComb<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, R> IntoGuts for FeedforwardComb<T, R> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const D: usize> Reset for FeedforwardCombArray<T, D>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const D: usize> ResetMut for FeedforwardCombArray<T, D> where Self: Reset {}

impl<T, R> Filter<T> for FeedforwardComb<T, R>
where
    T: Clone + Num,
    R: RingBuffer<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let Config { ref feedforward } = self.config;
        let State {
            ref mut input_delay,
        } = self.state;

        let forward = input_delay
            .push_back(input.clone())
            .map_or_else(T::zero, |delayed| feedforward.clone() * delayed);

        input + forward
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_fir_comb_feedforward_only() {
        let filter = FeedforwardCombArray::<f32, 2>::with_config(Config { feedforward: 1.0 });

        let input = [1.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [1.0, 0.0, 1.0, 0.0, 0.0];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedforward_comb_zero_coefficient() {
        let filter = FeedforwardCombArray::<f32, 2>::with_config(Config { feedforward: 0.0 });

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedforward_comb_reset() {
        let mut filter = FeedforwardCombArray::<i32, 2>::with_config(Config { feedforward: 1 });

        filter.filter(10);
        filter.filter(20);

        let reset_filter = filter.reset();
        let mut filter_mut = reset_filter;

        let out1 = filter_mut.filter(5);
        let out2 = filter_mut.filter(6);
        let out3 = filter_mut.filter(7);

        assert_eq!(out1, 5);
        assert_eq!(out2, 6);
        assert_eq!(out3, 5 + 7);
    }
}
