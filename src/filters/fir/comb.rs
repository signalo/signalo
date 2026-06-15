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

use circular_buffer::CircularBuffer;
use num_traits::Num;

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
/// Contains a circular buffer for the input delay line (feedforward component).
///
/// The `input_delay` is a [`CircularBuffer`] that starts empty and returns `None`
/// for the first `D` pushes, naturally representing zero input history without pre-filling.
#[derive(Clone)]
pub struct State<T, const D: usize> {
    /// Input delay line for feedforward component
    pub input_delay: CircularBuffer<D, T>,
}

impl<T, const D: usize> core::fmt::Debug for State<T, D>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("State")
            .field("input_delay", &self.input_delay)
            .finish()
    }
}

/// A feedforward comb filter.
///
/// The delay length `D` must be at least 1; `FeedforwardComb<T, 0>` is rejected at compile time.
#[derive(Clone, Debug)]
pub struct FeedforwardComb<T, const D: usize> {
    config: Config<T>,
    state: State<T, D>,
}

impl<T, const D: usize> Default for FeedforwardComb<T, D>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, const D: usize> ConfigTrait for FeedforwardComb<T, D> {
    type Config = Config<T>;
}

impl<T, const D: usize> StateTrait for FeedforwardComb<T, D> {
    type State = State<T, D>;
}

impl<T, const D: usize> WithConfig for FeedforwardComb<T, D>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        const {
            assert!(
                D >= 1,
                "FeedforwardComb<T, D>: delay length D must be at least 1"
            )
        };
        let state = {
            let input_delay = CircularBuffer::default();
            State { input_delay }
        };
        Self { config, state }
    }
}

impl<T, const D: usize> ConfigRef for FeedforwardComb<T, D> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const D: usize> ConfigClone for FeedforwardComb<T, D>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const D: usize> StateMut for FeedforwardComb<T, D> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const D: usize> HasGuts for FeedforwardComb<T, D> {
    type Guts = (Config<T>, State<T, D>);
}

impl<T, const D: usize> FromGuts for FeedforwardComb<T, D> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const D: usize> IntoGuts for FeedforwardComb<T, D> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const D: usize> Reset for FeedforwardComb<T, D>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const D: usize> ResetMut for FeedforwardComb<T, D> where Self: Reset {}

impl<T, const D: usize> Filter<T> for FeedforwardComb<T, D>
where
    T: Clone + Num,
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
        let filter = FeedforwardComb::<f32, 2>::with_config(Config { feedforward: 1.0 });

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
        let filter = FeedforwardComb::<f32, 2>::with_config(Config { feedforward: 0.0 });

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedforward_comb_reset() {
        let mut filter = FeedforwardComb::<i32, 2>::with_config(Config { feedforward: 1 });

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

    #[test]
    fn test_feedforward_comb_state_mut() {
        let mut filter = FeedforwardComb::<f32, 2>::default();
        filter.filter(1.0);
        filter.filter(2.0);

        let output = filter.filter(3.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_feedforward_comb_from_into_guts() {
        let filter: FeedforwardComb<i32, 2> = FeedforwardComb::default();
        let guts = filter.into_guts();
        let _new_filter: FeedforwardComb<i32, 2> = FromGuts::from_guts(guts);
    }

    #[test]
    fn smoke() {
        let filter = FeedforwardComb::<f32, 2>::with_config(Config { feedforward: 0.0 });
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }
}
