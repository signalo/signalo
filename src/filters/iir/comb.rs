// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Feedback-only comb filter implementation.
//!
//! A feedback comb filter uses delayed versions of the output to create
//! a resonant filtering effect.
//!
//! Difference equation: `y[n] = x[n] + fb·y[n−D]`
//!
//! where D is the delay in samples. This follows the standard Schroeder/Zölzer convention:
//! a positive `feedback` value produces an additive feedback comb. Stability requires `|fb| < 1`.

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The feedback comb filter's configuration.
///
/// Contains the feedback coefficient that controls the resonance
/// and decay characteristics of the comb filter.
///
/// # Stability
///
/// The feedback path is stable when `|feedback| < 1`.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Feedback coefficient (multiplies y[n−D]).
    ///
    /// Uses the additive (Schroeder) convention: a positive value produces constructive
    /// resonance. Stability requires `|feedback| < 1`.
    pub feedback: T,
}

impl<T> Default for Config<T>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self {
            feedback: T::zero(),
        }
    }
}

/// The feedback comb filter's state.
///
/// Contains an array for the output delay line (feedback component).
///
/// Uses a pre-zeroed `[T; D]` array with a manual `output_index` pointer.
///
/// # Invariant
///
/// At the start of each [`Filter::filter`] call (and between calls),
/// `output_delay[output_index]` holds `y[n−D]`; the output that is `D` steps old.
/// When pre-warming this array externally, write the oldest sample to
/// `output_delay[output_index]` and the most-recent sample to
/// `output_delay[(output_index + D - 1) % D]`.
/// External mutation of `output_delay` must preserve this invariant together with
/// `output_index`; otherwise the next [`Filter::filter`] call will read a stale
/// delayed sample.
#[derive(Clone)]
pub struct State<T, const D: usize> {
    /// Circular buffer of the last `D` outputs used for the feedback path.
    pub output_delay: [T; D],
    /// Index for circular `output_delay` array
    pub output_index: usize,
}

impl<T, const D: usize> core::fmt::Debug for State<T, D>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("State")
            .field("output_delay", &self.output_delay)
            .field("output_index", &self.output_index)
            .finish()
    }
}

/// A feedback comb filter.
///
/// The delay length `D` must be at least 1; `FeedbackComb<T, 0>` is rejected at compile time.
#[derive(Clone, Debug)]
pub struct FeedbackComb<T, const D: usize> {
    config: Config<T>,
    state: State<T, D>,
}

impl<T, const D: usize> Default for FeedbackComb<T, D>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, const D: usize> ConfigTrait for FeedbackComb<T, D> {
    type Config = Config<T>;
}

impl<T, const D: usize> StateTrait for FeedbackComb<T, D> {
    type State = State<T, D>;
}

impl<T, const D: usize> WithConfig for FeedbackComb<T, D>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        const {
            assert!(
                D >= 1,
                "FeedbackComb<T, D>: delay length D must be at least 1"
            )
        };
        let state = {
            let output_delay = core::array::from_fn(|_| T::zero());
            let output_index = 0;
            State {
                output_delay,
                output_index,
            }
        };
        Self { config, state }
    }
}

impl<T, const D: usize> ConfigRef for FeedbackComb<T, D> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const D: usize> ConfigClone for FeedbackComb<T, D>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const D: usize> StateMut for FeedbackComb<T, D> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const D: usize> HasGuts for FeedbackComb<T, D> {
    type Guts = (Config<T>, State<T, D>);
}

impl<T, const D: usize> FromGuts for FeedbackComb<T, D> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const D: usize> IntoGuts for FeedbackComb<T, D> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const D: usize> Reset for FeedbackComb<T, D>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const D: usize> ResetMut for FeedbackComb<T, D> where Self: Reset {}

impl<T, const D: usize> Filter<T> for FeedbackComb<T, D>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let Config { ref feedback } = self.config;
        let State {
            ref mut output_delay,
            ref mut output_index,
        } = self.state;

        let output = input + feedback.clone() * output_delay[*output_index].clone();

        output_delay[*output_index] = output.clone();
        *output_index = (*output_index + 1) % D;

        output
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_feedback_comb_simple() {
        let filter = FeedbackComb::<f32, 2>::with_config(Config { feedback: 0.5 });

        let input = [1.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [1.0, 0.0, 0.5, 0.0, 0.25];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedback_comb_zero_coefficient() {
        let filter = FeedbackComb::<f32, 2>::with_config(Config { feedback: 0.0 });

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedback_comb_reset() {
        let mut filter = FeedbackComb::<i32, 2>::with_config(Config { feedback: 1 });

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
    fn test_feedback_comb_state_mut() {
        let mut filter = FeedbackComb::<f32, 2>::default();
        filter.filter(1.0);
        filter.filter(2.0);

        let state = filter.state_mut();
        assert_eq!(state.output_delay[0], 1.0);
        assert_eq!(state.output_delay[1], 2.0);

        let output = filter.filter(3.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_feedback_comb_from_into_guts() {
        let filter: FeedbackComb<i32, 2> = FeedbackComb::default();
        let guts = filter.into_guts();
        let _new_filter: FeedbackComb<i32, 2> = FromGuts::from_guts(guts);
    }

    #[test]
    fn smoke() {
        let filter = FeedbackComb::<f32, 2>::with_config(Config { feedback: 0.0 });
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

    #[test]
    fn test_feedback_comb_marginally_stable_with_unit_feedback() {
        // With |feedback| = 1 the impulse circulates forever without decaying (marginal stability).
        // D=2, fb=1: impulse at n=0 reappears every 2 samples → outputs[n] = 1.0 for even n, 0.0 for odd n.
        let mut filter = FeedbackComb::<f64, 2>::with_config(Config { feedback: 1.0 });
        let outputs: Vec<_> = (0..20)
            .map(|i| filter.filter(if i == 0 { 1.0 } else { 0.0 }))
            .collect();

        for i in (0..20).step_by(2) {
            assert_abs_diff_eq!(outputs[i], 1.0, epsilon = 1e-12);
        }

        for i in (1..20).step_by(2) {
            assert_abs_diff_eq!(outputs[i], 0.0, epsilon = 1e-12);
        }
    }
}
