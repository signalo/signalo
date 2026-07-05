// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Feedback comb filter.
//!
//! A feedback comb filter uses delayed versions of the output to create
//! a resonant filtering effect. Stability requires `|feedback| < 1`.
//! A feedback comb filter uses delayed versions of the output to create
//! a resonant filtering effect.
//!
//! Difference equation: `y[n] = x[n] + fb·y[n−D]`
//!
//! where D is the delay in samples. This follows the standard Schroeder/Zölzer convention:
//! a positive `feedback` value produces an additive feedback comb. Stability requires `|fb| < 1`.

use num_traits::Num;

use circular_buffer::FixedCircularBuffer;

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

use crate::storage::{zero_filled_fixed_ring, RingBuffer};
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
/// Holds the delay-line ring buffer `R` that stores the last `D` output samples
/// used for the feedback tap. `R` is typically a [`FixedCircularBuffer<T, D>`]
/// (via the [`FeedbackCombArray`] alias) or, with the `alloc` feature, a
/// [`HeapCircularBuffer<T>`] (via [`FeedbackCombVec`]).
///
/// # Invariant
///
/// `output_delay` is a ring buffer at capacity `D`, pre-filled with zeros.
/// The front element (index 0, oldest) is `y[n−D]`.
/// External mutation must preserve the "oldest-at-front" invariant; otherwise
/// the next [`Filter::filter`] call will read a stale delayed sample.
#[derive(Clone)]
pub struct State<R> {
    /// Ring buffer of the last `D` outputs used for the feedback path.
    ///
    /// Index 0 (the front / oldest element) is `y[n−D]`.
    pub output_delay: R,
}

impl<R> core::fmt::Debug for State<R>
where
    R: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("State")
            .field("output_delay", &self.output_delay)
            .finish()
    }
}

/// A feedback comb filter generic over its delay-line storage `R`.
///
/// Prefer the concrete type aliases for everyday use:
/// - [`FeedbackCombArray<T, D>`] — stack-allocated, `no_std`-friendly.
/// - [`FeedbackCombVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone, Debug)]
pub struct FeedbackComb<T, R> {
    config: Config<T>,
    state: State<R>,
}

/// A feedback comb filter backed by a const-generic [`FixedCircularBuffer`] delay line.
///
/// This is the `no_std`-friendly, zero-allocation form. The delay ring-buffer
/// lives entirely on the stack. `D` is the delay in samples and must be at
/// least 1.
pub type FeedbackCombArray<T, const D: usize> = FeedbackComb<T, FixedCircularBuffer<T, D>>;

/// A feedback comb filter backed by a heap-allocated [`HeapCircularBuffer`] delay line.
///
/// Requires the `alloc` feature. Use [`FeedbackComb::from_parts`] to construct
/// this variant, providing a pre-allocated and pre-filled ring buffer.
#[cfg(feature = "alloc")]
pub type FeedbackCombVec<T> = FeedbackComb<T, HeapCircularBuffer<T>>;

impl<T, R> FeedbackComb<T, R>
where
    R: RingBuffer<T>,
{
    /// Creates a [`FeedbackComb`] filter from a [`Config`] and a pre-constructed delay-line `R`.
    ///
    /// Use this constructor when the ring buffer storage is not `Default`-constructible,
    /// for example for [`FeedbackCombVec`] whose capacity must be known at runtime.
    ///
    /// The caller is responsible for pre-filling `output_delay` with `D` zeros (or the
    /// desired initial state) before passing it here; otherwise the filter's first output
    /// samples may be incorrect.
    pub fn from_parts(config: Config<T>, output_delay: R) -> Self {
        Self {
            config,
            state: State { output_delay },
        }
    }
}

impl<T, const D: usize> Default for FeedbackCombArray<T, D>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, R> ConfigTrait for FeedbackComb<T, R> {
    type Config = Config<T>;
}

impl<T, R> StateTrait for FeedbackComb<T, R> {
    type State = State<R>;
}

impl<T, const D: usize> WithConfig for FeedbackCombArray<T, D>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        const {
            assert!(
                D >= 1,
                "FeedbackCombArray<T, D>: delay length D must be at least 1"
            );
        };
        let state = {
            let output_delay = zero_filled_fixed_ring::<T, D>();
            State { output_delay }
        };
        Self { config, state }
    }
}

impl<T, R> ConfigRef for FeedbackComb<T, R> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, R> ConfigClone for FeedbackComb<T, R>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, R> StateMut for FeedbackComb<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for FeedbackComb<T, R> {
    type Guts = (Config<T>, State<R>);
}

impl<T, R> FromGuts for FeedbackComb<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, R> IntoGuts for FeedbackComb<T, R> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const D: usize> Reset for FeedbackCombArray<T, D>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const D: usize> ResetMut for FeedbackCombArray<T, D> where Self: Reset {}

impl<T, R> Filter<T> for FeedbackComb<T, R>
where
    T: Clone + Num,
    R: RingBuffer<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let Config { ref feedback } = self.config;
        let State {
            ref mut output_delay,
        } = self.state;

        // The front (oldest) element of the ring is y[n−D].
        let delayed = output_delay.iter().next().cloned().unwrap_or_else(T::zero);

        let output = input + feedback.clone() * delayed;

        // Push the new output; the evicted element is the old y[n−D] we already consumed.
        output_delay.push_back(output.clone());

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
        let filter = FeedbackCombArray::<f32, 2>::with_config(Config { feedback: 0.5 });

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
        let filter = FeedbackCombArray::<f32, 2>::with_config(Config { feedback: 0.0 });

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_feedback_comb_reset() {
        let mut filter = FeedbackCombArray::<i32, 2>::with_config(Config { feedback: 1 });

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
        let mut filter = FeedbackCombArray::<f32, 2>::default();
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
        let filter: FeedbackCombArray<i32, 2> = FeedbackCombArray::default();
        let guts = filter.into_guts();
        let _new_filter: FeedbackCombArray<i32, 2> = FromGuts::from_guts(guts);
    }

    #[test]
    fn smoke() {
        let filter = FeedbackCombArray::<f32, 2>::with_config(Config { feedback: 0.0 });
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
        let mut filter = FeedbackCombArray::<f64, 2>::with_config(Config { feedback: 1.0 });
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
