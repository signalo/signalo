// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Allpass filter for phase manipulation without magnitude modification.
//!
//! An allpass filter has constant magnitude response across all frequencies.
//! It passes all frequency components with equal gain while altering only the phase response.
//! Transfer function: `H(z) = (c + z^-1) / (1 + c*z^-1)` where `c` is the allpass coefficient.
//! An allpass filter has constant magnitude response across all frequencies,
//! meaning it passes all frequency components with equal gain. It only alters
//! the phase response of the signal.
//!
//! This implementation provides a first-order allpass filter with the transfer function:
//! `H(z) = (c + z^-1) / (1 + c*z^-1)`
//!
//! Difference equation: `y[n] = c·(x[n]−y[n−1]) + x[n−1]`
//!
//! When `c = 0`, the filter becomes a pure delay (output equals previous input).

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The allpass filter's configuration.
///
/// # Stability
///
/// The filter is stable when `|c| < 1`. At `|c| = 1` the pole reaches the unit circle
/// (marginal stability); at `|c| > 1` the filter is unstable.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Allpass coefficient (Schroeder single-multiply form).
    ///
    /// Uses the additive pole convention: a value of `c = +p` places the feedback pole
    /// at `z = +p`. Stability requires `|c| < 1`.
    ///
    /// When `c = 0` the filter degenerates to a pure 1-sample delay (`y[n] = x[n-1]`),
    /// not an identity. Use `c ≠ 0` for actual all-pass phase rotation.
    pub c: T,
}

/// The allpass filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Previous input sample (x[n-1]).
    pub prev_input: T,
    /// Previous output sample (y[n-1]).
    pub prev_output: T,
}

/// A first-order allpass filter.
#[derive(Clone, Debug)]
pub struct Allpass<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> ConfigTrait for Allpass<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Allpass<T> {
    type State = State<T>;
}

impl<T> WithConfig for Allpass<T>
where
    T: Clone + num_traits::Zero,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let prev_input = T::zero();
            let prev_output = T::zero();
            State {
                prev_input,
                prev_output,
            }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Allpass<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Allpass<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Allpass<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Allpass<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Allpass<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Allpass<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Allpass<T>
where
    T: Clone + num_traits::Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Allpass<T> where Self: Reset {}

impl<T> Filter<T> for Allpass<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // y[n] = c·(x[n]−y[n−1]) + x[n−1]  (Schroeder single-multiply form)
        let output = self.config.c.clone() * (input.clone() - self.state.prev_output.clone())
            + self.state.prev_input.clone();

        self.state.prev_input = input;
        self.state.prev_output = output.clone();

        output
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_pure_delay_at_zero_coefficient() {
        // When c=0, the filter becomes a pure delay: y[n] = x[n-1]
        let config = Config { c: 0.0_f32 };
        let filter = Allpass::with_config(config);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        // First output is 0 (no previous input)
        assert_abs_diff_eq!(output[0], 0.0, epsilon = 0.0001);
        // Subsequent outputs are previous inputs
        assert_abs_diff_eq!(output[1], 1.0, epsilon = 0.0001);
        assert_abs_diff_eq!(output[2], 2.0, epsilon = 0.0001);
        assert_abs_diff_eq!(output[3], 3.0, epsilon = 0.0001);
        assert_abs_diff_eq!(output[4], 4.0, epsilon = 0.0001);
    }

    #[test]
    fn test_unit_magnitude_response() {
        // Verify |H(e^{jω})| ≈ 1 for several frequencies (the allpass property).
        // Drive with a pure sinusoid for 2000 samples; measure RMS of last 500 (steady state).
        for &(c, freq) in &[
            (0.5_f32, 0.05_f32),
            (0.5, 0.125),
            (0.5, 0.25),
            (0.5, 0.375),
            (0.5, 0.45),
            (-0.5, 0.05),
            (-0.5, 0.25),
            (-0.5, 0.45),
        ] {
            let mut filter = Allpass::with_config(Config { c });
            let n_total = 2000_usize;
            let n_steady = 500_usize;
            let mut in_ss = 0.0_f32;
            let mut out_ss = 0.0_f32;

            for n in 0..n_total {
                let x = (2.0 * core::f32::consts::PI * freq * n as f32).sin();
                let y = filter.filter(x);

                if n >= n_total - n_steady {
                    in_ss += x * x;
                    out_ss += y * y;
                }
            }

            let power_ratio = out_ss / in_ss;
            assert!(
                (power_ratio - 1.0).abs() < 1e-3,
                "unit magnitude failed at c={}, freq={}: power_ratio={}",
                c,
                freq,
                power_ratio
            );
        }
    }

    #[test]
    fn test_negative_coefficient() {
        // Test with negative coefficient
        let config = Config { c: -0.5_f32 };
        let filter = Allpass::with_config(config);

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        // y[0] = c*(x[0] - y[-1]) + x[-1] = -0.5*(1 - 0) + 0 = -0.5
        assert_abs_diff_eq!(output[0], -0.5, epsilon = 0.0001);
        // y[1] = c*(x[1] - y[0]) + x[0] = -0.5*(0 - (-0.5)) + 1.0 = -0.25 + 1.0 = 0.75
        assert_abs_diff_eq!(output[1], 0.75, epsilon = 0.0001);
    }

    #[test]
    fn test_reset() {
        let config = Config { c: 0.5_f32 };
        let mut filter = Allpass::with_config(config);

        // Feed some input to populate state
        filter.filter(1.0);
        filter.filter(2.0);

        // Reset and verify state is cleared
        let mut reset_filter = filter.reset();
        assert_abs_diff_eq!(reset_filter.state_mut().prev_input, 0.0, epsilon = 0.0001);
        assert_abs_diff_eq!(reset_filter.state_mut().prev_output, 0.0, epsilon = 0.0001);
    }

    #[test]
    fn smoke() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        // input[0] is non-zero so a 1-sample delay is immediately distinguishable from a
        // 2-sample delay: expected[0]=0 (initial state), expected[1]=input[0]=1.
        let filter = Allpass::with_config(Config { c: 0.0 });
        let input = [
            1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
            20.0, 20.0, 7.0, 0.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        let expected = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }
}
