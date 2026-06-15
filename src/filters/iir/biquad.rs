// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Biquad (second-order IIR) filters using Direct Form II Transposed topology.
//!
//! Provides a core building block for designing more complex digital filters with
//! arbitrary frequency response characteristics. Can be cascaded for higher-order responses.
//!
//! A biquad filter is a second-order infinite impulse response (IIR) filter that processes
//! signals according to the difference equation:
//!
//! ```text
//! y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//! ```
//!
//! This implementation uses the Direct Form II Transposed topology, which reduces
//! sensitivity to coefficient quantization and avoids overflow in intermediate calculations.
//!
//! A filter is stable if and only if all poles lie strictly inside the unit circle, i.e.,
//! the roots of `z^2 + a1*z + a2 = 0` have magnitude less than 1. Hand-crafted coefficients
//! that violate this condition will produce diverging (unstable) output.

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Cascaded biquad filters for higher-order IIR implementations.
pub mod cascade;

/// Coefficient factory traits for computing biquad filter coefficients from standard DSP design
/// equations (low-pass, high-pass, band-pass, notch, peaking, etc.).
#[cfg(any(feature = "libm", feature = "std"))]
pub mod coefficients;

/// The biquad filter's configuration.
///
/// Contains the five coefficients that define the biquad filter's frequency response.
/// The numerator coefficients (b0, b1, b2) control the zeros, while the denominator
/// coefficients (a1, a2) control the poles. The coefficient a0 is assumed to be 1.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Numerator coefficient b0 (feedforward)
    pub b0: T,
    /// Numerator coefficient b1 (feedforward)
    pub b1: T,
    /// Numerator coefficient b2 (feedforward)
    pub b2: T,
    /// Denominator coefficient a1 (feedback)
    pub a1: T,
    /// Denominator coefficient a2 (feedback)
    pub a2: T,
}

impl<T> From<[T; 5]> for Config<T> {
    fn from([b0, b1, b2, a1, a2]: [T; 5]) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }
}

impl<T> From<Config<T>> for [T; 5] {
    fn from(c: Config<T>) -> Self {
        [c.b0, c.b1, c.b2, c.a1, c.a2]
    }
}

impl<T> Default for Config<T>
where
    T: Num,
{
    fn default() -> Self {
        Self {
            b0: T::one(),
            b1: T::zero(),
            b2: T::zero(),
            a1: T::zero(),
            a2: T::zero(),
        }
    }
}

/// The biquad filter's state.
///
/// Contains the delay line values required for the DF2T implementation.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// First delay line value
    pub s1: T,
    /// Second delay line value
    pub s2: T,
}

impl<T> Default for State<T>
where
    T: Num,
{
    fn default() -> Self {
        Self {
            s1: T::zero(),
            s2: T::zero(),
        }
    }
}

/// A biquad (second-order IIR) filter using Direct Form II Transposed topology.
#[derive(Clone, Debug)]
pub struct Biquad<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Default for Biquad<T>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigTrait for Biquad<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Biquad<T> {
    type State = State<T>;
}

impl<T> WithConfig for Biquad<T>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T> ConfigRef for Biquad<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Biquad<T>
where
    T: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Biquad<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Biquad<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Biquad<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Biquad<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Biquad<T>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Biquad<T> where Self: Reset {}

impl<T> Filter<T> for Biquad<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        df2t_step(&self.config, &mut self.state, input)
    }
}

pub(crate) fn df2t_step<T>(config: &Config<T>, state: &mut State<T>, input: T) -> T
where
    T: Clone + Num,
{
    let output = config.b0.clone() * input.clone() + state.s1.clone();
    state.s1 =
        config.b1.clone() * input.clone() - config.a1.clone() * output.clone() + state.s2.clone();
    state.s2 = config.b2.clone() * input - config.a2.clone() * output.clone();
    output
}

pub use cascade::BiquadCascade;

#[cfg(any(feature = "libm", feature = "std"))]
pub use coefficients::Butterworth;

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_identity_coefficients() {
        use alloc::vec::Vec;

        // Identity coefficients: b0=1, b1=b2=0, a1=a2=0
        // Expected: output equals input
        let filter = Biquad::with_config(Config {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        });

        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_step_response_dc_gain() {
        // DC gain = (b0 + b1 + b2) / (1 + a1 + a2)
        // With b0=0.5, b1=0.25, b2=0.25, a1=0.0, a2=0.0 => DC gain = 1.0
        let mut filter = Biquad::with_config(Config {
            b0: 0.5_f64,
            b1: 0.25,
            b2: 0.25,
            a1: 0.0,
            a2: 0.0,
        });

        // Drive a step input long enough to reach steady state
        let mut output = 0.0;

        for _ in 0..1000 {
            output = filter.filter(1.0);
        }

        let expected_dc_gain = (0.5 + 0.25 + 0.25) / (1.0 + 0.0 + 0.0);

        assert_abs_diff_eq!(output, expected_dc_gain, epsilon = 1e-6);
    }

    #[test]
    fn test_impulse_response_matches_hand_computation() {
        // Simple 1-pole IIR: b0=1, b1=0, b2=0, a1=-0.5, a2=0
        // Impulse response: y[0]=1, y[1]=0.5, y[2]=0.25, ...
        let mut filter = Biquad::with_config(Config {
            b0: 1.0_f64,
            b1: 0.0,
            b2: 0.0,
            a1: -0.5,
            a2: 0.0,
        });

        let y0 = filter.filter(1.0);
        let y1 = filter.filter(0.0);
        let y2 = filter.filter(0.0);
        let y3 = filter.filter(0.0);

        assert_abs_diff_eq!(y0, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y1, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(y2, 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(y3, 0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_reset_clears_state() {
        let config = Config {
            b0: 1.0_f64,
            b1: 0.5,
            b2: 0.25,
            a1: -0.3,
            a2: 0.1,
        };

        let mut filter = Biquad::with_config(config.clone());

        // Drive filter to accumulate non-zero state
        for _ in 0..50 {
            filter.filter(1.0);
        }

        // State should be non-zero now
        {
            let st = filter.state_mut();
            #[allow(clippy::float_cmp)]
            let state_is_nonzero = st.s1 != 0.0 || st.s2 != 0.0;
            assert!(state_is_nonzero);
        }

        let mut filter = filter.reset();

        {
            let st = filter.state_mut();
            assert_eq!(st.s1.to_bits(), 0.0_f64.to_bits());
            assert_eq!(st.s2.to_bits(), 0.0_f64.to_bits());
        }

        // First output after reset should match a fresh filter
        let mut fresh = Biquad::with_config(config);
        assert_abs_diff_eq!(filter.filter(1.0), fresh.filter(1.0), epsilon = 1e-10);
    }
}
