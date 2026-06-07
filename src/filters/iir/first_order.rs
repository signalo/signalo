// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! First-order IIR filters.

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Configuration for the first-order IIR difference equation:
/// `y[n] = b0·x[n] + b1·x[n−1] − a1·y[n−1]`
///
/// `a1` follows the Audio EQ Cookbook sign convention: it is subtracted in the
/// difference equation, so a stable low-pass pole at `z = p` requires `a1 = −p`
/// (a negative value).
///
/// # Stability
///
/// The filter is stable when `|a1| < 1`. Because `a1` uses the subtractive convention,
/// a pole at `z = p` requires `a1 = −p`; stability therefore corresponds to `|a1| < 1`,
/// i.e., `p` strictly inside the unit circle.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Feedforward coefficient for current input.
    pub b0: T,
    /// Feedforward coefficient for previous input (x[n-1]).
    pub b1: T,
    /// Feedback coefficient for previous output (y[n-1]).
    ///
    /// Uses the subtractive (Audio EQ Cookbook) convention: a stable pole at `z = p`
    /// requires `a1 = −p`. Stability requires `|a1| < 1`.
    pub a1: T,
}

/// The first-order IIR filter's state.
///
/// Stores the Direct Form I delay elements. Direct Form I is used here (rather than DF2T)
/// because it makes the input and output history directly accessible as typed fields,
/// which suits library users who inspect or pre-warm filter state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Previous input sample (x[n-1]).
    pub prev_input: T,
    /// Previous output sample (y[n-1]).
    pub prev_output: T,
}

/// A first-order IIR filter.
#[derive(Clone, Debug)]
pub struct FirstOrder<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Default for Config<T>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    fn default() -> Self {
        Self {
            b0: T::one(),
            b1: T::zero(),
            a1: T::zero(),
        }
    }
}

impl<T> Default for FirstOrder<T>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigTrait for FirstOrder<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for FirstOrder<T> {
    type State = State<T>;
}

impl<T> WithConfig for FirstOrder<T>
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

impl<T> ConfigRef for FirstOrder<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for FirstOrder<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for FirstOrder<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for FirstOrder<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for FirstOrder<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for FirstOrder<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for FirstOrder<T>
where
    T: Clone + num_traits::Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for FirstOrder<T> where Self: Reset {}

impl<T> Filter<T> for FirstOrder<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // Compute: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        let output = (self.config.b0.clone() * input.clone())
            + (self.config.b1.clone() * self.state.prev_input.clone())
            - (self.config.a1.clone() * self.state.prev_output.clone());

        self.state.prev_input = input;
        self.state.prev_output = output.clone();

        output
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

    use super::*;

    #[test]
    fn test_lowpass() {
        // First-order lowpass filter with alpha=0.5
        // Standard form: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        // Rewritten as: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        // where: b0 = alpha, b1 = 0, a1 = -(1-alpha) = alpha-1
        let alpha = 0.5;
        let config = Config {
            b0: alpha,
            b1: 0.0,
            a1: alpha - 1.0,
        };
        let filter = FirstOrder::with_config(config);

        let input = [1.0, 1.0, 1.0];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, vec![0.5, 0.75, 0.875]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_lowpass_dc_gain() {
        // EMA with alpha=0.5: DC gain = (b0+b1)/(1+a1) = (0.5+0)/(1+(-0.5)) = 1.0
        // Drive with DC (constant 1.0) until steady state, then verify gain is 1.0.
        let alpha = 0.5_f64;
        let mut filter = FirstOrder::with_config(Config {
            b0: alpha,
            b1: 0.0,
            a1: alpha - 1.0,
        });

        let mut output = 0.0_f64;

        for _ in 0..2000 {
            output = filter.filter(1.0);
        }

        assert!(
            (output - 1.0).abs() < 1e-6,
            "DC gain should be 1.0, got {}",
            output
        );
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_lowpass_nyquist_gain() {
        // EMA with alpha=0.5: gain at Nyquist = (b0 - b1)/(1 - a1) = 0.5/(1+0.5) = 1/3
        // Drive with alternating ±1 until steady state.
        let alpha = 0.5_f64;
        let mut filter = FirstOrder::with_config(Config {
            b0: alpha,
            b1: 0.0,
            a1: alpha - 1.0,
        });

        let n_total = 2000_usize;
        let n_steady = 500_usize;
        let mut rms_in = 0.0_f64;
        let mut rms_out = 0.0_f64;

        for n in 0..n_total {
            let x = if n % 2 == 0 { 1.0_f64 } else { -1.0 };
            let y = filter.filter(x);

            if n >= n_total - n_steady {
                rms_in += x * x;
                rms_out += y * y;
            }
        }

        let gain = (rms_out / rms_in).sqrt();
        let expected = 1.0 / 3.0;
        assert!(
            (gain - expected).abs() < 1e-3,
            "Nyquist gain should be 1/3, got {}",
            gain
        );
    }

    #[test]
    fn smoke() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let filter = FirstOrder::with_config(Config {
            b0: 1.0,
            b1: 0.0,
            a1: 0.0,
        });
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        assert_nearly_eq!(output, input.to_vec(), 1e-6);
    }
}
