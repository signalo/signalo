// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! DC blocker filter (high-pass filter).
//!
//! The DC blocker is a first-order high-pass filter that removes DC bias (constant offset)
//! from a signal while preserving AC components (time-varying signals).
//!
//! Difference equation:
//! `y[n] = x[n] - x[n-1] + r * y[n-1]`
//!
//! This is mathematically equivalent to `first_order::FirstOrder` with
//! `b0 = 1`, `b1 = -1`, `a1 = -r`. `DcBlocker` is implemented as a thin newtype
//! wrapper around `first_order::FirstOrder` to guarantee the two remain numerically identical.
//!
//! Higher `r` values (closer to `1.0`) result in a lower cutoff frequency: only DC and
//! frequencies extremely close to it are attenuated, leaving the rest of the spectrum
//! unaffected. Lower `r` values widen the stopband, attenuating a broader range of
//! low-frequency content. A typical value of `r = 0.995` is a good default for audio.

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::first_order;

/// The DC blocker filter's configuration.
///
/// # Stability
///
/// The filter is stable when `|r| < 1` (pole inside the unit circle). In practice,
/// restrict to `0 < r < 1` for conventional DC-blocking behavior: negative values are
/// stable but produce high gain near Nyquist rather than a high-pass response.
/// Typical audio default: `r = 0.995`.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Pole radius controlling the cutoff frequency.
    ///
    /// A value of `r = +p` places the feedback pole at `z = +p` (additive pole-radius
    /// convention). Stability requires `0 < r < 1` for conventional DC-blocking use.
    pub r: T,
}

/// State type for [`DcBlocker`], shared with [`super::first_order::FirstOrder`].
pub type State<T> = first_order::State<T>;

/// A DC blocker filter (high-pass filter).
///
/// Removes DC bias from a signal using the equation `y[n] = x[n] - x[n-1] + r·y[n-1]`.
/// Implemented internally as a `first_order::FirstOrder` with `b0=1, b1=-1, a1=-r`.
#[derive(Clone, Debug)]
pub struct DcBlocker<T> {
    config: Config<T>,
    inner: first_order::FirstOrder<T>,
}

impl<T> ConfigTrait for DcBlocker<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for DcBlocker<T> {
    type State = first_order::State<T>;
}

impl<T> WithConfig for DcBlocker<T>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let inner_config = first_order::Config {
            b0: T::one(),
            b1: T::zero() - T::one(),
            a1: T::zero() - config.r.clone(),
        };

        Self {
            config,
            inner: first_order::FirstOrder::with_config(inner_config),
        }
    }
}

impl<T> ConfigRef for DcBlocker<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for DcBlocker<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for DcBlocker<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        self.inner.state_mut()
    }
}

impl<T> HasGuts for DcBlocker<T> {
    type Guts = (Config<T>, first_order::State<T>);
}

impl<T> FromGuts for DcBlocker<T>
where
    T: Clone + Num,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let mut dc_blocker = Self::with_config(config);
        *dc_blocker.inner.state_mut() = state;

        dc_blocker
    }
}

impl<T> IntoGuts for DcBlocker<T> {
    fn into_guts(self) -> Self::Guts {
        let (_, state) = self.inner.into_guts();
        (self.config, state)
    }
}

impl<T> Reset for DcBlocker<T>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for DcBlocker<T> where Self: Reset {}

impl<T> Filter<T> for DcBlocker<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.inner.filter(input)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

    use super::*;

    #[test]
    fn test_dc_removal() {
        // DC input: constant 1.0 exhibits step response of DC blocker
        let config = Config { r: 0.995_f32 };
        let filter = DcBlocker::with_config(config);

        let input: Vec<f32> = vec![1.0; 1000];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        // First output is 1.0 (the step edge)
        assert_nearly_eq!(output[0], 1.0, 0.001);

        // After 1000 samples with R=0.995, output decays to near zero
        // R^1000 ≈ 0.0066
        assert!(output[output.len() - 1] < 0.01);
    }

    #[test]
    fn test_ac_signal_passes() {
        // Alternating signal: should pass through with minimal attenuation.
        // Steady-state amplitude at Nyquist: H(e^{jπ}) = 2/(1+R). For R=0.995: ≈ 1.0025.
        let config = Config { r: 0.995 };
        let filter = DcBlocker::with_config(config);

        let input: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        let last = output[output.len() - 1].abs();
        assert!(last > 0.99 && last < 1.05, "expected ~1.0025, got {}", last);
    }

    #[test]
    fn test_reset() {
        let config = Config { r: 0.995_f32 };
        let mut filter = DcBlocker::with_config(config);

        // Feed some input to populate state
        filter.filter(1.0);
        filter.filter(1.0);

        // Reset and verify state is cleared
        let mut reset_filter = filter.reset();
        assert_nearly_eq!(reset_filter.state_mut().prev_input, 0.0, 0.0001);
        assert_nearly_eq!(reset_filter.state_mut().prev_output, 0.0, 0.0001);
    }

    #[test]
    fn test_dc_gain_is_zero() {
        // H(1) = 0: DC input must decay to zero at steady state.
        let mut filter = DcBlocker::with_config(Config { r: 0.995_f64 });

        for _ in 0..5000 {
            filter.filter(1.0);
        }

        let output = filter.filter(1.0);
        assert!(output.abs() < 0.01, "DC gain should be 0, got {}", output);
    }

    #[test]
    fn test_nyquist_gain() {
        // H(e^{jπ}) = 2/(1+R) ≈ 1.0025 for R=0.995.
        let r = 0.995_f64;
        let expected_gain = 2.0 / (1.0 + r);
        let mut filter = DcBlocker::with_config(Config { r });

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
        assert!(
            (gain - expected_gain).abs() < 1e-3,
            "Nyquist gain should be {}, got {}",
            expected_gain,
            gain
        );
    }

    #[test]
    fn test_bit_equivalence_with_first_order() {
        // DcBlocker(r) must be bit-exact with FirstOrder(b0=1, b1=-1, a1=-r).
        let r = 0.5_f64;
        let mut dc = DcBlocker::with_config(Config { r });
        let mut fo = first_order::FirstOrder::with_config(first_order::Config {
            b0: 1.0,
            b1: -1.0,
            a1: -r,
        });

        let input = [
            1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
            20.0, 20.0, 7.0, 0.0,
        ];

        for &x in &input {
            assert_eq!(dc.filter(x), fo.filter(x));
        }
    }

    #[test]
    fn smoke() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let filter = DcBlocker::with_config(Config { r: 0.0 });
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        let expected = [
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -13.0, 16.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0,
            -13.0, 8.0, 8.0, 0.0, -13.0,
        ];
        assert_nearly_eq!(output, expected.to_vec(), 1e-6);
    }
}
