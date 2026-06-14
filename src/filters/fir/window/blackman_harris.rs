// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! 4-term Blackman-Harris window.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Blackman-Harris window's configuration with precomputed weights.
///
/// Weights use the 4-term coefficients `a0 = 0.35875`, `a1 = 0.48829`,
/// `a2 = 0.14128`, `a3 = 0.01168`.
///
/// # Periodicity warning
///
/// Applied periodically: the k-th tap returned is `w[k mod N]`, not tied to
/// input sample index. This means the same coefficient sequence repeats
/// every N calls.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize> {
    /// Precomputed window weights.
    pub weights: [T; N],
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(
    clippy::new_without_default,
    clippy::unwrap_used,
    clippy::missing_panics_doc
)]
impl<T: num_traits::Float, const N: usize> Config<T, N> {
    /// Create a window configuration with precomputed weights.
    #[must_use]
    pub fn new() -> Self {
        use crate::filters::fir::convolve::windowed_sinc::blackman_harris_window;
        assert!(N > 0, "BlackmanHarris: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = blackman_harris_window::<T>(k, N);
        }
        Self { weights }
    }
}

/// The Blackman-Harris window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A 4-term Blackman-Harris window.
///
/// Each tap coefficient `w[k] = a0 − a1·cos(α) + a2·cos(2α) − a3·cos(3α)`
/// with `α = 2πk/(N−1)`, `a0 = 0.35875`, `a1 = 0.48829`, `a2 = 0.14128`,
/// `a3 = 0.01168`. This is the 4-term Blackman-Harris window. Sidelobe
/// attenuation is approximately −92 dB.
///
/// # Periodicity
///
/// Applied periodically: the k-th tap returned is w[k mod N], not tied to input
/// sample index. This means the same coefficient sequence repeats every N calls.
///
/// # Complexity
///
/// - **Time per sample:** O(1)
/// - **Space:** O(N · `sizeof::<T>()`)
///
/// # Examples
///
/// ```rust
/// use signalo::filters::fir::window::blackman_harris::{Config as BlackmanHarrisConfig, BlackmanHarris};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = BlackmanHarris::<f32, 4>::with_config(
///     BlackmanHarrisConfig::<f32, 4>::new(),
/// );
/// let output = window.filter(1.0);
/// // w[0] = a0 - a1 + a2 - a3 ≈ 0.00006
/// ```
#[derive(Clone, Debug)]
pub struct BlackmanHarris<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for BlackmanHarris<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for BlackmanHarris<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for BlackmanHarris<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "BlackmanHarris: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for BlackmanHarris<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for BlackmanHarris<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for BlackmanHarris<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for BlackmanHarris<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for BlackmanHarris<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "BlackmanHarris: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for BlackmanHarris<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for BlackmanHarris<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BlackmanHarris<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for BlackmanHarris<T, N>
where
    T: Clone + core::ops::Mul<Output = T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let w = self.config.weights[self.state.k].clone();
        self.state.k = (self.state.k + 1) % N;
        input * w
    }
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    /// Numeric test fixture for smoke tests.
    fn numeric_fixture() -> Vec<f32> {
        std::vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    #[test]
    fn smoke() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        let mut window = BlackmanHarris::<f32, N>::with_config(config.clone());
        let input = numeric_fixture();
        let expected: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| x * config.weights[i % N])
            .collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
    }

    #[test]
    fn periodicity() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        let mut window = BlackmanHarris::<f32, N>::with_config(config);
        let input: Vec<f32> = std::iter::repeat(1.0).take(3 * N).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
        for block in 0..3 {
            let start = block * N;
            let end = start + N;
            assert_eq!(&output[start..end], &output[0..N]);
        }
    }

    #[test]
    fn reset_restarts_counter() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        let mut window = BlackmanHarris::<f32, N>::with_config(config);

        let first_half: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output_a: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        window = window.reset();
        let output_b: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        assert_eq!(output_a, output_b);
    }

    #[test]
    fn endpoints() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        let mut window = BlackmanHarris::<f32, N>::with_config(config.clone());

        let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

        assert_abs_diff_eq!(output[0], input[0] * config.weights[0], epsilon = 1e-5);
        assert_abs_diff_eq!(
            output[N - 1],
            input[N - 1] * config.weights[N - 1],
            epsilon = 1e-5
        );
    }

    #[test]
    fn with_config() {
        let config = Config {
            weights: [1.0f32; 4],
        };
        let mut window = BlackmanHarris::<f32, 4>::with_config(config);
        let output = window.filter(42.0);
        assert_abs_diff_eq!(output, 42.0, epsilon = 1e-5);
    }

    #[test]
    fn from_guts() {
        let config = Config {
            weights: [1.0f32; 4],
        };
        let window: BlackmanHarris<f32, 4> = FromGuts::from_guts((config, State { k: 2 }));
        let mut w = window;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn into_guts() {
        let config = Config::<f32, 4>::new();
        let window = BlackmanHarris::<f32, 4>::with_config(config);
        let (_config, _state) = window.into_guts();
        let restored: BlackmanHarris<f32, 4> = FromGuts::from_guts((
            Config {
                weights: [1.0f32; 4],
            },
            State::default(),
        ));
        let mut w = restored;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn state_mut() {
        let config = Config {
            weights: [0.0f32; 4],
        };
        let mut window = BlackmanHarris::<f32, 4>::with_config(config);
        let state = window.state_mut();
        assert_eq!(state.k, 0);
        state.k = 3;
        assert_eq!(window.state_mut().k, 3);
    }

    #[test]
    fn config_new() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        let n_minus_1 = (N - 1) as f32;
        let two_pi = 2.0 * core::f32::consts::PI;
        for k in 0..N {
            let alpha = two_pi * k as f32 / n_minus_1;
            let expected = 0.35875 - 0.48829 * alpha.cos() + 0.14128 * (2.0 * alpha).cos()
                - 0.01168 * (3.0 * alpha).cos();
            assert_abs_diff_eq!(config.weights[k], expected, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn sidelobe_attenuation() {
        const N: usize = 64;
        const ZP: usize = 8;
        const L: usize = N * ZP;

        let config = Config::<f64, N>::new();
        let weights = &config.weights;

        let two_pi = core::f64::consts::PI * 2.0;
        let mut magnitude = [0.0_f64; L / 2];
        for bin in 0..(L / 2) {
            let mut re = 0.0;
            let mut im = 0.0;
            for k in 0..N {
                let theta = two_pi * bin as f64 * k as f64 / L as f64;
                re += weights[k] * theta.cos();
                im += weights[k] * -(theta.sin());
            }
            magnitude[bin] = (re * re + im * im).sqrt();
        }

        let main_peak = magnitude[0];

        let first_null = 4 * L / N;
        let mut side_peak = 0.0_f64;
        for bin in (first_null + 1)..magnitude.len() {
            if magnitude[bin] > side_peak {
                side_peak = magnitude[bin];
            }
        }

        let sidelobe_db = 20.0 * (side_peak / main_peak).log10();
        let documented = -92.0;

        assert!(
            (sidelobe_db - documented).abs() < 3.0,
            "Sidelobe {} dB not within ±3 dB of documented {} dB",
            sidelobe_db,
            documented
        );
    }

    #[test]
    fn n_eq_1() {
        let config = Config::<f32, 1>::new();
        assert_abs_diff_eq!(config.weights[0], 1.0, epsilon = 1e-7);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn blackman_harris_parity_with_windowed_sinc() {
        use crate::filters::fir::convolve::windowed_sinc::blackman_harris_window;

        const N: usize = 33;
        let config = Config::<f64, N>::new();
        let win_fn = blackman_harris_window::<f64>;

        for k in 0..N {
            let expected = win_fn(k, N);
            let got = config.weights[k];
            assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn blackman_harris_parity_with_windowed_sinc_n2() {
        use crate::filters::fir::convolve::windowed_sinc::blackman_harris_window;

        const N: usize = 2;
        let config = Config::<f64, N>::new();
        let win_fn = blackman_harris_window::<f64>;

        for k in 0..N {
            let expected = win_fn(k, N);
            let got = config.weights[k];
            assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _ = BlackmanHarris::<f32, 0>::with_config(Config {
            weights: [0.0f32; 0],
        });
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn config_new_n0_panics() {
        let _ = Config::<f64, 0>::new();
    }
}
