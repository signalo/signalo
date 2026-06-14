// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kaiser window.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

#[cfg(any(feature = "libm", feature = "std"))]
pub(crate) use crate::math::bessel_i0;

/// The Kaiser window's configuration with precomputed weights and beta parameter.
///
/// Weights are computed at construction time using the formula
/// `w[k] = I₀(β·√(1 − (2k/(N−1) − 1)²)) / I₀(β)`.
///
/// # Periodicity warning
///
/// Applied periodically: the k-th tap returned is `w[k mod N]`, not tied to
/// input sample index. This means the same coefficient sequence repeats
/// every N calls.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize> {
    /// Shape parameter β (higher = stronger attenuation, wider main lobe).
    pub beta: T,
    /// Precomputed window weights.
    pub weights: [T; N],
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
impl<T: Float + core::fmt::Debug, const N: usize> Config<T, N> {
    /// Create a Kaiser window configuration with the given shape parameter β.
    ///
    /// A larger β produces stronger sidelobe attenuation and a wider main lobe.
    /// Typical values: 5.0–8.0 for general-purpose use (~50–70 dB attenuation).
    ///
    /// When β = 0 the Kaiser window reduces to a rectangular (Dirichlet) window
    /// because `I₀(0) = 1` and all weights equal 1. This is mathematically
    /// consistent but may be surprising; see [`Config::beta_for_attenuation`]
    /// to choose β from a desired stopband attenuation.
    ///
    /// # Panics
    ///
    /// Panics if β is negative or NaN, or if N is 0.
    #[must_use]
    pub fn new(beta: T) -> Self {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        assert!(N > 0, "Kaiser: window size N must be > 0");
        let one = T::one();
        let mut weights = [T::zero(); N];
        if N == 1 {
            weights[0] = one;
            return Self { beta, weights };
        }
        let i0_beta = bessel_i0(beta);
        let n_minus_1 = T::from(N - 1).unwrap();
        let two = T::from(2.0).unwrap();
        for (k, weight) in weights.iter_mut().enumerate() {
            let k_f = T::from(k).unwrap();
            let arg = two * k_f / n_minus_1 - one;
            let root = (one - arg * arg).max(T::zero()).sqrt();
            *weight = bessel_i0(beta * root) / i0_beta;
        }
        Self { beta, weights }
    }

    /// Compute Kaiser β for a desired sidelobe attenuation in dB.
    ///
    /// Uses the Kaiser empirical formula:
    /// - A ≤ 21 dB → β = 0
    /// - 21 < A < 50 → β = 0.5842·(A−21)^0.4 + 0.07886·(A−21)
    /// - A ≥ 50 → β = 0.1102·(A−8.7)
    ///
    /// The two analytic arms join with a small (~0.02) discontinuity at
    /// `A = 50` dB; this is inherent to Kaiser & Reed's empirical fit and
    /// not a defect.
    #[must_use]
    pub fn beta_for_attenuation(atten_db: T) -> T {
        let zero = T::zero();
        let twenty_one = T::from(21.0).unwrap();
        let fifty = T::from(50.0).unwrap();

        if atten_db <= twenty_one {
            zero
        } else if atten_db < fifty {
            let a_minus_21 = atten_db - twenty_one;
            T::from(0.5842).unwrap() * a_minus_21.powf(T::from(0.4).unwrap())
                + T::from(0.07886).unwrap() * a_minus_21
        } else {
            T::from(0.1102).unwrap() * (atten_db - T::from(8.7).unwrap())
        }
    }
}

/// The Kaiser window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A Kaiser window.
///
/// Each tap coefficient `w[k] = I₀(β·√(1 − (2k/(N−1) − 1)²)) / I₀(β)`,
/// producing a flexible window parametrized by β that smoothly trades off
/// main-lobe width against sidelobe attenuation.
///
/// The Kaiser window is near-optimal in the sense of maximising energy in the
/// main lobe for a given sidelobe level.
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
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::kaiser::{Config as KaiserConfig, Kaiser};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = Kaiser::<f64, 4>::with_config(KaiserConfig::<f64, 4>::new(6.0));
/// let output = window.filter(1.0);
/// assert!(output > 0.0);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Kaiser<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for Kaiser<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Kaiser<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Kaiser<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for Kaiser<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Kaiser<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Kaiser<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Kaiser<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Kaiser<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Kaiser<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Kaiser<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Kaiser<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Kaiser<T, N>
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

    fn numeric_fixture() -> Vec<f32> {
        std::vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn bessel_i0_zero() {
        assert_abs_diff_eq!(bessel_i0(0.0f64), 1.0, epsilon = 1e-15);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn bessel_i0_one() {
        assert_abs_diff_eq!(bessel_i0(1.0f64), 1.266065877752008, epsilon = 1e-12);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn bessel_i0_ten() {
        assert_abs_diff_eq!(bessel_i0(10.0f64), 2815.716628466254, epsilon = 1e-10);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn kaiser_n_eq_1() {
        let config = Config::<f32, 1>::new(6.0);
        assert_abs_diff_eq!(config.weights[0], 1.0, epsilon = 1e-7);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn kaiser_n_eq_0_panics() {
        let _ = Config::<f64, 0>::new(6.0);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn config_new_n0_panics() {
        let _ = Config::<f64, 0>::new(6.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn smoke() {
        const N: usize = 8;
        let beta = 1.0f32;
        let config = Config::<f32, N>::new(beta);
        let mut window = Kaiser::<f32, N>::with_config(config.clone());
        let input = numeric_fixture();
        let expected: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| x * config.weights[i % N])
            .collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn periodicity() {
        const N: usize = 8;
        let beta = 1.0f32;
        let config = Config::<f32, N>::new(beta);
        let mut window = Kaiser::<f32, N>::with_config(config);
        let input: Vec<f32> = std::iter::repeat(1.0).take(3 * N).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
        for block in 0..3 {
            let start = block * N;
            let end = start + N;
            assert_eq!(&output[start..end], &output[0..N]);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn reset_restarts_counter() {
        const N: usize = 8;
        let beta = 1.0f32;
        let config = Config::<f32, N>::new(beta);
        let mut window = Kaiser::<f32, N>::with_config(config);

        let first_half: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output_a: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        window = window.reset();
        let output_b: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        assert_eq!(output_a, output_b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn endpoints() {
        const N: usize = 8;
        let beta = 1.0f32;
        let config = Config::<f32, N>::new(beta);
        let mut window = Kaiser::<f32, N>::with_config(config.clone());

        let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

        assert_abs_diff_eq!(output[0], input[0] * config.weights[0], epsilon = 1e-5);
        assert_abs_diff_eq!(
            output[N - 1],
            input[N - 1] * config.weights[N - 1],
            epsilon = 1e-5
        );
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn beta_for_attenuation_40() {
        let beta = Config::<f64, 8>::beta_for_attenuation(40.0);
        assert_abs_diff_eq!(beta, 3.3953, epsilon = 1e-4);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn beta_for_attenuation_60() {
        let beta = Config::<f64, 8>::beta_for_attenuation(60.0);
        assert_abs_diff_eq!(beta, 5.65326, epsilon = 1e-4);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn beta_for_attenuation_boundary_21() {
        // Attenuation = 21 dB is the boundary of the zero-return region.
        let beta = Config::<f64, 8>::beta_for_attenuation(21.0);
        assert_abs_diff_eq!(beta, 0.0, epsilon = 1e-12);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn beta_for_attenuation_boundary_50() {
        // Attenuation = 50 dB is the boundary between mid- and high-attenuation formulas.
        let beta = Config::<f64, 8>::beta_for_attenuation(50.0);
        // Expected from formula: 0.1102 * (50.0 - 8.7) = 0.1102 * 41.3 = 4.55126
        assert_abs_diff_eq!(beta, 4.55126, epsilon = 1e-3);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn beta_for_attenuation_below_21_returns_zero() {
        let beta = Config::<f64, 8>::beta_for_attenuation(10.0);
        assert_abs_diff_eq!(beta, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn with_config() {
        let config = Config {
            beta: 6.0f32,
            weights: [1.0f32; 4],
        };
        let mut window = Kaiser::<f32, 4>::with_config(config);
        let output = window.filter(42.0);
        assert_abs_diff_eq!(output, 42.0, epsilon = 1e-5);
    }

    #[test]
    fn from_guts() {
        let config = Config {
            beta: 6.0f32,
            weights: [1.0f32; 4],
        };
        let window: Kaiser<f32, 4> = FromGuts::from_guts((config, State { k: 2 }));
        let mut w = window;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 1.0, epsilon = 1e-5);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn into_guts() {
        let beta = 6.0f32;
        let config = Config::<f32, 4>::new(beta);
        let window = Kaiser::<f32, 4>::with_config(config);
        let (_config, _state) = window.into_guts();
        let restored: Kaiser<f32, 4> = FromGuts::from_guts((
            Config {
                beta: 6.0f32,
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
            beta: 6.0f32,
            weights: [0.0f32; 4],
        };
        let mut window = Kaiser::<f32, 4>::with_config(config);
        let state = window.state_mut();
        assert_eq!(state.k, 0);
        state.k = 3;
        assert_eq!(window.state_mut().k, 3);
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn sidelobe_attenuation() {
        const N: usize = 64;
        const ZP: usize = 8;
        const L: usize = N * ZP;

        let config = Config::<f64, N>::new(6.0);
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

        let first_null = 5 * L / N;
        let mut side_peak = 0.0_f64;
        for bin in (first_null + 1)..magnitude.len() {
            if magnitude[bin] > side_peak {
                side_peak = magnitude[bin];
            }
        }

        let sidelobe_db = 20.0 * (side_peak / main_peak).log10();
        let documented = -57.0;

        assert!(
            (sidelobe_db - documented).abs() < 3.0,
            "Sidelobe {} dB not within ±3 dB of documented {} dB",
            sidelobe_db,
            documented
        );
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn kaiser_weights_consistency() {
        const N: usize = 33;
        let beta = 6.0_f64;
        let config = Config::<f64, N>::new(beta);
        let weights = config.weights;

        // Center tap must be 1.0 by definition
        let m = (N - 1) / 2;
        assert_abs_diff_eq!(weights[m], 1.0, epsilon = 1e-12);

        // Endpoint: w[0] = I₀(0) / I₀(β) = 1 / I₀(β)
        let i0_beta = bessel_i0(beta);
        assert_abs_diff_eq!(weights[0], 1.0 / i0_beta, epsilon = 1e-12);
        assert_abs_diff_eq!(weights[N - 1], 1.0 / i0_beta, epsilon = 1e-12);

        // Symmetry
        for k in 0..N / 2 {
            assert_abs_diff_eq!(weights[k], weights[N - 1 - k], epsilon = 1e-12);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn kaiser_parity_with_windowed_sinc() {
        use crate::filters::fir::convolve::windowed_sinc::kaiser_window;

        const N: usize = 33;
        let beta = 6.0_f64;

        let config = Config::<f64, N>::new(beta);
        let win_fn = kaiser_window::<f64>(beta);

        for k in 0..N {
            let expected = win_fn(k, N);
            let got = config.weights[k];
            assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn all_weights_are_finite() {
        let config = Config::<f32, 16>::new(6.0);
        for weight in &config.weights {
            assert!(weight.is_finite());
            assert!(*weight >= 0.0);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "Kaiser beta must be non-negative")]
    fn negative_beta_panics() {
        let _ = Config::<f64, 8>::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _ = Kaiser::<f32, 0>::with_config(Config {
            beta: 0.0f32,
            weights: [0.0f32; 0],
        });
    }
}
