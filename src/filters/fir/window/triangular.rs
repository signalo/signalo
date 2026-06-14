// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Triangular (Bartlett) window.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The triangular window's configuration with precomputed weights.
///
/// Weights are computed at construction time using the formula
/// `w[k] = 1 − |2k/(N−1) − 1|`.
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
        use crate::filters::fir::convolve::windowed_sinc::triangular_window;
        assert!(N > 0, "Triangular: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = triangular_window::<T>(k, N);
        }
        Self { weights }
    }
}

/// The triangular window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A triangular (Bartlett) window.
///
/// Each tap coefficient `w[k] = 1 - |2k/(N-1) - 1|`, producing a linearly-
/// tapered sequence from 0 at the endpoints to 1 at the centre.
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
/// use signalo::filters::fir::window::triangular::{Config as TriangularConfig, Triangular};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = Triangular::<f32, 4>::with_config(TriangularConfig::<f32, 4>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.0).abs() < 1e-5); // w[0] = 0
/// ```
#[derive(Clone, Debug)]
pub struct Triangular<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T: num_traits::Float, const N: usize> Default for Triangular<T, N> {
    fn default() -> Self {
        Self::with_config(Config::new())
    }
}

impl<T, const N: usize> ConfigTrait for Triangular<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Triangular<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Triangular<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Triangular: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for Triangular<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Triangular<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Triangular<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Triangular<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Triangular<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "Triangular: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Triangular<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Triangular<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Triangular<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Triangular<T, N>
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
        let mut window = Triangular::<f32, N>::default();
        let input = numeric_fixture();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

        // w[k] = [0, 2/7, 4/7, 6/7, 6/7, 4/7, 2/7, 0]
        let expected: Vec<f32> = std::vec![
            0.0,
            2.0 / 7.0,
            4.0,
            12.0 / 7.0,
            30.0 / 7.0,
            32.0 / 7.0,
            32.0 / 7.0,
            0.0,
            0.0,
            12.0 / 7.0,
            8.0,
            54.0 / 7.0,
            54.0 / 7.0,
            68.0 / 7.0,
            34.0 / 7.0,
            0.0,
            0.0,
            40.0 / 7.0,
            80.0 / 7.0,
            6.0,
            6.0,
            60.0 / 7.0,
            30.0 / 7.0,
            0.0,
            0.0,
            20.0 / 7.0,
            444.0 / 7.0,
            108.0 / 7.0,
            108.0 / 7.0,
            72.0 / 7.0,
            212.0 / 7.0,
            0.0,
            0.0,
            26.0 / 7.0,
            52.0 / 7.0,
            18.0,
            18.0,
            12.0,
            68.0 / 7.0,
            0.0,
            0.0,
            16.0 / 7.0,
            116.0 / 7.0,
            96.0 / 7.0,
            96.0 / 7.0,
            64.0 / 7.0,
            208.0 / 7.0,
            0.0,
            0.0,
            48.0 / 7.0,
        ];
        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
    }

    #[test]
    fn periodicity() {
        const N: usize = 8;
        let mut window = Triangular::<f32, N>::default();
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
        let mut window = Triangular::<f32, N>::default();

        let first_half: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output_a: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        window = window.reset();
        let output_b: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        assert_eq!(output_a, output_b);
    }

    #[test]
    fn endpoints() {
        const N: usize = 8;
        let mut window = Triangular::<f32, N>::default();

        let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

        assert_abs_diff_eq!(output[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(output[N - 1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn degenerate_n1() {
        let mut window = Triangular::<f32, 1>::default();
        assert_abs_diff_eq!(window.filter(42.0), 42.0, epsilon = 1e-5);
        assert_abs_diff_eq!(window.filter(7.0), 7.0, epsilon = 1e-5);
    }

    #[test]
    fn with_config() {
        // N=4 triangular: w = [0, 2/3, 2/3, 0]
        let mut window = Triangular::<f32, 4>::with_config(Config {
            weights: [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0],
        });
        let output = window.filter(42.0);
        assert_abs_diff_eq!(output, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn from_guts() {
        let window: Triangular<f32, 4> = FromGuts::from_guts((
            Config {
                weights: [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0],
            },
            State { k: 2 },
        ));
        let mut w = window;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn into_guts() {
        let window = Triangular::<f32, 4>::default();
        let (_config, _state) = window.into_guts();
        let restored: Triangular<f32, 4> = FromGuts::from_guts((
            Config {
                weights: [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0],
            },
            State::default(),
        ));
        let mut w = restored;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn state_mut() {
        let mut window = Triangular::<f32, 4>::default();
        let state = window.state_mut();
        assert_eq!(state.k, 0);
        state.k = 3;
        assert_eq!(window.state_mut().k, 3);
    }

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _ = Triangular::<f32, 0>::with_config(Config {
            weights: [0.0f32; 0],
        });
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn config_new_n0_panics() {
        let _ = Config::<f64, 0>::new();
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn default_zero_window_panics() {
        let _ = Triangular::<f32, 0>::default();
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn config_new_weights_match_formula() {
        const N: usize = 8;
        let config = Config::<f32, N>::new();
        for k in 0..N {
            let k_f = k as f32;
            let n_f = N as f32;
            let expected = 1.0 - (2.0 * k_f / (n_f - 1.0) - 1.0).abs();
            assert_abs_diff_eq!(config.weights[k], expected, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn triangular_parity_with_windowed_sinc() {
        use crate::filters::fir::convolve::windowed_sinc::triangular_window;

        const N: usize = 33;
        let mut window = Triangular::<f64, N>::default();

        for k in 0..N {
            let expected = triangular_window::<f64>(k, N);
            let got = window.filter(1.0);
            assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
        }
    }
}
