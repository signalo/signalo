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
mod tests;
