// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Hamming window.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Hamming window's configuration with precomputed weights.
///
/// Weights use the original 0.54 / 0.46 coefficients (as opposed to the
/// optimised 0.53836 / 0.46164 variant). Optimised coefficients improve
/// scalloping loss by moving the first null; the original coefficients are
/// used here to match the classic definition.
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
        use crate::filters::fir::convolve::windowed_sinc::hamming_window;
        assert!(N > 0, "Hamming: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = hamming_window::<T>(k, N);
        }
        Self { weights }
    }
}

/// The Hamming window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A Hamming window.
///
/// Each tap coefficient `w[k] = 0.54 − 0.46 · cos(2πk / (N−1))`, using the
/// original coefficients from R. W. Hamming's 1977 *Digital Filters*.
/// Sidelobe attenuation is approximately −42.5 dB.
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
/// use signalo::filters::fir::window::hamming::{Config as HammingConfig, Hamming};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = Hamming::<f32, 4>::with_config(HammingConfig::<f32, 4>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.08).abs() < 1e-5); // w[0] = 0.54 - 0.46 = 0.08
/// ```
#[derive(Clone, Debug)]
pub struct Hamming<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for Hamming<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Hamming<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Hamming<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Hamming: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for Hamming<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Hamming<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Hamming<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Hamming<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Hamming<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "Hamming: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Hamming<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Hamming<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Hamming<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Hamming<T, N>
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
