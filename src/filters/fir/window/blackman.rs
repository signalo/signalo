// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Classical (truncated 3-term) Blackman window.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Blackman window's configuration with precomputed weights.
///
/// Weights use the classical (truncated) 3-term Blackman form rather than
/// the "exact Blackman" form. The exact Blackman adjusts the coefficients
/// to force the first and last samples to zero (`a0 = 7938/18608 ≈ 0.42659`);
/// the classical form keeps the simpler `a0 = 0.42` at the cost of a slight
/// discontinuity at the endpoints.
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
        use crate::filters::util::window::blackman;
        assert!(N > 0, "Blackman: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = blackman::<T>(k, N);
        }
        Self { weights }
    }
}

/// The Blackman window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A classical (truncated 3-term) Blackman window.
///
/// Each tap coefficient `w[k] = 0.42 − 0.5·cos(α) + 0.08·cos(2α)` with
/// `α = 2πk/(N−1)`. This is the classical three-term Blackman window
/// (not the "exact Blackman" variant). Sidelobe attenuation is approximately
/// −58 dB.
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
/// use signalo::filters::fir::window::blackman::{Config as BlackmanConfig, Blackman};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = Blackman::<f32, 4>::with_config(BlackmanConfig::<f32, 4>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.0).abs() < 1e-5); // w[0] = 0
/// ```
#[derive(Clone, Debug)]
pub struct Blackman<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for Blackman<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Blackman<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Blackman<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Blackman: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for Blackman<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Blackman<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Blackman<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Blackman<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Blackman<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "Blackman: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Blackman<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Blackman<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Blackman<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Blackman<T, N>
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
