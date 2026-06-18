// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Flat-top window (SRS / Matlab `flattopwin`, 5-term).

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The flat-top window's configuration with precomputed weights.
///
/// Weights use the 5-term SRS / Matlab `flattopwin` ("Bruce Hartmann")
/// coefficients: `a0 = 0.21557895`, `a1 = 0.41663158`,
/// `a2 = 0.277263158`, `a3 = 0.083578947`, `a4 = 0.006947368`.
///
/// This window is optimised for amplitude accuracy, not frequency resolution.
/// The passband is exceptionally flat, making it the preferred choice for
/// calibration and precision amplitude measurements.
///
/// Note: IEC 61000-4-7 (harmonic measurement standard) specifies a
/// rectangular window over an integer number of mains cycles plus Hann
/// smoothing, not this 5-term flat-top filter.
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
        use crate::filters::util::window::flat_top_window;
        assert!(N > 0, "FlatTop: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = flat_top_window::<T>(k, N);
        }
        Self { weights }
    }
}

/// The flat-top window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A flat-top window (SRS / Matlab flattopwin, 5-term Bruce Hartmann).
///
/// Each tap coefficient uses the 5-term SRS / Matlab flattopwin (Bruce Hartmann) formula:
/// `w[k] = a0 − a1·cos(α) + a2·cos(2α) − a3·cos(3α) + a4·cos(4α)`
/// with `α = 2πk/(N−1)`.
///
/// This window is optimised for amplitude accuracy — the passband ripple
/// is nearly zero — rather than frequency resolution. It is the standard
/// choice for precision amplitude measurements and calibration. Sidelobe
/// attenuation is approximately −88 dB.
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
/// use signalo::filters::fir::window::flat_top::{Config as FlatTopConfig, FlatTop};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = FlatTop::<f32, 4>::with_config(FlatTopConfig::<f32, 4>::new());
/// let output = window.filter(1.0);
/// // w[0] = a0 - a1 + a2 - a3 + a4 ≈ -0.00042
/// ```
#[derive(Clone, Debug)]
pub struct FlatTop<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for FlatTop<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for FlatTop<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for FlatTop<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "FlatTop: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for FlatTop<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for FlatTop<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for FlatTop<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for FlatTop<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for FlatTop<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "FlatTop: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for FlatTop<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for FlatTop<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for FlatTop<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for FlatTop<T, N>
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
