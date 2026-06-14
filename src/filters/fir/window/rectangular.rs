// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Rectangular (uniform / Dirichlet) window.

use core::marker::PhantomData;

use num_traits::One;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The rectangular window's configuration (zero-sized type).
///
/// The rectangular window has no parameters — every coefficient is 1.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize>(PhantomData<T>);

/// The rectangular window's state (unit struct — no runtime tracking needed).
#[derive(Clone, Debug, Default)]
pub struct State;

/// A rectangular (uniform / Dirichlet) window.
///
/// Every tap coefficient `w[k]` = 1, making this mathematically equivalent to
/// applying no window at all. It exists primarily as a base case for generic
/// code and for testing the window infrastructure.
///
/// # Periodicity
///
/// Applied periodically: the k-th tap returned is w[k mod N], not tied to input
/// sample index. This means the same coefficient sequence repeats every N calls.
///
/// # Complexity
///
/// - **Time per sample:** O(1)
/// - **Space:** O(1)
///
/// # Examples
///
/// ```rust
/// use signalo::filters::fir::window::rectangular::Rectangular;
/// use signalo::traits::Filter;
///
/// let mut window = Rectangular::<f32, 4>::default();
/// let output = window.filter(1.0);
/// assert_eq!(output, 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct Rectangular<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> Default for Rectangular<T, N> {
    fn default() -> Self {
        Self::with_config(Config(PhantomData))
    }
}

impl<T, const N: usize> ConfigTrait for Rectangular<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Rectangular<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Rectangular<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Rectangular: window size N must be > 0");
        Self {
            config,
            state: State,
        }
    }
}

impl<T, const N: usize> ConfigRef for Rectangular<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Rectangular<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Rectangular<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Rectangular<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Rectangular<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Rectangular<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Rectangular<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Rectangular<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Rectangular<T, N>
where
    T: Clone + One,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        input * T::one()
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
        let mut window = Rectangular::<f32, N>::default();
        let input = numeric_fixture();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
        assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-5);
    }

    #[test]
    fn periodicity() {
        const N: usize = 8;
        let mut window = Rectangular::<f32, N>::default();
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
        let mut window = Rectangular::<f32, N>::default();

        let first_half: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output_a: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        window = window.reset();
        let output_b: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

        assert_eq!(output_a, output_b);
    }

    #[test]
    fn endpoints() {
        const N: usize = 8;
        let mut window = Rectangular::<f32, N>::default();

        let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
        let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

        assert_abs_diff_eq!(output[0], input[0], epsilon = 1e-5);
        assert_abs_diff_eq!(output[N - 1], input[N - 1], epsilon = 1e-5);
    }

    #[test]
    fn with_config() {
        let mut window = Rectangular::<f32, 4>::with_config(Config(core::marker::PhantomData));
        let output = window.filter(42.0);
        assert_abs_diff_eq!(output, 42.0, epsilon = 1e-5);
    }

    #[test]
    fn from_guts() {
        let window: Rectangular<f32, 4> =
            FromGuts::from_guts((Config(core::marker::PhantomData), State));
        let mut w = window;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn into_guts() {
        let window = Rectangular::<f32, 4>::default();
        let (_config, _state) = window.into_guts();
        let restored: Rectangular<f32, 4> =
            FromGuts::from_guts((Config(core::marker::PhantomData), State));
        let mut w = restored;
        let output = w.filter(1.0);
        assert_abs_diff_eq!(output, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn state_mut() {
        let mut window = Rectangular::<f32, 4>::default();
        let _state = window.state_mut();
    }

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _ = Rectangular::<f32, 0>::with_config(Config(core::marker::PhantomData));
    }
}
