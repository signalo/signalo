// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Approximated exponential moving median filters.

use num_traits::Num;

use signalo_traits::filter::Filter;

use signalo_traits::{Configurable, InitialState, Resettable, Stateful, StatefulUnsafe};

use filter::mean::exp::mean::{Config as MeanConfig, Mean};

/// The median filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    pub alpha: T,
    pub beta: T,
    pub gamma: T,
}

/// The median filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    mean: (Mean<T>, Mean<T>),
    median: Option<T>,
}

/// A median filter producing the approximated exponential moving median over a given signal.
#[derive(Clone, Debug)]
pub struct Median<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Median<T>
where
    T: Clone,
{
    /// Creates a new `Median` filter with given `alpha`, `beta` and `gamma` coefficients.
    ///
    /// Note: `alpha`, `beta` and `gamma` are required to be in the range between `0.0` and `1.0`.
    ///
    /// Recommended values:
    /// - `alpha`: `beta`
    /// - `beta`: `0.0 .. 1.0`
    /// - `gamma`: `beta * 0.5`
    #[inline]
    pub fn new(config: Config<T>) -> Self {
        let state = Self::initial_state(&config);
        Median { config, state }
    }
}

impl<T> Configurable for Median<T> {
    type Config = Config<T>;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> Stateful for Median<T> {
    type State = State<T>;
}

unsafe impl<T> StatefulUnsafe for Median<T> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<'a, T> InitialState<&'a Config<T>> for Median<T>
where
    T: Clone,
{
    fn initial_state(config: &'a Config<T>) -> Self::State {
        let mean = (
            Mean::new(MeanConfig {
                beta: config.alpha.clone(),
            }),
            Mean::new(MeanConfig {
                beta: config.gamma.clone(),
            }),
        );
        let median = None;
        State { mean, median }
    }
}

impl<T> Resettable for Median<T>
where
    T: Clone,
{
    fn reset(&mut self) {
        self.state = Self::initial_state(self.config());
    }
}

impl<T> Filter<T> for Median<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // We calculate the mean and use it as an estimate of the median:
        let mean = self.state.mean.0.filter(input.clone());
        // We then calculate the approximate of the median:
        let median = match &self.state.median {
            None => mean,
            Some(ref state) => state.clone() + ((mean - state.clone()) * self.config.beta.clone()),
        };
        // The approximated median tends to oscillate,
        // so we apply another mean to smoothen those out:
        let state = self.state.mean.1.filter(median);
        // And we're done. Store a copy and return the result:
        self.state.median = Some(state.clone());
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.063, 0.523, 0.817, 1.207, 1.803, 2.950, 3.456, 4.648, 5.254, 6.066, 6.605,
            6.990, 7.784, 8.708, 8.817, 9.064, 9.856, 10.836, 11.025, 10.856, 11.042, 11.370,
            11.428, 12.177, 12.368, 18.616, 21.311, 22.284, 22.441, 27.733, 28.627, 28.854, 27.962,
            26.637, 25.705, 25.003, 24.446, 24.799, 23.904, 28.831, 29.684, 30.015, 29.284, 28.133,
            26.872, 31.140, 31.749, 31.531, 30.965,
        ]
    }

    #[test]
    fn test() {
        let filter = Median::new(Config {
            alpha: 0.5,
            beta: 0.5,
            gamma: 0.25,
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
