// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kalman filters.

use num_traits::{Num, One, Zero};

use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, guts::{FromGuts, HasGuts, IntoGuts}, Reset,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

/// The kalman filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Process noise covariance
    pub r: T,
    /// Measurement noise covariance
    pub q: T,
    /// State transition
    pub a: T,
    /// Control transition
    pub b: T,
    /// Measurement
    pub c: T,
}

impl<T> Default for Config<T>
where
    T: Zero + One,
{
    fn default() -> Self {
        Self {
            r: T::one(),
            q: T::one(),
            a: T::one(),
            b: T::zero(),
            c: T::one(),
        }
    }
}

/// The kalman filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Covariance (uncertainty)
    pub cov: T,
    /// Value estimation
    pub value: Option<T>,
}

/// A 1-dimensional kalman filter.
#[derive(Clone, Debug)]
pub struct Kalman<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Kalman<T>
where
    T: Clone + Num,
{
    fn process(&mut self, (input, control): (T, T)) -> T {
        let Config {
            ref r,
            ref q,
            ref a,
            ref b,
            ref c,
        } = self.config;
        let (value, cov) = {
            let State { ref cov, ref value } = self.state;
            let c_squared = c.clone() * c.clone();
            match value {
                None => {
                    let new_value = input / c.clone();
                    let new_cov = q.clone() / c_squared;
                    (new_value, new_cov)
                }
                Some(ref value) => {
                    // Compute prediction:
                    let pred_state = (a.clone() * value.clone()) + (b.clone() * control);
                    let pred_cov = (a.clone() * cov.clone() * a.clone()) + r.clone();

                    // Compute Kalman gain:
                    let gain =
                        pred_cov.clone() * c.clone() / ((pred_cov.clone() * c_squared) + q.clone());

                    // Correction:
                    let new_value =
                        pred_state.clone() + gain.clone() * (input - (c.clone() * pred_state));
                    let new_cov = pred_cov.clone() - (gain * c.clone() * pred_cov);
                    (new_value, new_cov)
                }
            }
        };
        self.state.value = Some(value.clone());
        self.state.cov = cov;
        value
    }
}

impl<T> Default for Kalman<T>
where
    T: Zero + One,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigTrait for Kalman<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Kalman<T> {
    type State = State<T>;
}

impl<T> WithConfig for Kalman<T>
where
    T: Zero,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let cov = T::zero();
            let value = None;
            State { cov, value }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Kalman<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Kalman<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Kalman<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Kalman<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Kalman<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Kalman<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Kalman<T>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Kalman<T> where Self: Reset {}

impl<T> Filter<T> for Kalman<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.process((input, T::zero()))
    }
}

impl<T> Filter<(T, T)> for Kalman<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, (input, control): (T, T)) -> Self::Output {
        self.process((input, control))
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

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
            0.000, 0.524, 3.012, 2.682, 3.375, 4.693, 7.837, 6.510, 9.912, 8.851, 10.245, 9.908,
            9.663, 11.646, 13.092, 10.636, 11.004, 13.435, 15.208, 12.991, 11.372, 12.352, 13.068,
            12.239, 15.146, 13.756, 40.027, 34.076, 29.733, 26.563, 48.024, 36.401, 33.591, 28.028,
            23.968, 23.166, 22.581, 22.154, 25.354, 20.666, 44.530, 34.661, 33.132, 28.503, 25.126,
            22.660, 44.635, 35.548, 32.428, 30.151,
        ]
    }

    #[test]
    fn test() {
        let filter = Kalman::with_config(Config {
            r: 0.0001, // Process noise
            q: 0.001,  // Measurement noise
            a: 1.0,    // State
            b: 0.0,    // Control
            c: 1.0,    // Measurement
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
