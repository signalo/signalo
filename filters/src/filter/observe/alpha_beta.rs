// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Alpha-Beta filters.

use num_traits::{Zero, Num};

use signalo_traits::filter::Filter;

use traits::{
    InitialState,
    Resettable,
    Stateful,
    StatefulUnsafe,
};

/// A Kalman filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Velocity
    pub velocity: T,
    /// Value estimation
    pub value: Option<T>,
}

/// An Alpha-Beta filter.
#[derive(Clone, Debug)]
pub struct AlphaBeta<T> {
    /// Alpha coefficient
    alpha: T,
    /// Beta coefficient
    beta: T,
    /// State
    state: State<T>,
}

impl<T> AlphaBeta<T>
where
    T: Zero
{
    /// Creates a new `AlphaBeta` filter with given `r`, `q`, `a`, `b`, and `c` coefficients.
    ///
    /// Note: _Values of `alpha` and `beta` typically are adjusted experimentally.
    /// In general, larger alpha and beta gains tend to produce faster response
    /// for tracking transient changes, while smaller alpha and beta gains reduce
    /// the level of noise in the state estimates._
    ///
    /// Coefficients:
    /// - `alpha`: the `alpha` coefficient
    /// - `beta`: the `beta` coefficient
    #[inline]
    pub fn new(alpha: T, beta: T) -> Self {
        let state = Self::initial_state(());
        AlphaBeta {alpha, beta, state }
    }
}

impl<T> Stateful for AlphaBeta<T> {
    type State = State<T>;
}

unsafe impl<T> StatefulUnsafe for AlphaBeta<T> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> InitialState<()> for AlphaBeta<T>
where
    T: Zero,
{
    fn initial_state(_: ()) -> Self::State {
        let velocity = T::zero();
        let value = None;
        State { velocity, value }
    }
}

impl<T> Resettable for AlphaBeta<T>
where
    T: Zero,
{
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T> Filter<T> for AlphaBeta<T>
where
    T: Copy + Num
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let (velocity, state) = match (self.state.velocity, self.state.value) {
            (velocity, None) => {
                (velocity, input)
            },
            (mut velocity, Some(mut state)) => {
                // Compute prediction:
                state = state + velocity;

                // Compute residual (error):
                let residual = input - state;

                // Correction:
                state = state + (self.alpha * residual);
                velocity = velocity + (self.beta * residual);

                (velocity, state)
            },
        };
        self.state.velocity = velocity;
        self.state.value = Some(state);
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
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.500, 3.813, 3.367, 4.474, 6.593, 11.828, 8.467, 14.103, 11.034, 12.870,
            11.429, 10.405, 13.717, 15.784, 10.469, 11.003, 15.395, 18.166, 13.281, 10.053,
            12.058, 13.428, 11.809, 17.274, 14.222, 62.668, 46.433, 34.761, 26.830, 65.761,
            39.756, 32.909, 22.122, 15.588, 15.998, 16.828, 17.764, 25.137, 16.931, 62.212,
            40.201, 35.670, 26.071, 20.013, 16.482, 58.656, 38.911, 32.050, 27.613
        ]
    }

    #[test]
    fn test() {
        let alpha = 0.5;
        let beta = 0.125;
        let filter = AlphaBeta::new(alpha, beta);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();

        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
