// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Numerical integration (cumulative sum) filter.
//!
//! Computes the discrete integral or cumulative sum of input signals, useful for slope
//! extraction, position tracking, and accumulation operations.
//!
//! [`Integrate`] is the minimal plain accumulator. [`KahanIntegrate`] uses
//! compensated summation for long-running floating-point accumulators.

use core::ops::{Add, Mul};

use num_traits::Zero;

use crate::{
    time::Timed,
    traits::{
        guts::{FromGuts, HasGuts, IntoGuts},
        Filter, Reset, State as StateTrait, StateMut,
    },
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

pub mod kahan;

mod trapezoidal;

pub use kahan::KahanIntegrate;

pub use self::trapezoidal::Trapezoidal;

/// The integration filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current value.
    pub value: T,
}

/// A integration filter that produces the integral of the signal.
///
/// # Complexity
///
/// - **Time per sample:** O(1); one addition.
/// - **Space:** O(1); stores one running sum.
#[derive(Clone, Debug)]
pub struct Integrate<T> {
    state: State<T>,
}

impl<T> Default for Integrate<T>
where
    T: Zero,
{
    fn default() -> Self {
        let state = {
            let value = T::zero();
            State { value }
        };
        Self { state }
    }
}

impl<T> StateTrait for Integrate<T> {
    type State = State<T>;
}

impl<T> StateMut for Integrate<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Integrate<T> {
    type Guts = State<T>;
}

impl<T> FromGuts for Integrate<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T> IntoGuts for Integrate<T> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T> Reset for Integrate<T>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Integrate<T> where Self: Reset {}

impl<T> Filter<T> for Integrate<T>
where
    T: Clone + Add<T, Output = T> + Zero,
{
    type Output = <T as Add<T>>::Output;

    fn filter(&mut self, input: T) -> Self::Output {
        let state = self.state.value.clone() + input;
        self.state.value = state.clone();
        state
    }
}

impl<T, Tm> Filter<Timed<T, Tm>> for Integrate<T>
where
    T: Clone + Add<T, Output = T> + Mul<Tm, Output = T> + Zero,
{
    type Output = T;

    fn filter(&mut self, input: Timed<T, Tm>) -> Self::Output {
        let contribution = input.value * input.dt;
        let state = self.state.value.clone() + contribution;
        self.state.value = state.clone();

        state
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test() {
        let filter = Integrate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(
            output.as_slice(),
            [
                0.0, 1.0, 8.0, 10.0, 15.0, 23.0, 39.0, 42.0, 61.0, 67.0, 81.0, 90.0, 99.0, 116.0,
                133.0, 137.0, 149.0, 169.0, 189.0, 196.0
            ]
            .as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn integrate_timed_rectangular_area() {
        use crate::time::Timed;
        use crate::traits::Filter;
        let mut f = Integrate::<f32>::default();
        let mut last = 0.0;
        for _ in 0..10 {
            last = f.filter(Timed::new(2.0_f32, 0.5_f32));
        }
        approx::assert_abs_diff_eq!(last, 2.0 * 10.0 * 0.5, epsilon = 1e-6); // 10.0
    }

    #[test]
    fn integrate_timed_reduces_to_running_sum() {
        use crate::time::Timed;
        use crate::traits::Filter;
        let inputs = [0.0_f32, 1.0, 7.0, 2.0, 5.0];
        let mut a = Integrate::<f32>::default();
        let mut b = Integrate::<f32>::default();
        for x in inputs {
            let via_timed = a.filter(Timed::new(x, 1.0_f32));
            let via_plain = b.filter(x);
            approx::assert_abs_diff_eq!(via_timed, via_plain, epsilon = 1e-6);
        }
    }
}
