// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kahan-compensated numerical integration filter.

use core::ops::{Add, Sub};

use num_traits::Zero;

use crate::math::KahanSum;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// An integration filter using Kahan compensated summation.
///
/// This variant stores an additional compensation term to reduce accumulated
/// floating-point rounding error. It is useful for long-running floating-point
/// integral paths whose state is updated continuously over long runtimes.
///
/// # Complexity
///
/// - **Time per sample:** O(1); several additions/subtractions.
/// - **Space:** O(1); stores one running sum and one compensation term.
#[derive(Clone, Debug)]
pub struct KahanIntegrate<T> {
    state: KahanSum<T>,
}

impl<T> Default for KahanIntegrate<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self {
            state: KahanSum::default(),
        }
    }
}

impl<T> StateTrait for KahanIntegrate<T> {
    type State = KahanSum<T>;
}

impl<T> StateMut for KahanIntegrate<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for KahanIntegrate<T> {
    type Guts = KahanSum<T>;
}

impl<T> FromGuts for KahanIntegrate<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T> IntoGuts for KahanIntegrate<T> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T> Reset for KahanIntegrate<T>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for KahanIntegrate<T> where Self: Reset {}

impl<T> Filter<T> for KahanIntegrate<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + Zero,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.add(input);
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use crate::filters::iir::integrate::Integrate;

    use super::*;

    #[test]
    fn kahan_integrate_matches_plain_sum_for_simple_sequence() {
        let mut filter = KahanIntegrate::default();

        assert_eq!(filter.filter(1.0_f32), 1.0);
        assert_eq!(filter.filter(2.0), 3.0);
        assert_eq!(filter.filter(3.0), 6.0);
    }

    #[test]
    fn kahan_integrate_reduces_repeated_small_increment_error() {
        let mut kahan = KahanIntegrate::<f32>::default();
        let mut plain = Integrate::<f32>::default();
        for _ in 0..10_000 {
            let _ = kahan.filter(0.1_f32);
            let _ = plain.filter(0.1);
        }

        let expected = 1_000.0_f32;
        assert!(
            (kahan.filter(0.0_f32) - expected).abs() < (plain.filter(0.0_f32) - expected).abs()
        );
    }
}
