// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Trapezoidal-rule integration for irregularly-timed samples.

use core::ops::{Add, Mul};

use num_traits::{FromPrimitive, Zero};

use crate::time::Timed;
use crate::traits::Filter;

/// Trapezoidal-rule integrator for irregularly-timed samples: `sum += ½·(prev+value)·dt`.
#[derive(Clone, Debug)]
pub struct Trapezoidal<T> {
    sum: T,
    prev: Option<T>,
    half: T,
}

impl<T> Default for Trapezoidal<T>
where
    T: FromPrimitive + Zero,
{
    fn default() -> Self {
        Self {
            sum: T::zero(),
            prev: None,
            // `FromPrimitive` yields the ½ factor for any scalar type without
            // requiring the full `Float` bound, keeping integer-capable `T` open.
            half: T::from_f64(0.5).expect("0.5 is representable"),
        }
    }
}

impl<T, Tm> Filter<Timed<T, Tm>> for Trapezoidal<T>
where
    T: Clone + Add<T, Output = T> + Mul<T, Output = T> + Mul<Tm, Output = T> + FromPrimitive,
{
    type Output = T;

    fn filter(&mut self, input: Timed<T, Tm>) -> Self::Output {
        let Timed { value, dt } = input;

        // The first sample has no predecessor, so it seeds `prev` and adds a
        // zero-width trapezoid. Subsequent samples add ½·(prev+value)·dt.
        if let Some(prev) = self.prev.take() {
            let area = self.half.clone() * (prev + value.clone()) * dt;
            self.sum = self.sum.clone() + area;
        }

        self.prev = Some(value);

        self.sum.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoidal_integrates_linear_ramp_exactly() {
        let mut f = Trapezoidal::<f32>::default();
        let dt = 0.25_f32;
        let mut last = 0.0;
        let mut t = 0.0_f32;
        for _ in 0..20 {
            last = f.filter(Timed::new(t, dt));
            t += dt;
        }
        let t_end = 19.0 * dt; // last value fed
        approx::assert_abs_diff_eq!(last, t_end * t_end / 2.0, epsilon = 1e-4);
    }
}
