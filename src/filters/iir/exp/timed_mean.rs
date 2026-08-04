// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Time-constant exponential moving average for irregularly-timed samples.

use crate::time::Timed;
use crate::traits::Filter;

/// Exponential moving average for irregularly-timed samples, parameterized by a time constant.
///
/// Per sample the smoothing factor is `alpha = 1 − exp(−dt / tau)`.
#[derive(Clone, Debug)]
pub struct TimedMean<T> {
    tau: T,
    mean: Option<T>,
}

impl<T> TimedMean<T> {
    /// Creates an EMA with time constant `tau` (same units as the samples' `dt`).
    pub fn from_time_constant(tau: T) -> Self {
        Self { tau, mean: None }
    }
}

impl<T> Filter<Timed<T, T>> for TimedMean<T>
where
    T: num_traits::Float,
{
    type Output = T;

    fn filter(&mut self, input: Timed<T, T>) -> Self::Output {
        let Timed { value, dt } = input;

        // The first sample seeds the mean directly; there is no prior estimate
        // to blend against. Subsequent samples blend toward `value` by the
        // per-sample smoothing factor derived from the elapsed `dt`.
        let mean = match self.mean.take() {
            Some(mean) => {
                let alpha = crate::time::smoothing_alpha(dt, self.tau);

                mean + (value - mean) * alpha
            }
            None => value,
        };

        self.mean = Some(mean);

        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_mean_step_response_at_tau() {
        let tau = 1.0_f32;
        let dt = 0.001_f32;
        let mut f = TimedMean::from_time_constant(tau);
        let _ = f.filter(Timed::new(0.0_f32, dt)); // seed at 0
        let mut y = 0.0;
        for _ in 0..((tau / dt) as usize) {
            y = f.filter(Timed::new(1.0_f32, dt));
        }
        approx::assert_abs_diff_eq!(y, 1.0 - core::f32::consts::E.recip(), epsilon = 1e-2);
    }

    #[test]
    fn timed_mean_seeds_first_sample_directly() {
        let mut f = TimedMean::from_time_constant(1.0_f32);
        let y = f.filter(Timed::new(3.5_f32, 0.25_f32));
        approx::assert_abs_diff_eq!(y, 3.5, epsilon = 1e-6);
    }

    #[test]
    fn timed_mean_is_dt_invariant_over_equal_elapsed_time() {
        // One big step of dt=1.0 vs four steps of dt=0.25 span the same elapsed time and must
        // land within a small tolerance for the same constant input.
        let tau = 2.0_f32;
        let mut coarse = TimedMean::from_time_constant(tau);
        let _ = coarse.filter(Timed::new(0.0_f32, 1.0));
        let yc = coarse.filter(Timed::new(1.0_f32, 1.0));
        let mut fine = TimedMean::from_time_constant(tau);
        let _ = fine.filter(Timed::new(0.0_f32, 1.0));
        let mut yf = 0.0;
        for _ in 0..4 {
            yf = fine.filter(Timed::new(1.0_f32, 0.25));
        }
        approx::assert_abs_diff_eq!(yc, yf, epsilon = 2e-2);
    }
}
