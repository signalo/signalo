// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Time-constant peak-hold filter for irregularly-timed samples.

use num_traits::{Float, Signed};

use crate::time::{decay_factor, Timed};
use crate::traits::Filter;

/// A peak hold sink for irregularly-timed samples, parameterized by a time constant.
///
/// Tracks the maximum absolute value of a signal, decaying the held peak by
/// `exp(−dt / tau)` per sample so that the decay depends on the elapsed time
/// `dt` rather than on a fixed per-sample factor. With no new peaks, the held
/// value decays to `1/e` of its magnitude after an elapsed time of `tau`.
#[derive(Clone, Debug)]
pub struct TimedPeakHold<T> {
    tau: T,
    peak: Option<T>,
}

impl<T> TimedPeakHold<T> {
    /// Creates a peak hold sink with time constant `tau` (same units as the samples' `dt`).
    ///
    /// Called as `TimedPeakHold::from_time_constant(tau)` instead of
    /// `value.from_time_constant()`.
    pub fn from_time_constant(tau: T) -> Self {
        Self { tau, peak: None }
    }
}

impl<T> Filter<Timed<T, T>> for TimedPeakHold<T>
where
    T: Float + Signed,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: Timed<T, T>) -> Self::Output {
        let Timed { value, dt } = input;

        let abs_input = value.abs();

        // The first sample seeds the peak directly; there is no prior peak to
        // decay against. Subsequent samples decay the held peak by the elapsed
        // time and take the larger of the decayed peak and the new magnitude.
        let new_peak = match self.peak {
            Some(peak) => {
                let decayed = peak * decay_factor(dt, self.tau);

                if abs_input >= decayed {
                    abs_input
                } else {
                    decayed
                }
            }
            None => abs_input,
        };

        self.peak = Some(new_peak);

        new_peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_peak_hold_decays_by_one_over_e_at_tau() {
        let tau = 1.0_f32;
        let dt = 0.001_f32;
        let mut p = TimedPeakHold::from_time_constant(tau);
        let _ = p.filter(Timed::new(1.0_f32, dt)); // establish peak at 1.0
        let mut y = 1.0;
        for _ in 0..((tau / dt) as usize) {
            y = p.filter(Timed::new(0.0_f32, dt));
        }
        approx::assert_abs_diff_eq!(y, core::f32::consts::E.recip(), epsilon = 1e-2);
    }

    #[test]
    fn timed_peak_hold_seeds_first_sample_to_abs_value() {
        let mut p = TimedPeakHold::from_time_constant(1.0_f32);
        let y = p.filter(Timed::new(-3.0_f32, 0.5_f32));
        approx::assert_abs_diff_eq!(y, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn timed_peak_hold_new_peak_overrides_decay() {
        let mut p = TimedPeakHold::from_time_constant(1.0_f32);
        let _ = p.filter(Timed::new(1.0_f32, 0.001_f32));
        let y = p.filter(Timed::new(5.0_f32, 1.0_f32)); // larger magnitude wins over decay
        approx::assert_abs_diff_eq!(y, 5.0, epsilon = 1e-6);
    }
}
