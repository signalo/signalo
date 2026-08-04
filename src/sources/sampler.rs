// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fixed-cadence sampling of a continuous-time signal.

use crate::time::Timed;
use crate::traits::{Filter, Source};

use core::ops::Add;

/// Drives a continuous-time signal (a [`Filter`] mapping time to amplitude) at a
/// fixed cadence, emitting one sample per [`Source::source`] call.
///
/// Each call evaluates the wrapped signal at the current time, then advances the
/// clock by `step`. The clock is never clamped or normalized and no sample rate
/// is stored, so sampling proceeds single-pass with `O(1)` memory.
#[derive(Clone, Debug)]
pub struct Sampler<F, Tm = f32> {
    signal: F,
    t: Tm,
    step: Tm,
}

impl<F, Tm> Sampler<F, Tm> {
    /// Creates a sampler starting at time `t0`, advancing by `step` each call.
    pub const fn new(signal: F, t0: Tm, step: Tm) -> Self {
        Self {
            signal,
            t: t0,
            step,
        }
    }

    /// Wraps this sampler so each emission carries its own `dt` (equal to `step`).
    pub fn timed(self) -> TimedSampler<F, Tm> {
        TimedSampler(self)
    }
}

impl<F, Tm> Source for Sampler<F, Tm>
where
    F: Filter<Tm>,
    Tm: Copy + Add<Output = Tm>,
{
    type Output = F::Output;

    fn source(&mut self) -> Option<Self::Output> {
        let v = self.signal.filter(self.t);
        self.t = self.t + self.step;

        Some(v)
    }
}

/// A [`Sampler`] whose emissions carry [`Timed`] values, tagging each amplitude
/// with `dt` equal to the sampling `step`.
#[derive(Clone, Debug)]
pub struct TimedSampler<F, Tm = f32>(Sampler<F, Tm>);

impl<F, Tm> Source for TimedSampler<F, Tm>
where
    F: Filter<Tm>,
    Tm: Copy + Add<Output = Tm>,
{
    type Output = Timed<F::Output, Tm>;

    fn source(&mut self) -> Option<Self::Output> {
        let dt = self.0.step;

        self.0.source().map(|value| Timed { value, dt })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ramp;

    impl Filter<f32> for Ramp {
        type Output = f32;

        fn filter(&mut self, t: f32) -> f32 {
            t * 2.0
        }
    }

    #[test]
    fn sampler_advances_by_step() {
        let mut s = Sampler::new(Ramp, 0.0_f32, 0.5_f32);
        assert_eq!(s.source(), Some(0.0));
        assert_eq!(s.source(), Some(1.0)); // filter(0.5) = 1.0
        assert_eq!(s.source(), Some(2.0)); // filter(1.0) = 2.0
    }

    #[test]
    fn timed_sampler_carries_step_as_dt() {
        let mut s = Sampler::new(Ramp, 0.0_f32, 0.5_f32).timed();
        assert_eq!(
            s.source(),
            Some(Timed {
                value: 0.0,
                dt: 0.5
            })
        );
        assert_eq!(
            s.source(),
            Some(Timed {
                value: 1.0,
                dt: 0.5
            })
        );
    }
}
