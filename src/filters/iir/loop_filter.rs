// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Proportional-integral (PI) filter for second-order synchronization loops.
//!
//! This module provides a PI controller for feedback loops such as carrier
//! and symbol-timing synchronizers. Considered by itself, the digital PI
//! controller contains one integrator and can be represented as a first-order
//! IIR filter.
//!
//! In PLL and timing-recovery terminology, the complete loop is commonly called
//! second-order, or type II, because the loop-filter integrator and the NCO or
//! timing-accumulator integrator provide two loop state variables.
//!
//! When stable and operating in lock, such a loop can track a constant
//! frequency offset or symbol-rate offset with zero steady-state phase or
//! timing-phase error. The gains are commonly derived from a normalized loop
//! bandwidth (`Bn * T`) and damping factor, together with the phase-detector and
//! controlled-oscillator gains.

use crate::filters::iir::integrate::Integrate;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Proportional-integral loop filter gains.
///
/// When constructed via [`Config::new`] or [`Config::new_with_loop_gain`], the
/// two gains are a mathematically coupled pair derived from the same loop
/// bandwidth, damping factor, and loop-gain convention. Modifying one field
/// independently changes the loop's natural frequency, damping, or loop-gain
/// compensation.
#[derive(Clone, Copy, Debug)]
pub struct Config<T = f32> {
    /// Proportional path gain.
    pub proportional_gain: T,
    /// Integral path gain.
    pub integral_gain: T,
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T> Config<T>
where
    T: Float + core::fmt::Debug,
{
    /// Creates loop-filter gains from normalized loop bandwidth (`Bn * T`) and damping factor.
    ///
    /// The coefficients are computed as:
    ///
    /// ```text
    /// theta = loop_bandwidth / (damping + 1 / (4 * damping))
    /// K_p   = 4 * damping * theta / (1 + 2 * damping * theta + theta^2)
    /// K_i   = 4 * theta^2         / (1 + 2 * damping * theta + theta^2)
    /// ```
    ///
    /// This constructor assumes a combined phase-detector and
    /// controlled-oscillator gain of `K_d * K_0 = 1`. Use
    /// [`Config::new_with_loop_gain`] when the combined loop gain differs from
    /// unity.
    ///
    /// # Panics
    ///
    /// Panics if `loop_bandwidth` or `damping` is not finite or is not positive,
    /// or if the derived gains are not finite.
    #[must_use]
    pub fn new(loop_bandwidth: T, damping: T) -> Self {
        assert!(
            loop_bandwidth.is_finite() && loop_bandwidth > T::zero(),
            "loop bandwidth must be finite and > 0 (got {loop_bandwidth:?})"
        );
        assert!(
            damping.is_finite() && damping > T::zero(),
            "damping factor must be finite and > 0 (got {damping:?})"
        );

        let quarter = T::from(0.25).expect("0.25 is representable");
        let two = T::from(2.0).expect("2.0 is representable");
        let four = T::from(4.0).expect("4.0 is representable");

        let theta = loop_bandwidth / (damping + quarter / damping);
        let denominator = T::one() + two * damping * theta + theta * theta;
        let config = Self {
            proportional_gain: four * damping * theta / denominator,
            integral_gain: four * theta * theta / denominator,
        };
        assert!(
            config.proportional_gain.is_finite() && config.integral_gain.is_finite(),
            "loop-filter gains must be finite: {config:?}"
        );
        config
    }

    /// Creates loop-filter gains for a specified combined loop gain.
    ///
    /// `loop_gain` is the product `K_d * K_0` of the phase-detector and
    /// controlled-oscillator gains. The unity-gain coefficients are scaled as:
    ///
    /// ```text
    /// K_p' = K_p / loop_gain
    /// K_i' = K_i / loop_gain
    /// ```
    ///
    /// This preserves the requested bandwidth and damping.
    ///
    /// # Panics
    ///
    /// Panics if `loop_bandwidth`, `damping`, or `loop_gain` is not finite or is
    /// not positive, or if the derived gains are not finite.
    #[must_use]
    pub fn new_with_loop_gain(loop_bandwidth: T, damping: T, loop_gain: T) -> Self {
        assert!(
            loop_gain.is_finite() && loop_gain > T::zero(),
            "combined loop gain must be finite and > 0 (got {loop_gain:?})"
        );

        let config = Self::new(loop_bandwidth, damping);

        Self {
            proportional_gain: config.proportional_gain / loop_gain,
            integral_gain: config.integral_gain / loop_gain,
        }
    }
}

/// A proportional-integral filter for second-order synchronization loops.
///
/// The proportional and integral gains are typically derived from a normalized
/// loop bandwidth (`Bn * T`, cycles per symbol) and a damping factor using
/// [`Config::new`]. The integral path uses [`Integrate<T>`].
#[derive(Clone, Debug)]
pub struct LoopFilter<T = f32> {
    config: Config<T>,
    integrator: Integrate<T>,
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T> LoopFilter<T>
where
    T: Float + core::fmt::Debug,
{
    /// Creates a loop filter.
    ///
    /// # Panics
    ///
    /// Panics if [`Config::new`] rejects the parameters.
    #[must_use]
    pub fn new(loop_bandwidth: T, damping: T) -> Self {
        Self::with_config(Config::new(loop_bandwidth, damping))
    }

    /// Creates a loop filter for a specified combined loop gain.
    ///
    /// # Panics
    ///
    /// Panics if [`Config::new_with_loop_gain`] rejects the parameters.
    #[must_use]
    pub fn new_with_loop_gain(loop_bandwidth: T, damping: T, loop_gain: T) -> Self {
        Self::with_config(Config::new_with_loop_gain(
            loop_bandwidth,
            damping,
            loop_gain,
        ))
    }
}

impl<T> LoopFilter<T> {
    /// Creates a loop filter from already-derived gains and integrator state.
    #[must_use]
    pub fn from_parts(config: Config<T>, integrator: Integrate<T>) -> Self {
        Self { config, integrator }
    }
}

impl<T> LoopFilter<T>
where
    T: Clone + core::ops::Add<T, Output = T> + core::ops::Mul<T, Output = T>,
    Integrate<T>: Filter<T, Output = T>,
{
    /// Advances the filter with a detector `error` and returns the control output.
    #[must_use]
    pub fn update(&mut self, error: T) -> T {
        self.filter(error)
    }
}

impl<T> ConfigTrait for LoopFilter<T> {
    type Config = Config<T>;
}

impl<T> ConfigRef for LoopFilter<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for LoopFilter<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateTrait for LoopFilter<T> {
    type State = Integrate<T>;
}

impl<T> StateMut for LoopFilter<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.integrator
    }
}

impl<T> WithConfig for LoopFilter<T>
where
    Integrate<T>: Default,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self::from_parts(config, Integrate::default())
    }
}

impl<T> HasGuts for LoopFilter<T> {
    type Guts = (Config<T>, Integrate<T>);
}

impl<T> FromGuts for LoopFilter<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, integrator) = guts;
        Self { config, integrator }
    }
}

impl<T> IntoGuts for LoopFilter<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.integrator)
    }
}

impl<T> Reset for LoopFilter<T>
where
    Integrate<T>: Reset,
{
    fn reset(self) -> Self {
        Self {
            config: self.config,
            integrator: self.integrator.reset(),
        }
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for LoopFilter<T> where Self: Reset {}

impl<T> Filter<T> for LoopFilter<T>
where
    T: Clone + core::ops::Add<T, Output = T> + core::ops::Mul<T, Output = T>,
    Integrate<T>: Filter<T, Output = T>,
{
    type Output = T;

    fn filter(&mut self, error: T) -> Self::Output {
        let proportional = self.config.proportional_gain.clone() * error.clone();
        let integral = self
            .integrator
            .filter(self.config.integral_gain.clone() * error);
        proportional + integral
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::traits::{ConfigRef, Filter, Reset};

    use super::*;

    #[test]
    fn config_computes_standard_active_pi_gains() {
        let config = Config::new(0.01, core::f32::consts::FRAC_1_SQRT_2);

        assert_abs_diff_eq!(config.proportional_gain, 0.026_313_48, epsilon = 1e-8);
        assert_abs_diff_eq!(config.integral_gain, 0.000_350_846_4, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "loop bandwidth")]
    fn config_rejects_zero_loop_bandwidth() {
        let _ = Config::new(0.0, 0.707);
    }

    #[test]
    #[should_panic(expected = "loop bandwidth")]
    fn config_rejects_negative_loop_bandwidth() {
        let _ = Config::new(-0.01_f32, 0.707);
    }

    #[test]
    #[should_panic(expected = "loop bandwidth")]
    fn config_rejects_nan_loop_bandwidth() {
        let _ = Config::new(f32::NAN, 0.707);
    }

    #[test]
    #[should_panic(expected = "loop bandwidth")]
    fn config_rejects_infinite_loop_bandwidth() {
        let _ = Config::new(f32::INFINITY, 0.707);
    }

    #[test]
    #[should_panic(expected = "damping")]
    fn config_rejects_zero_damping() {
        let _ = Config::new(0.01, 0.0);
    }

    #[test]
    #[should_panic(expected = "damping")]
    fn config_rejects_negative_damping() {
        let _ = Config::new(0.01_f32, -0.707);
    }

    #[test]
    #[should_panic(expected = "damping")]
    fn config_rejects_nan_damping() {
        let _ = Config::new(0.01_f32, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "damping")]
    fn config_rejects_infinite_damping() {
        let _ = Config::new(0.01_f32, f32::INFINITY);
    }

    #[test]
    fn config_compensates_for_combined_loop_gain() {
        let unity = Config::new(0.01, core::f32::consts::FRAC_1_SQRT_2);
        let compensated = Config::new_with_loop_gain(0.01, core::f32::consts::FRAC_1_SQRT_2, 2.0);

        assert_abs_diff_eq!(
            compensated.proportional_gain,
            unity.proportional_gain / 2.0,
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(
            compensated.integral_gain,
            unity.integral_gain / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    #[should_panic(expected = "combined loop gain")]
    fn config_rejects_zero_loop_gain() {
        let _ = Config::new_with_loop_gain(0.01, 0.707, 0.0);
    }

    #[test]
    #[should_panic(expected = "combined loop gain")]
    fn config_rejects_negative_loop_gain() {
        let _ = Config::new_with_loop_gain(0.01, 0.707, -2.0);
    }

    #[test]
    #[should_panic(expected = "combined loop gain")]
    fn config_rejects_nan_loop_gain() {
        let _ = Config::new_with_loop_gain(0.01, 0.707, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "combined loop gain")]
    fn config_rejects_infinite_loop_gain() {
        let _ = Config::new_with_loop_gain(0.01, 0.707, f32::INFINITY);
    }

    #[test]
    fn filter_uses_proportional_plus_integral_paths() {
        let config = Config {
            proportional_gain: 2.0,
            integral_gain: 0.5,
        };
        let mut filter = LoopFilter::<f32>::with_config(config);

        assert_abs_diff_eq!(filter.filter(1.0), 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(filter.filter(1.0), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn update_matches_filter() {
        let mut a = LoopFilter::new(0.01, 0.707);
        let mut b = LoopFilter::new(0.01, 0.707);

        assert_eq!(a.update(0.25), b.filter(0.25));
    }

    #[test]
    fn reset_clears_integrator_but_keeps_config() {
        let mut filter = LoopFilter::<f32>::with_config(Config {
            proportional_gain: 2.0,
            integral_gain: 0.5,
        });
        let _ = filter.filter(1.0);

        let mut reset = filter.reset();

        assert_abs_diff_eq!(reset.filter(1.0), 2.5, epsilon = 1e-6);
        assert_eq!(reset.config_ref().proportional_gain, 2.0);
    }

    #[test]
    fn supports_f64_gains() {
        let mut filter = LoopFilter::<f64>::new(0.01, core::f64::consts::FRAC_1_SQRT_2);

        let output = filter.filter(0.25);

        assert!(output > 0.0);
    }
}
