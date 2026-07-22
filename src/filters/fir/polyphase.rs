// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR filter banks, executors, and multirate filters.
//!
//! [`filter_bank`] contains the coefficient storage and selected-phase execution
//! primitive. [`fir`] adds sample history. The [`interpolator`], [`decimator`], and
//! [`rational_resampler`] modules build streaming
//! [`MultirateFilter`](crate::traits::MultirateFilter) adapters on top of those
//! primitives.
//!
//! # Prototype design rates
//!
//! A dense prototype is the ordinary FIR tap sequence before it is split into
//! polyphase branches. Its design rate is the sample rate where that FIR would
//! run in the equivalent non-polyphase resampler. Frequency parameters such as
//! cutoff frequencies and transition widths should be normalized to that same
//! rate.
//!
//! | Case | Prototype design rate | Notes |
//! | --- | --- | --- |
//! | [Interpolator `L`](interpolator::PolyphaseInterpolator) | `input_rate * L` | The prototype runs at the output rate, where it suppresses interpolation images. |
//! | [Decimator `M`](decimator::PolyphaseDecimator) | `input_rate` | The anti-aliasing filter sees the input-rate spectrum before downsampling. |
//! | [Rational `L/M`](rational_resampler::RationalResampler) | `input_rate * L` | The filter runs at the intermediate rate after interpolation and before decimation. |
//!
//! For rational resamplers, `max(L, M)` is useful when choosing a low-pass
//! anti-image and anti-aliasing prototype. At `input_rate * L`, the input
//! Nyquist maps to `1 / (2L)` and the output Nyquist maps to `1 / (2M)`, so the
//! lower Nyquist limit is `0.5 / max(L, M)`. Use that as the normalized
//! boundary when choosing the prototype's passband edge, cutoff, and transition
//! band for the desired image and alias rejection.
//!
//! # Prototype gain
//!
//! [`interpolator`] and [`rational_resampler`] do not apply interpolation gain
//! scaling. If a prototype designer normalizes taps to unity passband gain,
//! multiply the prototype coefficients by `L` before polyphase construction
//! when unity amplitude should be preserved. This compensates for the
//! interpolation step that inserts `L - 1` zero-valued samples between input
//! samples. A unity-gain prototype preserves that upsampled sequence's
//! `1 / L` baseband/DC gain, while scaling the prototype by `L` restores unity
//! passband amplitude.
//!
//! Decimators are usually constructed with a unity-passband-gain prototype. A
//! decimator is equivalent to filtering at the input rate and then keeping every
//! `M`th output sample, so downsampling changes sample spacing but does not
//! require any amplitude correction.

pub mod decimator;
pub mod filter_bank;
pub mod fir;
pub mod interpolator;
pub mod rational_resampler;

#[cfg(test)]
pub(crate) mod test_support {
    use core::ops::{Add, Mul};

    use num_traits::Zero;

    /// Test-only bundle proving one PFB pass can run two parallel real filters.
    ///
    /// With `T = i32`, this stores two coefficient weights. As the dot-product output, it stores the
    /// two corresponding output accumulators.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct Pair<T = i32> {
        /// First parallel filter value.
        pub(crate) first: T,
        /// Second parallel filter value.
        pub(crate) second: T,
    }

    impl<T> Add for Pair<T>
    where
        T: Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self {
                first: self.first + rhs.first,
                second: self.second + rhs.second,
            }
        }
    }

    impl<T> Zero for Pair<T>
    where
        T: Zero + Add<Output = T>,
    {
        fn zero() -> Self {
            Self {
                first: T::zero(),
                second: T::zero(),
            }
        }

        fn is_zero(&self) -> bool {
            self.first.is_zero() && self.second.is_zero()
        }
    }

    impl Mul<Pair> for i32 {
        type Output = Pair;

        fn mul(self, rhs: Pair) -> Pair {
            Self::Output {
                first: self * rhs.first,
                second: self * rhs.second,
            }
        }
    }
}
