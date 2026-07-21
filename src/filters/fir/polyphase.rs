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
