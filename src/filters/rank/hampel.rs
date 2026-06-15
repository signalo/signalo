// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use num_traits::{Num, Signed};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::median::Median;

/// The hampel filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The filter's outlier threshold.
    pub threshold: T,
}

/// The hampel filter's state.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    /// Median filter.
    pub median: Median<T, N>,
}

/// A hampel filter of fixed width.
///
/// J. Astola, P. Kuosmanen, "Fundamentals of Nonlinear Digital Filtering", CRC Press, 1997.
///
/// # Complexity
///
/// - **Time per sample:** O(N); delegates to the internal `Median` filter (O(N)), then
///   collects N absolute deviations and sorts them with an insertion sort (O(N²) worst-case,
///   but N is a small compile-time constant in practice).
/// - **Space:** O(N); stores the internal `Median<T, N>` window plus a stack-allocated
///   deviation array of size N.
#[derive(Clone, Debug)]
pub struct Hampel<T, const N: usize> {
    config: Config<T>,
    state: State<T, N>,
}

impl<T, const N: usize> Hampel<T, N>
where
    T: Clone + PartialOrd + Num + Signed,
{
    /// The Hampel Filter
    ///
    /// For each input sample the function computes the median of a window
    /// composed of the sample and its `N`-1 surrounding samples (assuming an odd window size).
    /// It also estimates the standard deviation of each sample around its
    /// window median using the median absolute deviation.
    /// If a sample differs from the median by more than `self.threshold` standard deviations,
    /// it is replaced with the median:
    fn filter_internal(&mut self, input: T, factor: T) -> T {
        // Read window's current median to use as outlier fallback:
        let pre_median = self.state.median.median().unwrap_or_else(|| input.clone());

        // Feed the input to the internal median filter:
        self.state.median.filter(input.clone());

        // We use the updated median to calculate deviations:
        let post_median = self.state.median.median().unwrap_or_else(|| input.clone());

        let mut count = 0usize;
        let mut devs: [T; N] = core::array::from_fn(|_| T::zero());

        for val in self.state.median.window_iter() {
            let dev = (val.clone() - post_median.clone()).abs();
            devs[count] = dev;
            count += 1;
        }

        let devs_init = &mut devs[..count];

        // In-place insertion sort to avoid `alloc` requirement in `no_std`
        for i in 1..count {
            let mut j = i;
            while j > 0 {
                let a = &devs_init[j - 1];
                let b = &devs_init[j];
                if a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                    == core::cmp::Ordering::Greater
                {
                    devs_init.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }

        let mad = if count == 0 {
            T::zero()
        } else {
            devs_init[count / 2].clone()
        };

        let std_dev = mad * factor;
        let dev = (input.clone() - pre_median.clone()).abs();
        let threshold = std_dev * self.config.threshold.clone();

        if dev > threshold {
            pre_median
        } else {
            input
        }
    }
}

impl<T, const N: usize> ConfigTrait for Hampel<T, N> {
    type Config = Config<T>;
}

impl<T, const N: usize> StateTrait for Hampel<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for Hampel<T, N>
where
    Median<T, N>: Default,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Hampel: window size N must be > 0");
        let state = {
            let median = Median::default();
            State { median }
        };
        Self { config, state }
    }
}

impl<T, const N: usize> ConfigRef for Hampel<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Hampel<T, N>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Hampel<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Hampel<T, N> {
    type Guts = (Config<T>, State<T, N>);
}

impl<T, const N: usize> FromGuts for Hampel<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Hampel<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Hampel<T, N>
where
    Median<T, N>: Default,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Hampel<T, N> where Self: Reset {}

macro_rules! impl_hampel_filter {
    ($t:ty => $f:expr) => {
        impl<const N: usize> Filter<$t> for Hampel<$t, N> {
            type Output = $t;

            fn filter(&mut self, input: $t) -> Self::Output {
                self.filter_internal(input, $f)
            }
        }
    };
}

// `1.4826` is our standard deviation estimation factor:
// https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
impl_hampel_filter!(f32 => 1.4826);
impl_hampel_filter!(f64 => 1.4826);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 0.0, 2.0, 5.0, 8.0, 2.0, 3.0, 5.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 17.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 10.0, 18.0, 18.0, 18.0,
            18.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 21.0, 8.0, 29.0, 16.0, 16.0,
            16.0, 16.0, 11.0, 24.0, 24.0,
        ]
    }

    #[test]
    fn test() {
        let filter: Hampel<_, 7> = Hampel::with_config(Config { threshold: 2.0 });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn consecutive_outliers_are_both_rejected() {
        // A correct Hampel filter must reject the second outlier too.
        let filter: Hampel<f64, 7> = Hampel::with_config(Config { threshold: 2.0 });
        let signal = vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0];
        let output: std::vec::Vec<_> = signal
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        // Both 100.0 values should be suppressed (replaced by ~0.0).
        assert!(
            output[3] < 50.0,
            "first outlier not rejected: {}",
            output[3]
        );
        assert!(
            output[7] < 50.0,
            "second outlier not rejected: {}",
            output[7]
        );
    }
}
