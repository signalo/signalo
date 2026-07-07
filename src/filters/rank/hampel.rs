// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use num_traits::{Num, Signed};

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::median::{ListNode, Median, MedianArray};

/// The hampel filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The filter's outlier threshold.
    pub threshold: T,
}

/// The hampel filter's state.
#[derive(Clone)]
pub struct State<T, B> {
    /// Median filter.
    pub median: Median<T, B>,
}

impl<T, B> core::fmt::Debug for State<T, B>
where
    T: core::fmt::Debug,
    B: AsSlice<ListNode<T>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("State")
            .field("median", &self.median)
            .finish()
    }
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
/// - **Space:** O(N); stores the internal `Median<T, B>` window plus a scratch deviation array.
#[derive(Clone)]
pub struct Hampel<T: Clone, B, S> {
    config: Config<T>,
    state: State<T, B>,
    scratch: S,
}

impl<T, B, S> core::fmt::Debug for Hampel<T, B, S>
where
    T: Clone + core::fmt::Debug,
    B: AsSlice<ListNode<T>>,
    S: AsSlice<T>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Hampel")
            .field("config", &self.config)
            .field("state", &self.state)
            .field("scratch", &self.scratch.as_slice())
            .finish()
    }
}

/// A [`Hampel`] filter backed by fixed-size arrays.
pub type HampelArray<T, const N: usize> = Hampel<T, [ListNode<T>; N], [T; N]>;

/// A [`Hampel`] filter backed by heap-allocated `Vec`s.
#[cfg(feature = "alloc")]
pub type HampelVec<T> = Hampel<T, alloc::vec::Vec<ListNode<T>>, alloc::vec::Vec<T>>;

/// A [`Hampel`] filter that borrows caller-owned slices.
pub type HampelRefMut<'a, T> = Hampel<T, &'a mut [ListNode<T>], &'a mut [T]>;

impl<T: Clone, B, S> Hampel<T, B, S>
where
    B: AsSlice<ListNode<T>>,
    S: AsSlice<T>,
{
    /// Creates a [`Hampel`] from pre-initialised parts.
    ///
    /// # Panics
    ///
    /// Panics if the median window length is 0 or if the scratch length does not
    /// equal the median window length.
    pub fn from_parts(config: Config<T>, median: Median<T, B>, scratch: S) -> Self {
        let window_len = median.len();
        assert!(window_len > 0, "Hampel: window size must be > 0");
        assert_eq!(
            scratch.as_slice().len(),
            window_len,
            "Hampel: scratch length must equal median window length"
        );
        Self {
            config,
            state: State { median },
            scratch,
        }
    }
}

impl<T: Clone, B, S> Hampel<T, B, S>
where
    T: PartialOrd + Num + Signed,
    B: AsSlice<ListNode<T>>,
    S: AsSlice<T>,
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
        let median = &self.state.median;
        let devs = self.scratch.as_mut_slice();

        for val in median.window_iter() {
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

impl<T: Clone, B, S> ConfigTrait for Hampel<T, B, S> {
    type Config = Config<T>;
}

impl<T: Clone, B, S> StateTrait for Hampel<T, B, S> {
    type State = State<T, B>;
}

impl<T: Clone, const N: usize> WithConfig for HampelArray<T, N>
where
    T: Num + Signed,
    MedianArray<T, N>: Default,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let median = MedianArray::default();
        let scratch: [T; N] = core::array::from_fn(|_| T::zero());
        Self::from_parts(config, median, scratch)
    }
}

impl<T: Clone, B, S> ConfigRef for Hampel<T, B, S> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Clone, B, S> ConfigClone for Hampel<T, B, S>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T: Clone, B, S> StateMut for Hampel<T, B, S> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T: Clone, B, S> HasGuts for Hampel<T, B, S> {
    type Guts = (Config<T>, State<T, B>, S);
}

impl<T: Clone, B, S> FromGuts for Hampel<T, B, S> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state, scratch) = guts;
        Self {
            config,
            state,
            scratch,
        }
    }
}

impl<T: Clone, B, S> IntoGuts for Hampel<T, B, S> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state, self.scratch)
    }
}

impl<T: Clone, const N: usize> Reset for HampelArray<T, N>
where
    T: Num + Signed,
    MedianArray<T, N>: Default,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T: Clone, const N: usize> ResetMut for HampelArray<T, N>
where
    T: Num + Signed,
    MedianArray<T, N>: Default,
{
}

macro_rules! impl_hampel_filter {
    ($t:ty => $f:expr) => {
        impl<B, S> Filter<$t> for Hampel<$t, B, S>
        where
            B: AsSlice<ListNode<$t>>,
            S: AsSlice<$t>,
        {
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
    use crate::filters::rank::median::MedianRefMut;

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
        let filter: HampelArray<f32, 7> = HampelArray::with_config(Config { threshold: 2.0 });
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
        let filter: HampelArray<f64, 7> = HampelArray::with_config(Config { threshold: 2.0 });
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

    #[test]
    #[should_panic(expected = "Hampel: scratch length must equal median window length")]
    fn from_parts_scratch_length_mismatch_panics() {
        let mut buffer: [ListNode<f32>; 3] = core::array::from_fn(|index| ListNode {
            value: None,
            previous: (index + 2) % 3,
            next: (index + 1) % 3,
        });
        let median = MedianRefMut::from_parts(&mut buffer);
        let mut scratch: [f32; 5] = [0.0; 5];
        let config = Config { threshold: 3.0 };
        let _ = HampelRefMut::from_parts(config, median, &mut scratch);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn hampel_vec_filters_without_panic() {
        let buffer: Vec<ListNode<f32>> = (0..3)
            .map(|index| ListNode {
                value: None,
                previous: (index + 2) % 3,
                next: (index + 1) % 3,
            })
            .collect();
        let median = Median::from_parts(buffer);
        let scratch = vec![0.0_f32; 3];
        let config = Config { threshold: 3.0 };
        let mut filter: HampelVec<f32> = Hampel::from_parts(config, median, scratch);
        let out = filter.filter(1.0);
        assert_eq!(out, 1.0);
    }

    #[test]
    fn hampel_ref_mut_filters_without_panic() {
        let mut buffer: [ListNode<f32>; 3] = core::array::from_fn(|index| ListNode {
            value: None,
            previous: (index + 2) % 3,
            next: (index + 1) % 3,
        });
        let median = MedianRefMut::from_parts(&mut buffer);
        let mut scratch: [f32; 3] = [0.0; 3];
        let config = Config { threshold: 3.0 };
        let mut filter = HampelRefMut::from_parts(config, median, &mut scratch);
        let out = filter.filter(1.0);
        assert_eq!(out, 1.0);
    }
}
