// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use generic_array::ArrayLength;
use num_traits::{Num, Signed};

use signalo_traits::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, FromGuts, Guts, IntoGuts, Reset,
    State as StateTrait, StateMut, WithConfig,
};

use median::{ListNode, Median};

/// The hampel filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The filter's outlier threshold.
    pub threshold: T,
}

/// The hampel filter's state.
#[derive(Clone, Debug)]
pub struct State<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    /// Median filter.
    pub median: Median<T, N>,
}

/// A hampel filter of fixed width.
///
/// J. Astola, P. Kuosmanen, "Fundamentals of Nonlinear Digital Filtering", CRC Press, 1997.
#[derive(Clone, Debug)]
pub struct Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    config: Config<T>,
    state: State<T, N>,
}

impl<T, N> Hampel<T, N>
where
    T: Clone + PartialOrd + Num + Signed,
    N: ArrayLength<ListNode<T>>,
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
        // Read window's current median and min/max boundaries:
        let min = self.state.median.min().unwrap_or_else(|| input.clone());
        let median = self.state.median.median().unwrap_or_else(|| input.clone());
        let max = self.state.median.max().unwrap_or_else(|| input.clone());

        // Feed the input to the internal median filter:
        self.state.median.filter(input.clone());

        // Calculate the boundary's absolute deviations from the median:
        let min_dev = (median.clone() - min).abs();
        let max_dev = (max - median.clone()).abs();

        // Calculate the overall median absolute deviation:
        let med_abs_dev = if min_dev < max_dev { max_dev } else { min_dev };

        // Estimate the standard deviation:
        let std_dev = med_abs_dev.clone() * factor;

        // Calculate the input's deviation from the median:
        let dev = (input.clone() - median.clone()).abs();

        // Calculate window's threshold:
        let threshold = std_dev.clone() * self.config.threshold.clone();

        // If input falls outside the threshold we return the median instead:
        if dev > threshold {
            median
        } else {
            input
        }
    }
}

impl<T, N> ConfigTrait for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    type Config = Config<T>;
}

impl<T, N> StateTrait for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    type State = State<T, N>;
}

impl<T, N> WithConfig for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
    Median<T, N>: Default,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let median = Median::default();
            State { median }
        };
        Self { config, state }
    }
}

impl<T, N> ConfigRef for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, N> ConfigClone for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, N> StateMut for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, N> Guts for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    type Guts = (Config<T>, State<T, N>);
}

impl<T, N> FromGuts for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, N> IntoGuts for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, N> Reset for Hampel<T, N>
where
    N: ArrayLength<ListNode<T>>,
    Median<T, N>: Default,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive_reset_mut")]
impl<T, N> ResetMut for Hampel<T, N>
where
    Self: Reset,
{}

macro_rules! impl_hampel_filter {
    ($t:ty => $f:expr) => {
        impl<N> Filter<$t> for Hampel<$t, N>
        where
            // T: Clone + PartialOrd + Num + Signed,
            N: ArrayLength<ListNode<$t>>,
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
    use super::*;

    use generic_array::typenum::*;

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
            0.0, 0.0, 0.0, 2.0, 1.0, 8.0, 16.0, 3.0, 5.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 10.0, 18.0, 18.0, 18.0, 18.0,
            5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 21.0, 8.0, 29.0, 16.0, 16.0, 16.0,
            16.0, 11.0, 24.0, 24.0,
        ]
    }

    #[test]
    fn test() {
        let filter: Hampel<_, U7> = Hampel::with_config(Config { threshold: 2.0 });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
