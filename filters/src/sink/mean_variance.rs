// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use num_traits::{Zero, One, Num};

use signalo_traits::sink::Sink;

/// A sink that computes the mean and variance of all received values of a signal.
#[derive(Clone, Debug)]
pub struct MeanVariance<T> {
    count: T, // number of values seen so far
    mean: T, // mean of values seen so far
    variance: T, // squared distance from the mean
}

impl<T> MeanVariance<T>
where
    T: Zero,
{
    /// Creates a new `MeanVariance` sink.
    #[inline]
    pub fn new() -> Self {
        let count = T::zero();
        let mean = T::zero();
        let variance = T::zero();
        MeanVariance { count, mean, variance }
    }
}

impl<T> Sink<T> for MeanVariance<T>
where
    T: Copy + PartialOrd + One + Num,
{
    type Output = (T, T);

    #[inline]
    fn sink(&mut self, input: T) {
        let count = self.count + T::one();
        let delta = input - self.mean;
        let mean = self.mean + (delta / count);
        let delta2 = input - mean;
        let variance = self.variance + (delta * delta2);

        self.count = count;
        self.mean = mean;
        self.variance = variance;
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        let variance = if self.count > T::one() {
            self.variance / (self.count - T::one())
        } else {
            self.variance
        };
        (self.mean, variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sink() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ];
        let mut sink = MeanVariance::new();
        for input in input {
            sink.sink(input);
        }
        println!("\n{:#?}\n", sink);
        let (mean, variance) = sink.finalize();
        assert_nearly_eq!(mean, 26.56);
        assert_nearly_eq!(variance, 1347.68);
    }
}
