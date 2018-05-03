// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kalman filters.

// use std::ops::{Sub, Add, Mul, Div};

use num_traits::{Zero, One, Num};

use signalo_traits::filter::Filter;
use traits::Stateful;

/// A 1-dimensional Kalman filter.
#[derive(Clone, Debug)]
pub struct Kalman<T> {
    r: T, /// Process noise
    q: T, /// Measurement noise
    a: T, /// State
    b: T, /// Control
    c: T, /// Measurement
    cov: T, /// Uncertainty
    x: Option<T>,
}

impl<T> Kalman<T>
where
    T: Zero + One
{
    /// Creates a new `Kalman` filter with given `r`, `q`, `a`, `b`, and `c` coefficients.
    ///
    /// Coefficients:
    /// - `r`: process noise covariance
    /// - `q`: measurement noise covariance
    /// - `a`: state transition
    /// - `b`: control transition
    /// - `c`: measurement
    #[inline]
    pub fn new(r: T, q: T, a: T, b: T, c: T) -> Self {
        let cov = T::zero();
        let x = None;
        Kalman { r, q, a, b, c, cov, x }
    }

    /// Process noise covariance
    #[inline]
    pub fn r(&self) -> &T {
        &self.r
    }

    /// Measurement noise covariance
    #[inline]
    pub fn q(&self) -> &T {
        &self.q
    }

    /// State transition
    #[inline]
    pub fn a(&self) -> &T {
        &self.a
    }

    /// Control transition
    #[inline]
    pub fn b(&self) -> &T {
        &self.b
    }

    /// Measurement
    #[inline]
    pub fn c(&self) -> &T {
        &self.c
    }
}

impl<T> Kalman<T>
where
    T: Copy + Num
{
    fn process(&mut self, (input, control): (T, T)) -> T {
        let c_squared = self.c * self.c;
        let (x, cov) = match self.x {
            None => {
                let x = input / self.c;
                let cov = self.q / c_squared;
                (x, cov)
            },
            Some(mut x) => {
                // Compute prediction
                let pred_state = (self.a * x) + (self.b * control);
                let pred_cov = (self.a * self.cov * self.a) + self.r;

                // Kalman gain
                let k = pred_cov * self.c / ((pred_cov * c_squared) + self.q);

                // Correction
                let x = pred_state + k * (input - (self.c * pred_state));
                let cov = pred_cov - (k * self.c * pred_cov);
                (x, cov)
            },
        };
        self.x = Some(x);
        self.cov = cov;
        x
    }
}

impl<T> Default for Kalman<T>
where
    T: Zero + One
{
    #[inline]
    fn default() -> Self {
        let r = T::one();
        let q = T::one();
        let a = T::one();
        let b = T::zero();
        let c = T::one();
        Kalman::new(r, q, a, b, c)
    }
}

impl<T> Filter<T> for Kalman<T>
where
    T: Copy + Num
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.process((input, T::zero()))
    }
}

impl<T> Filter<(T, T)> for Kalman<T>
where
    T: Copy + Num
{
    type Output = T;

    fn filter(&mut self, (input, control): (T, T)) -> Self::Output {
        self.process((input, control))
    }
}

impl<T> Stateful for Kalman<T> {
    #[inline]
    fn reset(&mut self) {
        self.x = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.502, 2.704, 2.522, 3.047, 3.946, 5.883, 5.463, 7.288, 7.125, 7.951, 8.071,
            8.175, 9.129, 9.960, 9.342, 9.614, 10.660, 11.591, 11.138, 10.731, 11.148, 11.522,
            11.375, 12.497, 12.256, 21.739, 21.381, 21.057, 20.765, 28.908, 26.625, 26.566,
            25.272, 24.103, 23.807, 23.540, 23.298, 24.317, 22.763, 30.971, 28.785, 28.806,
            27.587, 26.485, 25.487, 32.957, 30.868, 30.215, 29.623
        ]
    }

    #[test]
    fn floating_point() {
        let r = 0.01; // Process noise
        let q = 1.0; // Measurement noise
        let a = 1.0; // State
        let b = 0.0; // Control
        let c = 1.0; // Measurement
        let filter = Kalman::new(r, q, a, b, c);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();

        assert_nearly_eq!(output, get_output(), 0.001);
    }

    #[cfg(feature = "fpa")]
    #[test]
    fn fixed_point() {
        use std::convert::TryFrom;
        use fpa::I16F16;
        type Fixed = I16F16;

        let r = Fixed::try_from(0.01).unwrap(); // Process noise
        let q = Fixed::try_from(1.00).unwrap(); // Measurement noise
        let a = Fixed::try_from(1.00).unwrap(); // State
        let b = Fixed::try_from(0.00).unwrap(); // Control
        let c = Fixed::try_from(1.00).unwrap(); // Measurement
        let filter = Kalman::new(r, q, a, b, c);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let float_input = get_input();

        let input: Vec<_> = float_input.into_iter().map(|float| {
            Fixed::try_from(float).unwrap()
        }).collect();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();

        let float_output: Vec<_> = output.into_iter().map(|fixed| {
            f32::try_from(fixed).unwrap()
        }).collect();

        assert_nearly_eq!(float_output, get_output(), 0.01);
    }
}
