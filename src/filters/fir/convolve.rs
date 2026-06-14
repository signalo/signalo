// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use circular_buffer::CircularBuffer;
use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

pub mod differentiator;
pub mod lagrange;
pub mod moving_sum;
pub mod windowed_sinc;

/// The convolution filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize> {
    /// The convolution coefficients.
    pub coefficients: [T; N],
}

/// The convolution filter's state.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    /// The filter's taps (i.e. buffered input).
    pub taps: CircularBuffer<N, T>,
}

/// A convolution filter.
///
/// # Coefficient ordering
///
/// Coefficients `h[k]` pair with taps `x[n−k]` so that `h[0]` multiplies the
/// newest sample and `h[N−1]` the oldest. The dot product computes
/// `y[n] = Σ_{k=0}^{N−1} h[k]·x[n−k]` using zero-padding for negative
/// time indices. This convention is verified by the `coefficient_ordering`
/// test.
///
/// # Complexity
///
/// - **Time per sample:** O(N); dot product of N taps with N coefficients.
/// - **Space:** O(N); circular tap buffer of N elements plus N coefficient array.
///
/// # Cold-start behaviour
///
/// On construction, the tap buffer is pre-filled with `N` zeros. The first
/// `N − 1` outputs therefore reflect implicit zero-padding `x[n] = 0` for
/// `n < 0`, as verified by `cold_start_is_zero_padded_partial_convolution`.
/// Discard the warm-up window if zero-padding bias is unacceptable
/// for your application.
#[derive(Clone, Debug)]
pub struct Convolve<T, const N: usize> {
    config: Config<T, N>,
    state: State<T, N>,
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T, const N: usize> Convolve<T, N>
where
    T: num_traits::Float,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing
    /// them to unity DC gain.
    ///
    /// This constructor is float-only. For integer types, use
    /// [`with_config`](Self::with_config) directly with manually pre-scaled
    /// coefficients.
    ///
    /// # Behaviour
    ///
    /// If `sum == 0` (exact), normalisation is skipped — this is the documented
    /// DC-blocker escape hatch. Otherwise the sum must be finite and its
    /// magnitude must be at or above `T::min_positive_value().sqrt()`; smaller
    /// denominators (near-zero) panic.
    pub fn normalized(mut config: Config<T, N>) -> Self
    where
        T: core::fmt::Debug,
    {
        let sum = config
            .coefficients
            .iter()
            .copied()
            .fold(T::zero(), |a, b| a + b);
        if !sum.is_zero() {
            // Exact zero is treated as an explicit DC-blocker request; near-zero is treated as numerical error and rejected by safe_normalise_divisor.
            let denom =
                crate::math::safe_normalise_divisor(sum, "Convolve::normalized: coefficient sum");
            for coeff in &mut config.coefficients {
                *coeff = *coeff / denom;
            }
        }
        Self::with_config(config)
    }
}

impl<T, const N: usize> ConfigTrait for Convolve<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Convolve<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for Convolve<T, N>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Convolve: window size N must be > 0");
        let state = {
            let mut taps = CircularBuffer::new();
            for _ in 0..N {
                let _ = taps.push_back(T::zero());
            }
            State { taps }
        };
        Self { config, state }
    }
}

impl<T, const N: usize> ConfigRef for Convolve<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Convolve<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Convolve<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Convolve<T, N> {
    type Guts = (Config<T, N>, State<T, N>);
}

impl<T, const N: usize> FromGuts for Convolve<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Convolve<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Convolve<T, N>
where
    T: Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Convolve<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Convolve<T, N>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.taps.push_back(input);

        let state_iter = self.state.taps.iter();
        // See "Coefficient ordering" in the struct-level documentation.
        // coeff_iter.rev(): state iterates oldest->newest; reversing pairs h[N-1] with oldest, h[0] with newest. See struct-level "Coefficient ordering".
        let coeff_iter = self.config.coefficients.iter().rev();

        state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + (state.clone() * coeff.clone())
            })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _ = Convolve::<f32, 0>::with_config(Config { coefficients: [] });
    }

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0,
            8.0, 8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0,
            88.0, -101.0, 21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0, -13.0,
            0.0, 0.0, 88.0, -93.0, 13.0, 0.0,
        ]
    }

    #[test]
    fn test() {
        // Effectively calculates the derivative:
        let filter = Convolve::with_config(Config {
            coefficients: [1.000, -1.000],
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn test_normalized() {
        // Test normalizing coefficients
        let filter = Convolve::<f32, 3>::normalized(Config {
            coefficients: [2.0, 4.0, 6.0],
        });
        let config = filter.config_ref();
        // Sum is 12.0, so normalized coefficients should be [2/12, 4/12, 6/12] = [1/6, 1/3, 1/2]
        assert_abs_diff_eq!(config.coefficients[0], 1.0 / 6.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[1], 1.0 / 3.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[2], 1.0 / 2.0, epsilon = 0.0001);
    }

    #[test]
    fn test_normalized_zero_sum() {
        // Test normalizing coefficients that sum to zero
        let filter = Convolve::<f32, 3>::normalized(Config {
            coefficients: [1.0, -1.0, 0.0],
        });
        let config = filter.config_ref();
        // Sum is 0.0, so coefficients should remain unchanged
        assert_abs_diff_eq!(config.coefficients[0], 1.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[1], -1.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[2], 0.0, epsilon = 0.0001);
    }

    #[test]
    fn test_normalized_zero_sum_exact() {
        // Coefficients [1, -1] sum to exactly zero; should round-trip unchanged.
        let filter = Convolve::<f32, 2>::normalized(Config {
            coefficients: [1.0, -1.0],
        });
        let config = filter.config_ref();
        assert_abs_diff_eq!(config.coefficients[0], 1.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[1], -1.0, epsilon = 0.0001);
    }

    #[test]
    fn test_normalized_negative_sum() {
        // Negative-sum coefficients are divided through correctly to unity gain.
        let filter = Convolve::<f32, 2>::normalized(Config {
            coefficients: [-1.0, 0.5],
        });
        let config = filter.config_ref();
        // Sum is -0.5, so normalized: [-1.0 / -0.5, 0.5 / -0.5] = [2.0, -1.0]
        assert_abs_diff_eq!(config.coefficients[0], 2.0, epsilon = 0.0001);
        assert_abs_diff_eq!(config.coefficients[1], -1.0, epsilon = 0.0001);
    }

    #[test]
    fn test_config_ref() {
        let config = Config {
            coefficients: [0.5, 0.25, 0.25],
        };
        let filter = Convolve::<f32, 3>::with_config(config.clone());
        let config_ref = filter.config_ref();
        assert_abs_diff_eq!(config_ref.coefficients[0], 0.5, epsilon = 0.0001);
        assert_abs_diff_eq!(config_ref.coefficients[1], 0.25, epsilon = 0.0001);
        assert_abs_diff_eq!(config_ref.coefficients[2], 0.25, epsilon = 0.0001);
    }

    #[test]
    fn test_config_clone() {
        let config = Config {
            coefficients: [0.5, 0.25, 0.25],
        };
        let filter = Convolve::<f32, 3>::with_config(config.clone());
        let cloned_config = filter.config();
        assert_abs_diff_eq!(cloned_config.coefficients[0], 0.5, epsilon = 0.0001);
        assert_abs_diff_eq!(cloned_config.coefficients[1], 0.25, epsilon = 0.0001);
        assert_abs_diff_eq!(cloned_config.coefficients[2], 0.25, epsilon = 0.0001);
    }

    #[test]
    fn test_state_mut() {
        use circular_buffer::CircularBuffer;

        let config = Config {
            coefficients: [1.0, 0.5],
        };
        let mut filter = Convolve::<f32, 2>::with_config(config);
        filter.filter(1.0);
        filter.filter(2.0);

        let state = filter.state_mut();
        // Modify the internal taps buffer
        state.taps = CircularBuffer::from([3.0, 4.0]);

        // Next filter call should use the modified state
        let output = filter.filter(5.0);
        // Output should be: 5.0 * 1.0 + 4.0 * 0.5 = 5.0 + 2.0 = 7.0
        assert_abs_diff_eq!(output, 7.0, epsilon = 0.0001);
    }

    #[test]
    fn test_from_into_guts() {
        use crate::traits::guts::{FromGuts, IntoGuts};

        let config = Config {
            coefficients: [1.0, 0.5],
        };
        let mut filter = Convolve::<f32, 2>::with_config(config.clone());
        filter.filter(3.0);
        filter.filter(4.0);

        let (guts_config, guts_state) = filter.into_guts();
        assert_abs_diff_eq!(guts_config.coefficients[0], 1.0, epsilon = 0.0001);
        assert_abs_diff_eq!(guts_config.coefficients[1], 0.5, epsilon = 0.0001);

        let filter2 = Convolve::from_guts((guts_config, guts_state));
        let output = filter2.config_ref();
        assert_abs_diff_eq!(output.coefficients[0], 1.0, epsilon = 0.0001);
    }

    #[test]
    fn test_reset() {
        let config = Config {
            coefficients: [1.0, 1.0, 1.0],
        };
        let mut filter = Convolve::<f32, 3>::with_config(config);

        // Fill the buffer
        filter.filter(1.0);
        filter.filter(2.0);
        filter.filter(3.0);
        let output = filter.filter(4.0);
        assert_abs_diff_eq!(output, 9.0, epsilon = 0.0001); // 2 + 3 + 4

        // Reset the filter
        let mut reset_filter = filter.reset();

        // After reset, buffer should be empty and fill again
        reset_filter.filter(10.0);
        reset_filter.filter(20.0);
        reset_filter.filter(30.0);
        let output = reset_filter.filter(40.0);
        assert_abs_diff_eq!(output, 90.0, epsilon = 0.0001); // 20 + 30 + 40
    }

    #[test]
    fn test_filter_buffer_filling() {
        // With zero-padded cold-start (tap buffer pre-filled with zeros),
        // the first N outputs are partial convolutions.
        let config = Config {
            coefficients: [0.25, 0.25, 0.25, 0.25],
        };
        let mut filter = Convolve::<f32, 4>::with_config(config);

        // taps=[0,0,0,0] → push(4.0) → taps=[0,0,0,4.0] → sum=1.0
        let output1 = filter.filter(4.0);
        assert_abs_diff_eq!(output1, 1.0, epsilon = 0.0001);

        // taps=[0,0,0,4.0] → push(8.0) → taps=[0,0,4.0,8.0] → sum=3.0
        let output2 = filter.filter(8.0);
        assert_abs_diff_eq!(output2, 3.0, epsilon = 0.0001);
    }

    #[test]
    fn coefficient_ordering() {
        // Asymmetric 3-tap coefficients to verify the pairing convention.
        // coeff[0] pairs with the newest tap, coeff[N-1] with the oldest tap.
        // coeff = [3, 2, 1], impulse = [1, 0, 0, ...] → impulse response:
        //   taps pre-filled with zeros: [0,0,0]
        //   filter(1): taps = [0,0,1]  →  output = 0*1 + 0*2 + 1*3 = 3 = h[0]
        //   filter(0): taps = [0,1,0]  →  output = 0*1 + 1*2 + 0*3 = 2 = h[1]
        //   filter(0): taps = [1,0,0]  →  output = 1*1 + 0*2 + 0*3 = 1 = h[2]
        //   filter(0): taps = [0,0,0]  →  output = 0
        let mut filter = Convolve::<f32, 3>::with_config(Config {
            coefficients: [3.0, 2.0, 1.0],
        });
        assert_abs_diff_eq!(filter.filter(1.0), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(filter.filter(0.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(filter.filter(0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(filter.filter(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(filter.filter(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cold_start_is_zero_padded_partial_convolution() {
        let h = [0.5_f32, -0.25, 0.125];
        let mut filter = Convolve::<f32, 3>::with_config(Config { coefficients: h });
        assert!((filter.filter(4.0) - 0.5 * 4.0).abs() < 1e-7);
        assert!((filter.filter(8.0) - (0.5 * 8.0 + -0.25 * 4.0)).abs() < 1e-7);
        assert!((filter.filter(2.0) - (0.5 * 2.0 + -0.25 * 8.0 + 0.125 * 4.0)).abs() < 1e-7);
    }

    #[test]
    fn impulse_response_equals_coefficients_n9() {
        let h = [0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9];
        let mut f = Convolve::<f32, 9>::with_config(Config { coefficients: h });
        let mut response = [0.0_f32; 9];
        response[0] = f.filter(1.0);
        for k in 1..9 {
            response[k] = f.filter(0.0);
        }
        for k in 0..9 {
            assert!(
                (response[k] - h[k]).abs() < 1e-7,
                "k={k}: got {} expected {}",
                response[k],
                h[k]
            );
        }
    }

    #[test]
    fn impulse_response() {
        // Verify the canonical FIR convolution contract y[n] = Σ h[k]·x[n−k]
        // with zero-padding (x[n] = 0 for n < 0).
        // The impulse response must reproduce h[0], h[1], …, h[N−1] exactly.
        let h = [0.1, 0.2, 0.3, 0.4, 0.5_f32];
        let mut filter = Convolve::<f32, 5>::with_config(Config { coefficients: h });
        let response: Vec<f32> = [1.0_f32]
            .into_iter()
            .chain(core::iter::repeat(0.0).take(h.len()))
            .map(|x| filter.filter(x))
            .collect();
        assert_abs_diff_eq!(response[0], h[0], epsilon = 1e-7);
        assert_abs_diff_eq!(response[1], h[1], epsilon = 1e-7);
        assert_abs_diff_eq!(response[2], h[2], epsilon = 1e-7);
        assert_abs_diff_eq!(response[3], h[3], epsilon = 1e-7);
        assert_abs_diff_eq!(response[4], h[4], epsilon = 1e-7);
        // After N+1 samples the buffer is fully zero again
        assert_abs_diff_eq!(response[5], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn integer_convolution() {
        // Integer convolution must work without overflow surprises.
        // 2-tap moving sum: output(n) = x[n] + x[n-1]
        let mut filter = Convolve::<i32, 2>::with_config(Config {
            coefficients: [1, 1],
        });
        // taps=[0,0], push(4): taps=[0,4], output = 0*1 + 4*1 = 4
        assert_eq!(filter.filter(4), 4);
        // taps=[0,4], push(6): taps=[4,6], output = 4*1 + 6*1 = 10
        assert_eq!(filter.filter(6), 10);
        // taps=[4,6], push(8): taps=[6,8], output = 6*1 + 8*1 = 14
        assert_eq!(filter.filter(8), 14);
        // Reset and check cold-start zero-padding
        let mut filter2 = filter.reset();
        assert_eq!(filter2.filter(1), 1);
        assert_eq!(filter2.filter(2), 3);
    }

    #[test]
    #[should_panic(expected = "denominator magnitude")]
    fn tiny_sum_rejected() {
        let _ = Convolve::<f32, 3>::normalized(Config {
            coefficients: [1.0, -1.0, f32::from_bits(1)],
        });
    }

    #[test]
    fn zero_sum_passes_through() {
        let f = Convolve::<f32, 3>::normalized(Config {
            coefficients: [1.0, -1.0, 0.0],
        });
        let c = f.config_ref().coefficients;
        assert_eq!([c[0], c[1], c[2]], [1.0, -1.0, 0.0]);
    }
}
