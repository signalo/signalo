use std::ops::BitOr;
use std::ops::{Sub, Add, Mul, Div};

use num_traits::{Zero, One};

use filter::pipe::Pipe;
use filter::Filter;

use filter::mean::approx::Mean;

#[derive(Clone, Debug)]
pub struct Median<T> {
    beta: T,
    mean: (Mean<T>, Mean<T>),
    state: Option<T>,
}

impl<T> Median<T>
where
    T: PartialOrd + Zero + One
{
    /// Recommended values:
    // - `alpha`: `beta`
    // - `beta`: `0.0 .. 1.0`
    // - `gamma`: `beta * 0.5`
    #[inline]
    pub fn new(alpha: T, beta: T, gamma: T) -> Self {
        assert!(alpha > T::zero() && alpha <= T::one());
        assert!(beta > T::zero() && beta <= T::one());
        assert!(gamma > T::zero() && gamma <= T::one());
        let mean = (Mean::new(alpha), Mean::new(gamma));
        Median { beta, mean, state: None }
    }

    pub fn beta(&self) -> &T {
        &self.beta
    }

    pub fn mean(&self) -> &(Mean<T>, Mean<T>) {
        &self.mean
    }
}

impl_pipe!(Median<T>);

impl<T> Filter<T> for Median<T>
where
    T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        // We calculate the mean and use it as an estimate of the median:
        let mean = self.mean.0.apply(input);
        // We then calculate the approximate of the median:
        let median = match self.state {
            None => {
                mean
            },
            Some(mut state) => {
                state + ((mean - state) * self.beta)
            }
        };
        // The approximated median tends to oscillate,
        // so we apply another mean to smoothen those out:
        let state = self.mean.1.apply(median);
        // And we're done. Store a copy and return the result:
        self.state = Some(state);
        state
    }

    fn reset(&mut self) {
        self.mean.0.reset();
        self.mean.1.reset();
        self.state = None;
    }

    fn phase_shift(&self) -> isize {
        0 // FIXME!!!
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
            0.000, 0.063, 0.523, 0.817, 1.207, 1.803, 2.950, 3.456, 4.648, 5.254, 6.066, 6.605,
            6.990, 7.784, 8.708, 8.817, 9.064, 9.856, 10.836, 11.025, 10.856, 11.042, 11.370,
            11.428, 12.177, 12.368, 18.616, 21.311, 22.284, 22.441, 27.733, 28.627, 28.854,
            27.962, 26.637, 25.705, 25.003, 24.446, 24.799, 23.904, 28.831, 29.684, 30.015,
            29.284, 28.133, 26.872, 31.140, 31.749, 31.531, 30.965
        ]
    }

    #[test]
    fn floating_point() {
        let alpha = 0.5;
        let beta = 0.5;
        let gamma = 0.25;

        let filter = Median::new(alpha, beta, gamma);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }

    #[test]
    #[cfg(feature = "fpa")]
    fn fixed_point() {
        use std::convert::TryFrom;
        use fpa::I16F16;
        type Fixed = I16F16;

        let alpha = Fixed::try_from(0.5).unwrap();
        let beta = Fixed::try_from(0.5).unwrap();
        let gamma = Fixed::try_from(0.25).unwrap();
        let filter = Median::new(alpha, beta, gamma);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let float_input = get_input();
        println!("input: {:?}", float_input);

        let input: Vec<_> = float_input.into_iter().map(|float| {
            Fixed::try_from(float).unwrap()
        }).collect();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();

        let float_output: Vec<_> = output.into_iter().map(|fixed| {
            f32::try_from(fixed).unwrap()
        }).collect();
        println!("output: {:?}", float_output);

        assert_nearly_eq!(float_output, get_output(), 0.001);
    }
}
