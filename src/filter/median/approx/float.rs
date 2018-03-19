use std::ops::BitOr;
use std::ops::{Sub, Add, Mul};

use num_traits::Float;

use filter::pipe::Pipe;
use filter::Filter;

use filter::mean::approx::float::Mean;

#[derive(Clone, Debug)]
pub struct Median<T> {
    beta: T,
    mean: (Mean<T>, Mean<T>),
    state: Option<T>,
}

impl<T> Median<T>
where
    T: Clone + Float + From<f32> + Mul<T, Output=T>
{
    #[inline]
    pub fn new(beta: T) -> Self {
        let mean = (Mean::new(beta.clone()), Mean::new(beta * (0.5).into()));
        Median { beta, mean, state: None }
    }

    pub fn beta(&self) -> T {
        self.beta.clone()
    }

    pub fn mean(&self) -> &(Mean<T>, Mean<T>) {
        &self.mean
    }
}

impl_pipe!(Median<T>);

impl<T> Filter<T> for Median<T>
where
    T: Clone + Float + Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        // We calculate the mean and use it as an estimate of the median:
        let mean = self.mean.0.apply(input.clone());
        // We then calculate the approximate of the median:
        let median = match self.state.clone() {
            None => {
                mean
            },
            Some(mut state) => {
                state.clone() + ((mean - state) * self.beta)
            }
        };
        // The approximated median tends to oscillate,
        // so we apply another mean to smoothen those out:
        let state = self.mean.1.apply(median);
        // And we're done. Store a copy and return the result:
        self.state = Some(state.clone());
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

    #[test]
    fn floating_point() {
        let filter = Median::new(0.5);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0, 11.0, 24.0, 24.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.000, 0.063, 0.523, 0.817, 1.207, 1.803, 2.950, 3.456, 4.648, 5.254, 6.066, 6.605, 6.990, 7.784, 8.708, 8.817, 9.064, 9.856, 10.836, 11.025, 10.856, 11.042, 11.370, 11.428, 12.177, 12.368, 18.616, 21.311, 22.284, 22.441, 27.733, 28.627, 28.854, 27.962, 26.637, 25.705, 25.003, 24.446, 24.799, 23.904, 28.831, 29.684, 30.015, 29.284, 28.133, 26.872, 31.140, 31.749, 31.531, 30.965], 0.001);
    }
}
