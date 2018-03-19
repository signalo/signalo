use std::ops::BitOr;
use std::ops::{Sub, Add, Shl, Shr};

use num_integer::Integer;

use filter::pipe::Pipe;
use filter::Filter;

use filter::mean::approx::fixed::Mean;

#[derive(Clone, Debug)]
pub struct Median<T> {
    beta: usize,
    shift: usize,
    mean: (Mean<T>, Mean<T>),
    state: Option<T>,
}

impl<T> Median<T> {
    #[inline]
    pub fn new(beta: usize, shift: usize) -> Self {
        let mean = (Mean::new(beta, shift), Mean::new(beta + 1, shift));
        Median { beta, shift, mean, state: None }
    }

    pub fn beta(&self) -> usize {
        self.beta
    }

    pub fn shift(&self) -> usize {
        self.shift
    }

    pub fn mean(&self) -> &(Mean<T>, Mean<T>) {
        &self.mean
    }
}

impl_pipe!(Median<T>);

impl<T> Filter<T> for Median<T>
where
    T: Clone + Integer + Add<T, Output=T> + Sub<T, Output=T> + Shl<usize, Output=T> + Shr<usize, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let input = input << self.shift;
        // We calculate the mean and use it as an estimate of the median:
        let mean = self.mean.0.apply(input.clone());
        // We then calculate the approximate of the median:
        let median = match self.state.clone() {
            None => {
                mean
            },
            Some(mut state) => {
                state.clone() + ((mean - state) >> self.beta)
            }
        };
        // The approximated median tends to oscillate,
        // so we apply another mean to smoothen those out:
        let state = self.mean.1.apply(median);
        // And we're done. Store a copy and return the result:
        self.state = Some(state.clone());
        state >> self.shift
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
    fn fixed() {
        let filter = Median::new(1, 0);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7, 7, 15, 15, 10, 23, 10, 111, 18, 18, 18, 106, 5, 26, 13, 13, 21, 21, 21, 34, 8, 109, 8, 29, 16, 16, 16, 104, 11, 24, 24];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 8, 14, 17, 18, 18, 23, 24, 24, 23, 22, 21, 20, 20, 20, 19, 24, 25, 25, 24, 23, 22, 26, 27, 27, 27]);
    }
}
