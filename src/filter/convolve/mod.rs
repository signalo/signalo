use std::ops::BitOr;
use std::ops::{Add, Mul, Div};

use arraydeque::{Array, ArrayDeque, Wrapping};
use num_traits::Zero;

use piping::filter::Pipe;
use filter::Filter;

#[derive(Clone)]
pub struct Convolve<A>
where
    A: Array,
    A::Item: Copy,
{
    coefficients: A,
    state: ArrayDeque<A, Wrapping>,
}

impl<T, A> Convolve<A>
where
    T: Copy + PartialOrd + Zero,
    A: Array<Item=T>,
{
    #[inline]
    pub fn new(coefficients: A) -> Self {
        Convolve { coefficients, state: ArrayDeque::new() }
    }

    #[inline]
    pub fn coefficients(&self) -> &A {
        &self.coefficients
    }
}

impl<T, A> Convolve<A>
where
    T: Copy + PartialOrd + Zero + Div<T, Output = T>,
    A: Array<Item=T>,
{
    #[inline]
    pub fn normalized(mut coefficients: A) -> Self {
        let sum = coefficients.as_slice().iter().fold(T::zero(), |sum, coeff| {
            sum + (*coeff)
        });
        assert!(sum > T::zero());
        for coeff in coefficients.as_mut_slice() {
            *coeff = *coeff / sum;
        }
        Convolve { coefficients, state: ArrayDeque::new() }
    }
}

impl<T, A, Rhs> BitOr<Rhs> for Convolve<A>
where
    T: Copy,
    A: Array<Item = T>,
{
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T, A> Filter<T> for Convolve<A>
where
    T: Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>,
    A: Array<Item = T>,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        loop {
            if self.state.push_back(input).is_some() {
                break;
            }
        }

        let state_iter = self.state.iter();
        let coeff_iter = self.coefficients.as_slice().iter().rev();

        let output = state_iter.zip(coeff_iter).fold(T::zero(), |sum, (state, coeff)| {
            sum + ((*state) * (*coeff))
        });

        output
    }

    fn reset(&mut self) {
        self.state.clear()
    }

    fn phase_shift(&self) -> isize {
        0 // FIXME!!!
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0,
            8.0, 8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0,
            88.0, -101.0, 21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0,
            -13.0, 0.0, 0.0, 88.0, -93.0, 13.0, 0.0
        ]
    }

    #[test]
    fn floating_point() {
        // Effectively calculates the derivative:
        let filter = Convolve::new([1.000, -1.000]);
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
