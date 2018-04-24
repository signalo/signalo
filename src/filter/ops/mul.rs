use std::ops::BitOr;
use std::ops::Mul as StdMul;

use piping::filter::Pipe;
use filter::Filter;

#[derive(Clone, Debug)]
pub struct Mul<T> {
    value: T
}

impl<T> Mul<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Mul { value }
    }

    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T, Rhs> BitOr<Rhs> for Mul<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T, U> Filter<U> for Mul<T>
where
    T: Copy,
    U: StdMul<T>,
{
    type Output = <U as StdMul<T>>::Output;

    #[inline]
    fn filter(&mut self, input: U) -> Self::Output {
        input * self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Mul::new(42);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 42, 294, 84, 210, 336, 672, 126, 798, 252, 588, 378, 378, 714, 714, 168, 504, 840, 840, 294]);
    }

    #[test]
    fn floating_point() {
        let filter = Mul::new(4.2);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 4.2, 29.4, 8.4, 21.0, 33.6, 67.2, 12.6, 79.8, 25.2, 58.8, 37.8, 37.8, 71.4, 71.4, 16.8, 50.4, 84.0, 84.0, 29.4]);
    }
}
