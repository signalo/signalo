use std::ops::Mul;

use filter::Filter;

#[derive(Clone, Debug)]
pub struct Square;

impl Square {
    #[inline]
    pub fn new() -> Self {
        Square
    }
}

impl<T> Filter<T> for Square
where
    T: Copy + Mul<T>
{
    type Output = <T as Mul<T>>::Output;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        input * input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Square::new();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 1, 49, 4, 25, 64, 256, 9, 361, 36, 196, 81, 81, 289, 289, 16, 144, 400, 400, 49]);
    }

    #[test]
    fn floating_point() {
        let filter = Square::new();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 1.0, 49.0, 4.0, 25.0, 64.0, 256.0, 9.0, 361.0, 36.0, 196.0, 81.0, 81.0, 289.0, 289.0, 16.0, 144.0, 400.0, 400.0, 49.0]);
    }
}
