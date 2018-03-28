use std::ops::BitOr;
use std::ops::Add as StdAdd;

use filter::pipe::Pipe;
use filter::Filter;

#[derive(Clone, Debug)]
pub struct Add<T> {
    value: T
}

impl<T> Add<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Add { value }
    }
}

impl_pipe!(Add<T>);

impl<T, U> Filter<U> for Add<T>
where
    T: Copy,
    U: StdAdd<T>,
{
    type Output = <U as StdAdd<T>>::Output;

    #[inline]
    fn apply(&mut self, input: U) -> Self::Output {
        input + self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Add::new(42);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![42, 43, 49, 44, 47, 50, 58, 45, 61, 48, 56, 51, 51, 59, 59, 46, 54, 62, 62, 49]);
    }

    #[test]
    fn floating_point() {
        let filter = Add::new(4.2);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, vec![4.2, 5.2, 11.2, 6.2, 9.2, 12.2, 20.2, 7.2, 23.2, 10.2, 18.2, 13.2, 13.2, 21.2, 21.2, 8.2, 16.2, 24.2, 24.2, 11.2]);
    }
}
