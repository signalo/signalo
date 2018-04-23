use std::ops::BitOr;
use std::ops::Sub as StdSub;


use filter::pipe::Pipe;
use filter::Filter;

#[derive(Clone, Debug)]
pub struct Sub<T> {
    value: T
}

impl<T> Sub<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Sub { value }
    }

    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T, Rhs> BitOr<Rhs> for Sub<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T, U> Filter<U> for Sub<T>
where
    T: Copy,
    U: StdSub<T>,
{
    type Output = <U as StdSub<T>>::Output;

    #[inline]
    fn apply(&mut self, input: U) -> Self::Output {
        input - self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Sub::new(42);
        let input = vec![42, 43, 49, 44, 47, 50, 58, 45, 61, 48, 56, 51, 51, 59, 59, 46, 54, 62, 62, 49];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7]);
    }

    #[test]
    fn floating_point() {
        let filter = Sub::new(4.2);
        let input = vec![4.2, 5.2, 11.2, 6.2, 9.2, 12.2, 20.2, 7.2, 23.2, 10.2, 18.2, 13.2, 13.2, 21.2, 21.2, 8.2, 16.2, 24.2, 24.2, 11.2];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0]);
    }
}
