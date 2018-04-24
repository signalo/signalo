use std::ops::BitOr;

use piping::filter::Pipe;
use filter::Filter;

#[derive(Default, Clone)]
pub struct Identity;

impl<Rhs> BitOr<Rhs> for Identity {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T> Filter<T> for Identity {
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Identity::default();
        let input = vec![1, 1, 2, 3, 5, 8, 13, 21, 34];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![1, 1, 2, 3, 5, 8, 13, 21, 34]);
    }

    #[test]
    fn floating_point() {
        let filter = Identity::default();
        let input = vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]);
    }
}
