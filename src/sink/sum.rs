use std::ops::Add;

use num_traits::Zero;

use sink::Sink;

pub struct Sum<T> {
    state: T,
}

impl<T> Sum<T>
where
    T: Zero,
{
    #[inline]
    pub fn new() -> Self {
        Sum { state: T::zero() }
    }
}

impl<T> Sink<T> for Sum<T>
where
    T: Copy + Add<T, Output=T>,
{
    type Output = T;

    #[inline]
    fn sink(&mut self, input: T) {
        self.state = self.state + input;
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sink() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let mut sink = Sum::new();
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, 196);
    }
}
