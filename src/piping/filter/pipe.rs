use std::ops::BitOr;

use filter::Filter;

/// A `Pipe` is a simple container joining a pair of `Filter`s
///
/// ```plain
/// ════════════ + ════════════
///  ╭────────╮  +  ╭────────╮
///  │ Filter │  +  │ Filter │
///  ╰────────╯  +  ╰────────╯
/// ════════════ + ════════════
/// └─┬───────────────────────┘
///   └ Pipe
/// ```
#[derive(Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U> {
    #[inline]
    pub fn new(lhs: T, rhs: U) -> Self {
        Self { lhs, rhs }
    }
}

impl<T, U> From<(T, U)> for Pipe<T, U> {
    #[inline]
    fn from(parts: (T, U)) -> Self {
        let (lhs, rhs) = parts;
        Self::new(lhs, rhs)
    }
}

impl<T, U, Rhs> BitOr<Rhs> for Pipe<T, U> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, U, I> Filter<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Filter<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.rhs.filter(self.lhs.filter(input))
    }

    #[inline]
    fn phase_shift(&self) -> isize {
        self.rhs.phase_shift() + self.lhs.phase_shift()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Value = usize;

    struct DummyFilterAdd;

    impl Filter<Value> for DummyFilterAdd {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input + 1
        }
    }

    struct DummyFilterMul;

    impl Filter<Value> for DummyFilterMul {
        type Output = Value;

        #[inline]
        fn filter(&mut self, input: Value) -> Self::Output {
            input * 2
        }
    }

    #[test]
    fn filter() {
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let filter_add = DummyFilterAdd;
        let filter_mul = DummyFilterMul;
        let pipe = Pipe::new(filter_add, filter_mul);
        let subject: Vec<_> = input.iter().scan(pipe, |pipe, &input| {
            Some(pipe.filter(input))
        }).collect();
        let expected = vec![2, 4, 16, 6, 12, 18, 34, 8, 40, 14, 30, 20, 20, 36, 36, 10, 26, 42, 42, 16];
        assert_eq!(subject, expected);
    }
}
