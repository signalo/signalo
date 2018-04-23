mod iter;

pub use self::iter::Iter;

pub trait Source: Iterator {
    type Output;

    fn next(&mut self) -> Self::Output;

    fn reset(&mut self) {
        // specialize for stateful source types
    }
}
