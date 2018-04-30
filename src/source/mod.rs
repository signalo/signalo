mod iter;
mod constant;

pub use self::iter::Iter;
pub use self::constant::Constant;

pub trait Source: Sized {
    type Output;

    fn source(&mut self) -> Option<Self::Output>;

    fn reset(&mut self) {
        // specialize for stateful source types
    }

    #[inline]
    fn phase_shift(&self) -> isize {
        0 // specialize for phase-shifting filter types
    }
}

impl<F, T> Source for F
where
    F: FnMut() -> Option<T>,
{
    type Output = T;

    fn source(&mut self) -> Option<Self::Output> {
        self()
    }
}
