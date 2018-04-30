mod iter;
mod constant;
mod increment;

pub use self::iter::Iter;
pub use self::constant::Constant;
pub use self::increment::Increment;

pub trait Source: Sized {
    type Output;

    fn source(&mut self) -> Option<Self::Output>;

    fn reset(&mut self) {
        // specialize for stateful source types
    }

    #[inline]
    fn phase_shift(&self) -> isize {
        0 // specialize for phase-shifting source types
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
