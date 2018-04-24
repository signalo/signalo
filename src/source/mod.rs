pub trait Source: Sized {
    type Output;

    fn source(&mut self) -> Option<Self::Output>;

    fn reset(&mut self) {
        // specialize for stateful source types
    }
}
