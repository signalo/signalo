mod pipe;
mod unit_pipe;

pub use self::pipe::*;
pub use self::unit_pipe::*;

// #[derive(Clone, Debug)]
// pub struct Pipe<T, U> {
//     lhs: T,
//     rhs: U,
// }
//
// #[derive(Clone, Debug)]
// pub struct SourcePipe<T, U> {
//     lhs: T,
//     rhs: U,
// }
//
// #[derive(Clone, Debug)]
// pub struct SinkPipe<T, U> {
//     lhs: T,
//     rhs: U,
// }
