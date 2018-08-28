// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Pipes compatible with implementations of `trait Sink`.

/// Convenience macros for assembling sink pipes.
#[macro_use]
pub mod macros {
    #[allow(unused_macros)]
    macro_rules! sink_pipe {
        ($($filters:expr),*) => ({
            #[allow(unused_imports)]
            use sink::{Pipe, UnitPipe};
            sink_pipe!(@internal $($filters),*)
        });
        (@internal $lhs:expr, $rhs:expr, $($filter:expr),*) => {
            let lhs = sink_pipe!(@internal $lhs, $rhs);
            let rhs = sink_pipe!(@internal $($filter),*);
            Pipe::new(lhs, rhs)
        };
        (@internal $lhs:expr, $rhs:expr) => {
            Pipe::new($lhs, $rhs)
        };
        (@internal $filter:expr) => {
            UnitPipe::new($filter)
        };
    }
}

mod pipe;
mod unit_pipe;

pub use self::pipe::*;
pub use self::unit_pipe::*;

#[cfg(test)]
mod tests {
    use super::*;

    use signalo_traits::filter::Filter;
    use signalo_traits::sink::Sink;

    struct DummyFilter;

    impl Filter<()> for DummyFilter {
        type Output = ();

        #[inline]
        fn filter(&mut self, _input: ()) -> Self::Output {
            ()
        }
    }

    struct DummySink;

    impl Sink<()> for DummySink {
        type Output = ();

        #[inline]
        fn sink(&mut self, _input: ()) {
            ()
        }

        #[inline]
        fn finalize(self) -> Self::Output {
            ()
        }
    }

    #[test]
    fn test() {
        let _: UnitPipe<_> = sink_pipe!(DummySink);
        let _: Pipe<_, _> = sink_pipe!(DummyFilter, DummySink);
    }
}
