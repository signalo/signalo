// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Pipes compatible with implementations of `trait Source`.

/// Convenience macros for assembling source pipes.
#[macro_use]
pub mod macros {
    #[allow(unused_macros)]
    macro_rules! source_pipe {
        ($($filters:expr),*) => ({
            #[allow(unused_imports)]
            use source::{pipe::Pipe, unit_pipe::UnitPipe};
            source_pipe!(@internal $($filters),*)
        });
        (@internal $lhs:expr, $rhs:expr, $($filters:expr),*) => {
            let lhs = source_pipe!(@internal $lhs, $rhs);
            let rhs = source_pipe!(@internal $($filters),*);
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

pub mod pipe;
pub mod unit_pipe;

#[cfg(test)]
mod tests {
    use super::*;

    use signalo_traits::filter::Filter;
    use signalo_traits::source::Source;

    struct DummyFilter;

    impl Filter<()> for DummyFilter {
        type Output = ();

        #[inline]
        fn filter(&mut self, _input: ()) -> Self::Output {
            ()
        }
    }

    struct DummySource;

    impl Source for DummySource {
        type Output = ();

        #[inline]
        fn source(&mut self) -> Option<Self::Output> {
            None
        }
    }

    #[test]
    fn test() {
        let _: unit_pipe::UnitPipe<_> = source_pipe!(DummySource);
        let _: pipe::Pipe<_, _> = source_pipe!(DummySource, DummyFilter);
    }
}
