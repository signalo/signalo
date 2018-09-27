// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Pipes compatible with implementations of `trait Filter`.

/// Convenience macros for assembling filter pipes.
#[macro_use]
pub mod macros {
    #[allow(unused_macros)]
    macro_rules! filter_pipe {
        ($($filters:expr),*) => ({
            #[allow(unused_imports)]
            use filter::{pipe::Pipe, unit_pipe::UnitPipe};
            filter_pipe!(@internal $($filters),*)
        });
        (@internal $lhs:expr, $rhs:expr, $($tail:expr),*) => ({
            let lhs = filter_pipe!(@internal $lhs, $rhs);
            let rhs = filter_pipe!(@internal $($tail),*);
            Pipe::new(lhs, rhs)
        });
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

    struct DummyFilter;

    impl Filter<()> for DummyFilter {
        type Output = ();

        #[inline]
        fn filter(&mut self, _input: ()) -> Self::Output {
            ()
        }
    }

    #[test]
    fn test() {
        let _: unit_pipe::UnitPipe<_> = filter_pipe!(DummyFilter);
        let _: pipe::Pipe<_, _> = filter_pipe!(DummyFilter, DummyFilter);
    }
}
