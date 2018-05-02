// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[macro_use]
pub mod macros {
    #[allow(unused_macros)]
    macro_rules! filter_pipe {
        ($($filters:expr),*) => ({
            #[allow(unused_imports)]
            use filter::{Pipe, UnitPipe};
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

mod pipe;
mod unit_pipe;

pub use self::pipe::*;
pub use self::unit_pipe::*;

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
    fn filter_pipe() {
        let _: UnitPipe<_> = filter_pipe!(DummyFilter);
        let _: Pipe<_, _> = filter_pipe!(DummyFilter, DummyFilter);
    }
}
