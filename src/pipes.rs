// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Pipeline composition utilities.
//!
//! Provides adapters and macros for connecting Sources, Filters, and Sinks in composable chains.

pub use crate::traits;

pub mod pipe;

pub mod unit_pipe;

/// Convenience macros for assembling filter pipes.
#[macro_use]
pub mod macros {
    #[allow(unused_macros)]
    macro_rules! pipe {
        ($($filters:expr),*) => ({
            #[allow(unused_imports)]
            use filter::{pipe::Pipe, unit_pipe::UnitPipe};
            pipe!(@internal $($filters),*)
        });
        (@internal $lhs:expr, $rhs:expr, $($tail:expr),*) => ({
            let lhs = pipe!(@internal $lhs, $rhs);
            let rhs = pipe!(@internal $($tail),*);
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
