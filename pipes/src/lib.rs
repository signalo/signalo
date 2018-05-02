// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]

extern crate signalo_traits;
extern crate signalo_filters;

pub mod source;
pub mod filter;
pub mod sink;

pub mod prelude {
    pub use {
        source::{
            // macros::*,
            UnitPipe as SourceUnitPipe,
            Pipe as SourcePipe
        },
        filter::{
            macros::*,
            UnitPipe as FilterUnitPipe,
            Pipe as FilterPipe
        },
        sink::{
            // macros::*,
            UnitPipe as SinkUnitPipe,
            Pipe as SinkPipe
        },
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
