#![cfg_attr(feature = "nightly", feature(try_from))]

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]

extern crate num_traits;
extern crate num_integer;

#[cfg(feature = "fpa")]
extern crate fpa;

#[cfg(feature = "fpa")]
extern crate typenum;

extern crate arraydeque;

#[cfg(test)]
#[macro_use]
extern crate nearly_eq;

pub mod source;
pub mod sink;
pub mod filter;
pub mod piping;

pub mod prelude {
    pub use source::Source;
    pub use sink::Sink;
    pub use filter::Filter;

    pub use piping::{
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
