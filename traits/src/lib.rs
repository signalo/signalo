// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of traits used in 'signalo' crates.

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

pub mod filter;
pub mod sink;
pub mod source;

pub use filter::Filter;
pub use sink::Sink;
pub use source::Source;

/// The crate's prelude.
pub mod prelude {
    pub use filter::{self, Filter};
    pub use sink::{self, Sink};
    pub use source::{self, Source};
}

/// Trait for **stateful** systems.
///
/// # Background:
///
/// Stateful systems **can react to the same input differently depending on the current state**.
pub trait Stateful: Sized {
    /// The filter's internal state.
    type State;
}

/// Unsafe trait for accessing the state of **stateful** systems.
///
/// # Caution:
///
/// Use methods ins this trait with caution and care, as their use harms encapsulation. If possible avoid.
/// As such any state exposed through `StatefulUnsafe` is to be considered an unstable implementation
/// detail, as internal refactoring can cause breaking changes in connected systems any time.
pub unsafe trait StatefulUnsafe: Stateful {
    /// Returns a mutable reference to the internal state of the filter.
    unsafe fn state(&self) -> &Self::State;

    /// Returns a mutable reference to the internal state of the filter.
    unsafe fn state_mut(&mut self) -> &mut Self::State;
}

/// Axiliary trait for **stateful** systems.
///
/// # Background:
///
/// `InitialState` is an auxiliary trait intended for use in combination with `Resettable`:
///
/// ```rust, ignore
/// impl Resettable for Scale {
///     fn reset(&mut self) {
///         self.state = Self::initial_state(...);
///     }
/// }
/// ```
pub trait InitialState<T>: Stateful {
    /// Returns the filter's initial state for a given parameter.
    fn initial_state(parameter: T) -> Self::State;
}

/// Trait for **resettable** systems.
///
/// # Background:
///
/// Resettable systems **can be reset to their initial state**.
pub trait Resettable: Stateful {
    /// Resets the internal state of the filter.
    fn reset(&mut self);
}
