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
pub mod finalize;
pub mod sink;
pub mod source;

pub use filter::Filter;
pub use finalize::Finalize;
pub use sink::Sink;
pub use source::Source;

/// The crate's prelude.
pub mod prelude {}

/// Trait for **configurable** systems.
pub trait Config {
    /// The filter's configuration.
    type Config;
}

/// Trait for **config-constructable** systems.
pub trait WithConfig: Config {
    /// The return type of `fn with_config(…)`.
    type Output;

    /// Creates an instance from a given config.
    fn with_config(config: Self::Config) -> Self::Output;
}

/// Trait for **configurable** systems.
pub trait ConfigClone: Config {
    /// Returns the config.
    fn config(&self) -> Self::Config;
}

/// Trait for **configurable** systems.
pub trait ConfigRef: Config {
    /// Returns a reference to the config.
    fn config_ref(&self) -> &Self::Config;
}

/// Trait for **stateful** systems.
///
/// # Background:
///
/// Stateful systems **can react to the same input differently depending on the current state**.
pub trait State: Sized {
    /// The filter's internal state.
    type State;
}

/// Trait for systems with mutably accessible state.
pub trait StateMut: State {
    /// Returns a mutable reference to the internal state of the filter.
    /// # Important:
    ///
    /// Relying on the internal structure of an object breaks encapsulation.
    unsafe fn state_mut(&mut self) -> &mut Self::State;
}

/// Trait for **destructable** systems.
pub trait Destruct: Sized {
    /// The return type of `fn into_guts(…)`.
    type Output;

    /// Destructs `self` into its `self.config` and `self.state` components.
    fn destruct(self) -> Self::Output;
}

/// Trait for **resettable** systems.
///
/// # Background:
///
/// Reset systems **can be reset to their initial state**.
pub trait Reset: Sized {
    /// Returns an instance with a freshly reset internal state.
    fn reset(self) -> Self;

    /// Resets the internal state.
    fn reset_mut(&mut self) {
        use std::{mem, ptr};

        unsafe {
            let owned_self = ptr::replace(self, mem::uninitialized());
            ptr::write(self, owned_self.reset());
        }
    }
}
