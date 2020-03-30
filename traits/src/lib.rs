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
    ///
    /// # Safety
    ///
    /// Relying on the internal structure of an object breaks encapsulation,
    /// putting your code at risk of breaking your invariants.
    unsafe fn state_mut(&mut self) -> &mut Self::State;
}

/// Trait for **gut-exposing** systems.
pub trait Guts: Sized {
    /// The system's guts.
    type Guts;
}

/// Trait for **gut-constructable** systems.
pub trait FromGuts: Guts {
    /// Constructs `self` from its raw guts.
    ///
    /// # Safety
    ///
    /// It is your responsibility to ensure that you're not breaking your code's invariants.
    unsafe fn from_guts(guts: Self::Guts) -> Self;
}

/// Trait for **gut-destructable** systems.
pub trait IntoGuts: Guts {
    /// Destructs `self` into its raw guts.
    fn into_guts(self) -> Self::Guts;
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
    #[cfg(all(feature = "std", not(feature = "panic_abort")))]
    fn reset_mut(&mut self) {
        replace_with::replace_with_or_abort(self, |owned_self| owned_self.reset())
    }

    /// Resets the internal state.
    #[cfg(feature = "panic_abort")]
    fn reset_mut(&mut self) {
        unsafe {
            replace_with::replace_with_or_abort_unchecked(self, |owned_self| owned_self.reset())
        }
    }
}
