// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of traits used in 'signalo' crates.

#![cfg_attr(not(feature = "std"), no_std)]

#![warn(missing_docs)]

extern crate guts;

pub use guts::{Guts, FromGuts, IntoGuts};

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

/// Trait for **resettable** systems.
///
/// # Background:
///
/// Resettable systems **can be reset to their initial state**.
pub trait Reset: Sized {
    /// Returns an instance with a freshly reset internal state.
    fn reset(self) -> Self;
}

/// Trait for mutably **resettable** systems.
///
/// # Background:
///
/// Resettable systems **can be reset to their initial state**.
pub trait ResetMut: Reset {
    /// Resets the internal state.
    #[cfg(not(feature = "derive_reset_mut"))]
    fn reset_mut(&mut self);

    /// Resets the internal state.
    ///
    /// On panic (or to be more precise, unwinding) the process will **abort**
    /// to avoid returning control while `self` is in a potentially invalid state.
    #[cfg(all(feature = "derive_reset_mut", feature = "std"))]
    fn reset_mut(&mut self) {
        self.reset_mut_or_abort()
    }

    /// Resets the internal state.
    ///
    /// On panic (or to be more precise, unwinding) the process will **abort**
    /// to avoid returning control while `self` is in a potentially invalid state.
    ///
    /// You are expected to have `features = ["panic_abort", …]` defined in `Cargo.toml`
    /// and `panic = "abort"` defined in your profile for it to behave semantically correct:
    ///
    /// ```toml
    /// # Cargo.toml
    ///
    /// [profile.debug]
    /// panic = "abort"
    ///
    /// [profile.release]
    /// panic = "abort"
    /// ```
    ///
    /// # Safety
    ///
    /// It is crucial to only ever use this function having defined `panic = "abort"`, or else bad
    /// things may happen. It's *up to you* to uphold this invariant!
    #[cfg(all(feature = "derive_reset_mut", not(feature = "std"), feature = "panic_abort"))]
    fn reset_mut(&mut self) {
        unsafe {
            self.reset_mut_or_abort_unchecked()
        }
    }

    /// Resets the internal state.
    ///
    /// On panic (or to be more precise, unwinding) the process will **abort**
    /// to avoid returning control while `self` is in a potentially invalid state.
    #[cfg(feature = "std")]
    fn reset_mut_or_abort(&mut self) {
        replace_with::replace_with_or_abort(self, |owned_self| owned_self.reset())
    }

    /// Resets the internal state.
    ///
    /// On panic (or to be more precise, unwinding) the process will **abort**
    /// to avoid returning control while `self` is in a potentially invalid state.
    ///
    /// You are expected to have `features = ["panic_abort", …]` defined in `Cargo.toml`
    /// and `panic = "abort"` defined in your profile for it to behave semantically correct:
    ///
    /// ```toml
    /// # Cargo.toml
    ///
    /// [profile.debug]
    /// panic = "abort"
    ///
    /// [profile.release]
    /// panic = "abort"
    /// ```
    ///
    /// # Safety
    ///
    /// It is crucial to only ever use this function having defined `panic = "abort"`, or else bad
    /// things may happen. It's *up to you* to uphold this invariant!
    #[cfg(feature = "panic_abort")]
    unsafe fn reset_mut_or_abort_unchecked(&mut self) {
        replace_with::replace_with_or_abort_unchecked(self, |owned_self| owned_self.reset())
    }
}
