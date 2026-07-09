// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Core trait definitions for the signal processing framework.
//!
//! Defines the fundamental traits: [`Source`], [`Filter`],
//! [`MultirateFilter`], [`Sink`], [`Finalize`], and auxiliary traits for
//! configuration, state, and reset operations.

pub use guts;

pub mod filter;

pub mod finalize;

pub mod multirate_filter;

pub mod sink;

pub mod source;

pub use self::filter::Filter;
pub use self::finalize::Finalize;
pub use self::multirate_filter::MultirateFilter;
pub use self::sink::Sink;
pub use self::source::Source;

/// Trait for **configurable** systems that expose their configuration type.
///
/// This trait associates a configuration type with a system, typically used alongside
/// [`WithConfig`] to enable construction from configuration, or [`ConfigClone`]/[`ConfigRef`]
/// to access configuration state.
pub trait Config {
    /// The filter's configuration.
    type Config;
}

/// Trait for **config-constructable** systems.
///
/// Enables factory-pattern construction of filters, sources, and sinks from a configuration.
/// The configuration is provided at construction time and may be accessed later via
/// [`ConfigClone`] or [`ConfigRef`].
pub trait WithConfig: Config {
    /// The return type of `fn with_config(…)`.
    type Output;

    /// Creates an instance from a given config.
    #[must_use]
    fn with_config(config: Self::Config) -> Self::Output;
}

/// Trait for systems with cloneable configuration.
///
/// Provides mutable access to configuration state by cloning it. Use this when the configuration
/// can be cheaply copied or when mutations should not affect the internal state.
pub trait ConfigClone: Config {
    /// Returns the config.
    fn config(&self) -> Self::Config;
}

/// Trait for systems with borrowable configuration.
///
/// Provides non-mutable reference access to configuration without cloning. Use this for
/// read-only configuration inspection or when cloning is expensive.
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
    /// Returns a mutable reference to the internal state.
    ///
    /// # Warning
    ///
    /// Relying on the internal structure of a filter breaks encapsulation
    /// and risks violating internal invariants. Prefer using the public API.
    #[doc(hidden)]
    fn state_mut(&mut self) -> &mut Self::State;
}

/// Trait for **resettable** systems.
///
/// # Background:
///
/// Resettable systems **can be reset to their initial state**.
pub trait Reset: Sized {
    /// Returns an instance with a freshly reset internal state.
    #[must_use]
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
    /// # Safety
    ///
    /// This method is only sound when the active profile has `panic = "abort"`.
    /// If `panic = "unwind"` is in effect and a panic occurs inside `reset()`,
    /// this method triggers **undefined behaviour**: the `replace_with` crate
    /// will leave `self` in an invalid state while unwinding continues.
    ///
    /// You **must** have both `features = ["panic_abort", …]` in `Cargo.toml`
    /// **and** `panic = "abort"` in the relevant `[profile.*]` sections:
    ///
    /// ```toml
    /// [profile.release]
    /// panic = "abort"
    ///
    /// [profile.dev]
    /// panic = "abort"
    /// ```
    ///
    /// # Panics
    ///
    /// If the inner `reset()` call panics and the profile does **not** have
    /// `panic = "abort"`, behaviour is undefined.
    #[cfg(all(
        feature = "derive_reset_mut",
        not(feature = "std"),
        feature = "panic_abort"
    ))]
    fn reset_mut(&mut self) {
        // SAFETY: caller is responsible for ensuring `panic = "abort"` in
        // every active Cargo profile. Violating this causes UB on panic.
        unsafe { self.reset_mut_or_abort_unchecked() }
    }

    /// Resets the internal state.
    ///
    /// On panic (or to be more precise, unwinding) the process will **abort**
    /// to avoid returning control while `self` is in a potentially invalid state.
    #[cfg(feature = "std")]
    fn reset_mut_or_abort(&mut self) {
        replace_with::replace_with_or_abort(self, Reset::reset);
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
        // SAFETY: the caller guarantees that `panic = "abort"` is set in
        // every active Cargo profile, so unwinding cannot occur. Without
        // this guarantee `replace_with_or_abort_unchecked` causes UB on panic.
        replace_with::replace_with_or_abort_unchecked(self, Reset::reset)
    }
}
