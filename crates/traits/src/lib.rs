// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A collection of traits used in 'signalo' crates.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub use guts;

/// Signal transformers that accept an input value and produce an output.
///
/// Filters are the primary building blocks for signal processing pipelines. They transform
/// input signals according to their internal state and logic, supporting complex operations
/// like filtering, decimation, and statistical analysis.
pub mod filter;

/// Trait for extracting results from pipelines.
///
/// Finalizers consume the pipeline and produce a final output value, typically used to
/// extract accumulated results (sums, means, collected values, etc.) from sinks or filters.
pub mod finalize;

/// Signal consumers that process values without direct output.
///
/// Sinks accept values from filters or sources and perform side effects or accumulate results
/// (writing to buffers, computing statistics, etc.). They must be paired with a [`Finalize`]
/// implementation to extract the final result.
pub mod sink;

/// Signal generators that produce values when queried.
///
/// Sources are the entry points for signal processing pipelines. They generate or provide
/// signal values on demand, such as waveform generators or iterators over buffered data.
pub mod source;

pub use filter::Filter;
pub use finalize::Finalize;
pub use sink::Sink;
pub use source::Source;

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
    #[cfg(all(
        feature = "derive_reset_mut",
        not(feature = "std"),
        feature = "panic_abort"
    ))]
    fn reset_mut(&mut self) {
        unsafe { self.reset_mut_or_abort_unchecked() }
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
