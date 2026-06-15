// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/// Implements the standard Config/State/Guts/Reset trait suite for an oscillator.
///
/// The oscillator struct must have fields `config` (of type `Config<T>`) and `state`
/// (of type `State<T>`). The `State<T>` type must implement `Default`.
///
/// # Parameters
///
/// - `$osc`; the oscillator type name (e.g. `SineOscillator`)
/// - `$($bound)*`; the where-clause bounds for `WithConfig`, `Default`, and `Reset`
///   (e.g. `T: num_traits::float::FloatCore`)
macro_rules! impl_oscillator_traits {
    ($osc:ident, $($bound:tt)*) => {
        impl<T> $crate::traits::Config for $osc<T> {
            type Config = Config<T>;
        }

        impl<T> $crate::traits::State for $osc<T> {
            type State = State<T>;
        }

        impl<T> $crate::traits::WithConfig for $osc<T>
        where
            $($bound)*
        {
            type Output = Self;

            fn with_config(config: Self::Config) -> Self::Output {
                Self {
                    config,
                    state: State::default(),
                }
            }
        }

        impl<T> Default for $osc<T>
        where
            $($bound)*
        {
            fn default() -> Self {
                Self {
                    config: Config::default(),
                    state: State::default(),
                }
            }
        }

        impl<T> $crate::traits::ConfigRef for $osc<T> {
            fn config_ref(&self) -> &Self::Config {
                &self.config
            }
        }

        impl<T> $crate::traits::ConfigClone for $osc<T>
        where
            Config<T>: Clone,
        {
            fn config(&self) -> Self::Config {
                self.config.clone()
            }
        }

        impl<T> $crate::traits::StateMut for $osc<T> {
            #[doc(hidden)]
            fn state_mut(&mut self) -> &mut Self::State {
                // SAFETY: `&mut self` guarantees exclusive access; no other references
                // to state exist within the program at this point.
                &mut self.state
            }
        }

        impl<T> $crate::traits::guts::HasGuts for $osc<T> {
            type Guts = (Config<T>, State<T>);
        }

        impl<T> $crate::traits::guts::FromGuts for $osc<T> {
            fn from_guts(guts: Self::Guts) -> Self {
                let (config, state) = guts;
                Self { config, state }
            }
        }

        impl<T> $crate::traits::guts::IntoGuts for $osc<T> {
            fn into_guts(self) -> Self::Guts {
                (self.config, self.state)
            }
        }

        impl<T> $crate::traits::Reset for $osc<T>
        where
            $($bound)*
        {
            fn reset(self) -> Self {
                Self {
                    config: self.config,
                    state: State::default(),
                }
            }
        }

        #[cfg(feature = "derive")]
        impl<T> $crate::traits::ResetMut for $osc<T> where Self: $crate::traits::Reset {}
    };
}
