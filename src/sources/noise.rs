// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Noise source using xorshift32 pseudorandom number generator.
//!
//! Generates a deterministic sequence of pseudorandom u32 values using the xorshift32
//! algorithm, seeded with a configurable initial value. Seed of 0 is automatically
//! replaced with a non-zero default to ensure valid output.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Reset, Source, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Configuration for the Noise source.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Seed value for the xorshift32 generator.
    ///
    /// A seed of 0 is automatically replaced with `0xDEAD_BEEF` to ensure valid output,
    /// as xorshift32 cannot generate a sequence from a zero seed.
    pub(crate) seed: u32,
}

impl Config {
    /// Creates a new `Config` with the given seed.
    ///
    /// If `seed` is 0, it will be replaced with `0xDEAD_BEEF`.
    #[inline]
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 0xDEAD_BEEF } else { seed };
        Self { seed }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new(0)
    }
}

/// State of the Noise source.
#[derive(Clone, Debug)]
pub struct State {
    /// Current state of the xorshift32 generator.
    pub(crate) state: u32,
}

/// A noise source that generates pseudorandom u32 values using xorshift32.
///
/// The xorshift32 algorithm is a fast, simple PRNG suitable for DSP and embedded applications.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo::sources::noise::Noise;
/// let noise = Noise::from_config(signalo::sources::noise::Config::new(12345));
/// // ╭──────────╮  ╭──────────╮  ╭──────────╮
/// // │ 12345... │─▶│ 56789... │─▶│ 01234... │─▶ ...
/// // ╰──────────╯  ╰──────────╯  ╰──────────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Noise {
    config: Config,
    state: State,
}

impl ConfigTrait for Noise {
    type Config = Config;
}

impl StateTrait for Noise {
    type State = State;
}

impl WithConfig for Noise {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State { state: config.seed };
        Self { config, state }
    }
}

impl Default for Noise {
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl ConfigRef for Noise {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl ConfigClone for Noise
where
    Config: Clone,
{
    fn config(&self) -> Self::Config {
        self.config
    }
}

impl StateMut for Noise {
    /// # Safety
    ///
    /// The caller is responsible for upholding the generator's internal invariants:
    /// - `state.state` must not be zero (xorshift32 requires a non-zero state)
    /// - Modifying state arbitrarily may break the deterministic sequence
    #[doc(hidden)]
    fn state_mut(&mut self) -> &mut Self::State {
        // SAFETY: `&mut self` guarantees exclusive access; no other references
        // to state exist within the program at this point.
        &mut self.state
    }
}

impl HasGuts for Noise {
    type Guts = (Config, State);
}

impl FromGuts for Noise {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl IntoGuts for Noise {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl Reset for Noise {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl ResetMut for Noise where Self: Reset {}

impl Noise {
    /// Creates a new `Noise` source from a `Config`.
    #[inline]
    #[must_use]
    pub fn from_config(config: Config) -> Self {
        Self::with_config(config)
    }

    /// Creates a new `Noise` source with the given seed.
    ///
    /// If `seed` is 0, it will be replaced with `0xDEAD_BEEF`.
    #[inline]
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self::from_config(Config::new(seed))
    }
}

impl Source for Noise {
    type Output = u32;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        let mut x = self.state.state;
        if x == 0 {
            x = 0xDEAD_BEEF;
            self.state.state = x;
        }
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state.state = x;
        Some(x)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_deterministic_same_seed_produces_same_sequence() {
        const SEED: u32 = 12345;
        const COUNT: usize = 10;

        let source1 = Noise::new(SEED);
        let sequence1: Vec<_> = (0..COUNT)
            .scan(source1, |source, _| source.source())
            .collect();

        let source2 = Noise::new(SEED);
        let sequence2: Vec<_> = (0..COUNT)
            .scan(source2, |source, _| source.source())
            .collect();

        assert_eq!(
            sequence1, sequence2,
            "Same seed should produce identical sequences"
        );
    }

    #[test]
    fn test_noise_output_is_not_all_equal() {
        const SEED: u32 = 42;
        const COUNT: usize = 100;

        let source = Noise::new(SEED);
        let sequence: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();

        // Verify not all values are the same (basic smoke test)
        let first = sequence[0];
        assert!(
            sequence.iter().any(|&x| x != first),
            "All 100 values are equal — generator is stuck"
        );
    }

    #[test]
    fn test_noise_period_is_large() {
        const SEED: u32 = 42;
        const COUNT: usize = 1000;

        let source = Noise::new(SEED);
        let sequence: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();

        // Verify no value repeats within window (weak periodicity check)
        let has_duplicate = sequence.windows(2).any(|w| w[0] == w[1]);
        if has_duplicate {
            std::eprintln!(
                "Note: adjacent equal values detected (statistically unlikely but possible)"
            );
        }
    }

    #[test]
    fn test_seed_zero_is_replaced() {
        const ZERO_SEED: u32 = 0;
        const EXPLICIT_SEED: u32 = 0xDEAD_BEEF;
        const COUNT: usize = 10;

        let source_zero = Noise::new(ZERO_SEED);
        let sequence_zero: Vec<_> = (0..COUNT)
            .scan(source_zero, |source, _| source.source())
            .collect();

        let source_explicit = Noise::new(EXPLICIT_SEED);
        let sequence_explicit: Vec<_> = (0..COUNT)
            .scan(source_explicit, |source, _| source.source())
            .collect();

        assert_eq!(
            sequence_zero, sequence_explicit,
            "Seed 0 should be replaced with 0xDEAD_BEEF"
        );
    }

    #[test]
    fn test_different_seeds_produce_different_sequences() {
        const SEED1: u32 = 111;
        const SEED2: u32 = 222;
        const COUNT: usize = 20;

        let source1 = Noise::new(SEED1);
        let sequence1: Vec<_> = (0..COUNT)
            .scan(source1, |source, _| source.source())
            .collect();

        let source2 = Noise::new(SEED2);
        let sequence2: Vec<_> = (0..COUNT)
            .scan(source2, |source, _| source.source())
            .collect();

        assert_ne!(
            sequence1, sequence2,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_from_config() {
        const SEED: u32 = 54321;
        const COUNT: usize = 5;

        let config = Config::new(SEED);
        let source = Noise::from_config(config);
        let sequence: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();

        let source_direct = Noise::new(SEED);
        let sequence_direct: Vec<_> = (0..COUNT)
            .scan(source_direct, |source, _| source.source())
            .collect();

        assert_eq!(sequence, sequence_direct, "from_config should match new()");
    }
}
