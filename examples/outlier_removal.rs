// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A Hampel filter over a finite spike-carrying source, summed by an integrating
//! sink one sample at a time.
//!
//! Run with `cargo run --example outlier_removal --features alloc`.

use signalo::filters::rank::hampel::{self, HampelArray};
use signalo::sinks::integrate::Integrate;
use signalo::sources::from_iter::FromIter;
use signalo::traits::{Filter, Finalize, Sink, Source, WithConfig};

/// Length of the input stream.
const LENGTH: usize = 32;

/// Index of the first spike.
const FIRST_SPIKE: usize = 8;

/// Index of the second spike.
const SECOND_SPIKE: usize = 20;

/// Amplitude of both spikes.
const SPIKE: f32 = 10.0;

/// Largest accepted residual of a cleaned spike.
const TOLERANCE: f32 = 1.0e-3;

fn main() {
    let mut input = [0.0_f32; LENGTH];
    input[FIRST_SPIKE] = SPIKE;
    input[SECOND_SPIKE] = SPIKE;

    let mut source = FromIter::from(input);
    let mut hampel = HampelArray::<f32, 5>::with_config(hampel::Config { threshold: 3.0 });
    let mut sink = Integrate::<f32>::default();

    // The stream is a zero baseline apart from the two spikes, so the sum of the
    // cleaned samples equals the total residual the filter leaves behind.
    let mut cleaned_sum = 0.0_f32;
    while let Some(sample) = source.source() {
        let cleaned = hampel.filter(sample);
        sink.sink(cleaned);
        cleaned_sum += cleaned;
    }

    let residual = sink.finalize().unwrap_or_default();
    println!(
        "cleaned: {LENGTH} samples, spikes at {FIRST_SPIKE} and {SECOND_SPIKE} \
         (amplitude {SPIKE}) -> residual {residual:.6}"
    );

    // The Hampel filter replaces each isolated spike with its local median, a zero
    // run here, so nothing of either spike reaches the sink.
    assert_eq!(cleaned_sum, residual);
    assert!(residual.abs() < TOLERANCE);
}
