// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A [`Cycle`] source repeating a synthetic ECG through a [`Peaks`] detector and
//! an amplitude level, so that only the R wave counts as a beat.
//!
//! Run with `cargo run --example heartbeat --features alloc`.
//!
//! Every wave of the waveform is a local maximum, so [`Peaks`] alone reports the P
//! wave, the QRS onset and the T wave onset. Comparing the crest of each peak
//! against [`R_LEVEL`] discards the P and T waves and leaves the R wave of each
//! cycle.
//!
//! [`Cycle`]: signalo::sources::cycle::Cycle
//! [`Peaks`]: signalo::filters::classify::peaks::Peaks
//! [`R_LEVEL`]: self::R_LEVEL

use signalo::filters::classify::peaks::{self, Peak, Peaks};
use signalo::filters::classify::Classification;
use signalo::sources::cycle::Cycle;
use signalo::sources::from_iter::FromIter;
use signalo::traits::{Filter, Source, WithConfig};

/// One beat of the synthetic ECG at four samples per wave.
///
/// The P wave peaks at index 3 (amplitude `0.2`), the QRS complex at index 11
/// (amplitude `1.0`) and the T wave at indices 16 through 19 (amplitude `0.4`);
/// the values between them are the isoelectric baseline. All three waves are
/// local maxima, so a detector that reacts to any local maximum reports the P
/// wave, the QRS onset and the T wave onset, and not only the R wave.
const BEAT: [f32; 24] = [
    0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 1.0, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4,
    0.3, 0.2, 0.1, 0.0, 0.0,
];

/// Number of beat cycles fed through the detector.
const CYCLES: usize = 8;

/// Level separating the R wave from the P and T waves.
///
/// It sits above the largest P and T amplitude (`0.4`) and below the QRS crest
/// (`1.0`), so comparing the crest of each detected peak against it accepts the
/// QRS only.
const R_LEVEL: f32 = 0.5;

fn main() {
    let beat_samples = BEAT.len() * CYCLES;

    let mut source = Cycle::new(FromIter::from(BEAT));
    let mut peaks = Peaks::<f32, Peak>::with_config(peaks::Config {
        outputs: Peak::classes(),
    });
    let mut beats = 0;
    let mut indices = Vec::new();
    let mut previous_sample = BEAT[0];

    for sample_index in 0..beat_samples {
        let sample = source.source().expect("cycle source is infinite");
        let detected = peaks.filter(sample);
        let crest = previous_sample;
        previous_sample = sample;

        if detected != Peak::Max || crest <= R_LEVEL {
            continue;
        }

        beats += 1;
        indices.push(sample_index - 1);
    }

    println!("beats: peaks at {indices:?}, total {beats}");

    // Every wave is a detected peak; only the QRS crest clears the level, so each
    // cycle contributes exactly one beat.
    assert_eq!(beats, CYCLES);
}
