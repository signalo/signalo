// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A complex NCO tone mixed to baseband and low-pass decimated by four.
//!
//! Run with `cargo run --example ddc --features complex`.
//!
//! The tone sits at `+TONE_FREQUENCY`, so mixing with an NCO at
//! `-TONE_FREQUENCY` translates it to DC. The low-pass prototype is designed at
//! the input rate and keeps the passband, so the decimated samples settle at
//! `1 + 0j`: unit amplitude, translated to DC.

use signalo::complex::Complex32;
use signalo::filters::fir::design::windowed_sinc;
use signalo::filters::fir::polyphase::decimator::PolyphaseDecimatorVec;
use signalo::filters::mixer::Mixer;
use signalo::sources::oscillator::nco::Nco;
use signalo::traits::{Filter, MultirateFilter};

/// Input sample rate of the tone.
const SAMPLE_RATE: f32 = 48_000.0;

/// Frequency of the complex tone being received, in Hz.
const TONE_FREQUENCY: f32 = 2_400.0;

/// Number of input samples retained per decimated output sample.
const DECIMATION: usize = 4;

/// Number of input samples fed through the chain.
const INPUT_SAMPLES: usize = 4_096;

/// Number of leading outputs discarded before measuring the steady state.
const TRANSIENT_OUTPUTS: usize = 16;

fn main() {
    let prototype = windowed_sinc::hann::lowpass_vec::<f32>(65, 0.1);
    let mut tone_nco = Nco::<f32>::from_frequency(TONE_FREQUENCY, SAMPLE_RATE);
    let mut mixer = Mixer::new(Nco::<f32>::from_frequency(-TONE_FREQUENCY, SAMPLE_RATE));
    let mut decimator =
        PolyphaseDecimatorVec::<Complex32, f32>::from_prototype_taps(DECIMATION, &prototype);
    let mut samples = Vec::with_capacity(INPUT_SAMPLES / DECIMATION);

    for _ in 0..INPUT_SAMPLES {
        let tone = tone_nco.phasor_then_step();
        let baseband = mixer.filter(tone);
        let mut output = [Complex32::default(); 1];
        let (_, produced) = decimator.process(&[baseband], &mut output);

        if produced == 1 {
            samples.push(output[0]);
        }
    }

    // Both NCOs start at phase zero and emit their phasor before stepping, so their
    // opposite frequencies cancel without a phase offset. Comparing against
    // `1 + 0j` rather than only the magnitude also catches a tone that was
    // never translated, since a 2.4 kHz tone still sits inside the passband.
    let max_error = samples[TRANSIENT_OUTPUTS..]
        .iter()
        .map(|sample| (*sample - Complex32::new(1.0, 0.0)).norm())
        .fold(0.0_f32, f32::max);

    println!(
        "samples: {}, decimation: {DECIMATION}, max_error: {max_error}",
        samples.len(),
    );

    assert!(
        samples.len() == INPUT_SAMPLES / DECIMATION,
        "the decimator emits exactly one output per decimation input samples"
    );
    assert!(
        max_error < 1.0e-3,
        "the downconverted tone settles near 1 + 0j"
    );
}
