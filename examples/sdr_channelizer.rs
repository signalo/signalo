// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo::complex::Complex;
use signalo::filters::iir::analog::{butterworth_lowpass, lowpass_biquad_configs};
use signalo::filters::iir::biquad::cascade::{self, BiquadCascadeArray};
use signalo::filters::mixer::Mixer;
use signalo::sources::noise::Noise;
use signalo::sources::oscillator::nco::Nco;
use signalo::traits::{Filter, Source, WithConfig};

/// Scales a `Noise` source's raw `u32` output to `[-half_width, +half_width]`.
fn noise_to_range(raw: u32, half_width: f64) -> f64 {
    (f64::from(raw) / f64::from(u32::MAX) - 0.5) * 2.0 * half_width
}

fn main() {
    let fs = 200_000.0_f64;
    let n_samples = 4000_usize;
    let warm_up = 1000_usize;

    let mut signal_nco = Nco::<f64>::from_frequency(15_000.0, fs as f32);
    let mut interferer_nco = Nco::<f64>::from_frequency(-60_000.0, fs as f32);
    let mut mixer = Mixer::new(Nco::<f64>::from_frequency(-15_000.0, fs as f32));

    let mut noise_i = Noise::new(101);
    let mut noise_q = Noise::new(202);

    let cfgs = lowpass_biquad_configs::<f64, 2>(butterworth_lowpass::<f64>(4), fs, 5_000.0);
    let cascade_config = cascade::Config::new(cfgs);
    let mut channel_filter: BiquadCascadeArray<Complex<f64>, 2, f64> =
        BiquadCascadeArray::with_config(cascade_config);

    let mut sum_sq_before = 0.0_f64;
    let mut sum_sq_after = 0.0_f64;
    let mut last_filtered = Complex::new(0.0_f64, 0.0_f64);

    for i in 0..n_samples {
        let signal_phasor = signal_nco.phasor_then_step();
        let interferer_phasor = interferer_nco.phasor_then_step() * 2.0;

        let noise_sample = Complex::new(
            noise_to_range(noise_i.source().expect("failed to source noise i"), 0.05),
            noise_to_range(noise_q.source().expect("failed to source noise q"), 0.05),
        );

        let capture = signal_phasor + interferer_phasor + noise_sample;
        let downconverted = mixer.filter(capture);
        let filtered = channel_filter.filter(downconverted);

        if i % 500 == 0 {
            println!(
                "Sample {:4}: downconverted = ({:.4}, {:.4}), filtered = ({:.4}, {:.4})",
                i, downconverted.re, downconverted.im, filtered.re, filtered.im
            );
        }

        if i >= warm_up {
            let ideal_tone = Complex::new(1.0_f64, 0.0_f64);
            sum_sq_before += (downconverted - ideal_tone).norm_sqr();
            sum_sq_after += (filtered - ideal_tone).norm_sqr();
            last_filtered = filtered;
        }
    }

    let count = n_samples - warm_up;
    let rms_before = (sum_sq_before / count as f64).sqrt();
    let rms_after = (sum_sq_after / count as f64).sqrt();
    let attenuation_db = 20.0 * (rms_before / rms_after).log10();
    let recovered_magnitude = last_filtered.norm();

    println!("Recovered magnitude: {:.4} (1.0000)", recovered_magnitude);
    println!("Interference+noise attenuation: {:.4} dB", attenuation_db);

    assert!(
        attenuation_db > 0.0,
        "channel filter must reduce residual energy versus the raw downconverted capture"
    );
    assert!(
        (recovered_magnitude - 1.0).abs() < 0.1,
        "recovered tone magnitude must stay close to the ideal 1.0"
    );
}
