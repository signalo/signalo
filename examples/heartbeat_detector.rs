// MARK: Heartbeat Detector Example

use signalo::filters::classify::peaks::{Config as PeaksConfig, Peak, Peaks};
use signalo::filters::classify::Classification;
use signalo::filters::fir::mean::MeanArray;
use signalo::filters::rank::median::MedianArray;
use signalo::pipes::pipe::Pipe;
use signalo::sources::noise::Noise;
use signalo::sources::oscillator::sine::{Config as SineConfig, SineOscillator};
use signalo::traits::{Filter, Source, WithConfig};

/// Scales a `Noise` source's raw `u32` output to `[-half_width, +half_width]`.
fn noise_to_range(raw: u32, half_width: f64) -> f64 {
    (f64::from(raw) / f64::from(u32::MAX) - 0.5) * 2.0 * half_width
}

fn main() {
    let sample_rate = 100.0_f64;
    let duration_secs = 10.0_f64;
    let n_samples = 1000usize;
    let heart_rate_bpm = 72.0_f64;
    let ppg_freq_hz = heart_rate_bpm / 60.0;

    let mut ppg = SineOscillator::<f64>::with_config(SineConfig::new(ppg_freq_hz, sample_rate));
    let mut noise = Noise::new(7);

    let median: MedianArray<f64, 5> = MedianArray::default();
    let mean: MeanArray<f64, 3> = MeanArray::default();
    let peaks: Peaks<f64, Peak> = Peaks::with_config(PeaksConfig {
        outputs: Peak::classes(),
    });

    let mut pipeline = Pipe::new(median, mean) | peaks;

    let mut display_pipeline = Pipe::new(
        MedianArray::<f64, 5>::default(),
        MeanArray::<f64, 3>::default(),
    );

    let mut peak_count = 0;

    for i in 0..n_samples {
        let ppg_val = ppg.source().expect("PPG source failed");
        let noise_val = noise.source().expect("Noise source failed");

        let spike_i = if i % 250 == 125 { 2.5 } else { 0.0 };

        let raw_value = ppg_val + spike_i + noise_to_range(noise_val, 0.03);

        let classification = pipeline.filter(raw_value);
        let _filtered_for_display = display_pipeline.filter(raw_value);

        if classification == Peak::Max {
            peak_count += 1;
        }

        if i % 100 == 0 {
            println!(
                "i = {}, raw = {}, max = {}",
                i,
                raw_value,
                classification == Peak::Max
            );
        }
    }

    let estimated_bpm = peak_count as f64 / duration_secs * 60.0;

    println!("Detected beats: {}", peak_count);
    println!("Estimated BPM: {}", estimated_bpm);
    println!("Ground-truth BPM: 72.00");

    assert!(
        (estimated_bpm - 72.0).abs() / 72.0 < 0.20,
        "estimated BPM must be within 20% of ground truth"
    );
}
