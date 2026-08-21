// audio/transcription/wav.rs
//
// Minimal WAV encoding shared by the online transcription providers.
//
// Cloud STT endpoints (Sarvam, OpenAI, and OpenAI-compatible services) accept
// audio files rather than raw sample buffers, so the f32 PCM the VAD hands us
// has to be wrapped in a RIFF container before upload. Manual construction
// avoids pulling in an encoder crate (the project deliberately dropped `hound`).

/// Encodes 16kHz mono f32 samples (range roughly [-1.0, 1.0]) as a 16-bit PCM
/// WAV byte buffer.
pub fn encode_wav_16k_mono(samples: &[f32]) -> Vec<u8> {
    const SAMPLE_RATE: u32 = 16_000;
    const CHANNELS: u16 = 1;
    const BITS_PER_SAMPLE: u16 = 16;

    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);
    let data_len = (samples.len() * 2) as u32;
    let riff_len = 36 + data_len;

    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_len.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&CHANNELS.to_le_bytes());
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * i16::MAX as f32) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32_le(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn read_u16_le(b: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([b[off], b[off + 1]])
    }

    #[test]
    fn wav_header_is_well_formed_16k_mono_pcm16() {
        let samples = vec![0.0f32; 8];
        let wav = encode_wav_16k_mono(&samples);

        // RIFF/WAVE container
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // fmt chunk size = 16, PCM format = 1, mono = 1
        assert_eq!(read_u32_le(&wav, 16), 16);
        assert_eq!(read_u16_le(&wav, 20), 1);
        assert_eq!(read_u16_le(&wav, 22), 1);
        // sample rate 16k, bits 16
        assert_eq!(read_u32_le(&wav, 24), 16_000);
        assert_eq!(read_u16_le(&wav, 34), 16);
        // byte_rate = 16000 * 1 * 2, block_align = 2
        assert_eq!(read_u32_le(&wav, 28), 32_000);
        assert_eq!(read_u16_le(&wav, 32), 2);
        // data chunk
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(read_u32_le(&wav, 40), (samples.len() * 2) as u32);
        // total length = 44 header + data
        assert_eq!(wav.len(), 44 + samples.len() * 2);
        // riff length = 36 + data
        assert_eq!(read_u32_le(&wav, 4), 36 + (samples.len() * 2) as u32);
    }

    #[test]
    fn wav_encodes_samples_and_clamps_out_of_range() {
        // 1.0 -> i16::MAX, -1.0 -> -i16::MAX (symmetric scaling), 2.0 clamps to max
        let samples = vec![1.0f32, -1.0, 2.0, -2.0, 0.0];
        let wav = encode_wav_16k_mono(&samples);
        let data = &wav[44..];
        let s0 = i16::from_le_bytes([data[0], data[1]]);
        let s1 = i16::from_le_bytes([data[2], data[3]]);
        let s2 = i16::from_le_bytes([data[4], data[5]]);
        let s3 = i16::from_le_bytes([data[6], data[7]]);
        let s4 = i16::from_le_bytes([data[8], data[9]]);
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, -i16::MAX);
        assert_eq!(s2, i16::MAX); // clamped
        assert_eq!(s3, -i16::MAX); // clamped
        assert_eq!(s4, 0);
    }

    #[test]
    fn empty_input_produces_header_only() {
        let wav = encode_wav_16k_mono(&[]);
        assert_eq!(wav.len(), 44);
        assert_eq!(read_u32_le(&wav, 40), 0);
    }
}
