import io
import numpy as np
from pydub import AudioSegment
from loguru import logger
from find_tunes.core.config import SAMPLE_RATE, CHANNELS

def _decode_and_normalize(audio_bytes: bytes) -> AudioSegment:
    """
    Internal helper to safely decode incoming browser bytes into a 16kHz Mono AudioSegment.
    It catches the byte stream in memory (io.BytesIO) to avoid slow disk I/O.
    """
    try:
        # Pydub can auto-detect the format from the byte header (webm, ogg, mp4, etc.)
        byte_stream = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(byte_stream)
        
        # Force strict compliance for our ML and DSP models
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
        return audio
    except Exception as e:
        logger.error(f"❌ Failed to decode incoming audio bytes: {e}")
        # Return empty audio segment as fallback
        return AudioSegment.empty()

def convert_webm_to_wav_array(audio_bytes: bytes) -> np.ndarray:
    """
    Converts raw browser bytes directly into a NumPy array in RAM.
    Used by the Shazam (DSP) route for instantaneous, disk-free streaming.
    """
    audio = _decode_and_normalize(audio_bytes)
    
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    # Extract raw samples as a NumPy array
    samples = np.array(audio.get_array_of_samples())
    
    # DSP and ML models prefer normalized float32 arrays (-1.0 to 1.0)
    # AudioSegment usually returns int16 (-32768 to 32767)
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)
        
    return samples

def save_bytes_to_wav(audio_bytes: bytes, output_path: str) -> bool:
    """
    Converts raw browser bytes and writes them to a clean .wav file on disk.
    Used by the ML Fallback route because torchaudio and CREPE expect a physical file.
    """
    audio = _decode_and_normalize(audio_bytes)
    
    if len(audio) == 0:
        return False
        
    try:
        # Export as strict uncompressed PCM WAV
        audio.export(output_path, format="wav")
        logger.info(f"💾 Saved fallback audio to {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save WAV file to disk: {e}")
        return False