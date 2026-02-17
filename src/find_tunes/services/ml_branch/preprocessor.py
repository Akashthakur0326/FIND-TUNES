import numpy as np
import torch
import torchaudio
import torchcrepe
import soundfile as sf
import scipy.signal as sg
from loguru import logger

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, win_sec=3.0, hop_sec=1.5):
        self.sample_rate = sample_rate
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        
        self.samples_win = int(sample_rate * win_sec) #16000 * 3 = 48000
        self.samples_hop = int(sample_rate * hop_sec) #16000 * 1.5 = 24000
        
        # Spectrogram Transforms (CPU)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    #This creates: STFT (n_fft=1024) , hop=512 samples (~32ms) ,64 mel bins
    def load_audio(self, path: str):
        try:
            # 1. Bypass torchaudio completely and read natively with soundfile
            wav_np, sr = sf.read(path, dtype='float32')
            
            # 2. soundfile returns (time, channels), PyTorch expects (channels, time)
            if wav_np.ndim == 1:
                wav_np = np.expand_dims(wav_np, axis=1) # Make it 2D
            
            wav = torch.from_numpy(wav_np).T # Transpose to (channels, time)
            
            # 3. Standard normalization
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            return wav
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def extract_pitch_track(self, wav: torch.Tensor):
        """Runs CREPE over the audio in 30-second chunks to prevent OOM."""
        try:
            # 1. Define Chunk Size (30 seconds is safe for CPU RAM)
            chunk_seconds = 30
            chunk_samples = chunk_seconds * self.sample_rate
            total_samples = wav.shape[1]
            
            all_pitches = []
            
            # 2. Iterate through the audio in chunks
            logger.info(f"    Running Pitch Detection on {total_samples/self.sample_rate:.1f}s of audio...")
            for i in range(0, total_samples, chunk_samples):
                # Slice the tensor (no memory copy, just a view)
                chunk = wav[:, i : i + chunk_samples]
                
                # Pad the last chunk if it's too tiny (CREPE dislikes tiny inputs)
                if chunk.shape[1] < 1600: # Less than 0.1s
                     chunk = torch.nn.functional.pad(chunk, (0, 1600 - chunk.shape[1]))

                # Run CREPE on this small slice
                pitch_chunk = torchcrepe.predict(
                    chunk, 
                    self.sample_rate, 
                    hop_length=160, 
                    fmin=50, 
                    fmax=2000, 
                    model='tiny', 
                    decoder=torchcrepe.decode.argmax, 
                    device='cpu',
                    batch_size=256 # Internal batching for speed
                )
                all_pitches.append(pitch_chunk)
            
            # 3. Stitch the chunks back together
            pitch = torch.cat(all_pitches, dim=1)
            
            # 4. Standard Smoothing & Log Scaling
            pitch_np = pitch.squeeze().numpy()
            pitch_log = np.log1p(pitch_np)
            pitch_smooth = sg.medfilt(pitch_log, kernel_size=5).astype(np.float32)
            
            return pitch_smooth

        except Exception as e:
            logger.error(f"Pitch extraction failed: {e}")
            return None

    def process_into_windows(self, path: str):
        """
        Returns aligned lists of:
        times, spectrogram_arrays, pitch_arrays
        """
        wav = self.load_audio(path)
        if wav is None: return [], [], []
        
        # 1. Pad audio if it's too short
        if wav.shape[1] < self.samples_win:
            wav = torch.nn.functional.pad(wav, (0, self.samples_win - wav.shape[1]))
            
        # 2. Get global pitch track
        global_pitch = self.extract_pitch_track(wav) #compute pitch ONCE for whole audio then slice it per window
        if global_pitch is None: return [], [], []

        times, spec_batches, pitch_batches = [], [], []
        
        # Pitch frames mapping: 1.5s hop = 150 frames, 3.0s win = 300 frames
        pitch_win = 300
        pitch_hop = 150
        
        # 3. Slide the window
        pitch_idx = 0
        for i in range(0, wav.shape[1] - self.samples_win + 1, self.samples_hop):
            # --- Spectrogram Window ---
            audio_win = wav[:, i:i + self.samples_win]
            spec = self.db_transform(self.mel_transform(audio_win))
            # Normalize exactly like eval script:
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
            # Shape for ONNX: (Batch=1, Channels=1, Mels=64, Time=94)
            spec_np = spec.unsqueeze(0).numpy().astype(np.float32)
            
            # --- Pitch Window ---
            # Ensure we don't index out of bounds on the pitch array
            if pitch_idx + pitch_win <= len(global_pitch):
                p_win = global_pitch[pitch_idx : pitch_idx + pitch_win]
            else:
                # Pad if we hit the very end
                p_win = np.pad(global_pitch[pitch_idx:], (0, pitch_win - len(global_pitch[pitch_idx:])), 'constant')
            
            # Shape for ONNX: (Batch=1, Channels=1, Seq=300)
            p_np = p_win.reshape(1, 1, 300).astype(np.float32)

            # --- Append ---
            times.append(i / self.sample_rate)
            spec_batches.append(spec_np)
            pitch_batches.append(p_np)
            
            pitch_idx += pitch_hop

        return times, spec_batches, pitch_batches