import os
import glob
import random
import numpy as np
import scipy.signal as sg
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

# 🌟 THE FIX: Swapped AddBackgroundNoise for AddColorNoise and AddGaussianNoise
from audiomentations import Compose, PitchShift, TimeStretch, Gain, AddColorNoise, AddGaussianNoise

# ==========================================
# SPECTROGRAM DATASET (Self-Supervised)
# ==========================================
class SelfSupervisedSiameseDataset(Dataset):
    # 🌟 THE FIX: Removed noise_dir from the parameters
    def __init__(self, data_dir, sample_rate=16000, duration=3.0):
        
        self.files = glob.glob(os.path.join(data_dir, "*.wav"))
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.num_songs = len(self.files)

        # 🌟 THE FIX: 100% Algorithmic Augmentations (Zero external files needed)
        self.augment = Compose([
            Gain(min_gain_db=-10.0, max_gain_db=10.0, p=0.5),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.8), 
            TimeStretch(min_rate=0.85, max_rate=1.15, leave_length_unchanged=True, p=0.8), 
            
            # Synthetically generates Pink, Brown, or White noise on the fly
            AddColorNoise(min_snr_in_db=5.0, max_snr_in_db=20.0, p=0.7),
            # Adds digital static/hiss
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
        ])

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def _load_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav
        except Exception:
            return None

    def _crop(self, waveform, start_sample=None):
        total_len = waveform.shape[1]
        if total_len < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - total_len))
            return waveform[:, :self.num_samples]
            
        if start_sample is None:
            start_sample = random.randint(0, total_len - self.num_samples)
            
        start_sample = max(0, min(start_sample, total_len - self.num_samples))
        return waveform[:, start_sample:start_sample + self.num_samples]

    def _to_spec(self, audio_np):
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        spec = self.mel_transform(tensor)
        spec = self.db_transform(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec

    def __len__(self):
        return self.num_songs * 3 

    def __getitem__(self, idx):
        real_idx = idx % self.num_songs
        anchor_path = self.files[real_idx]
        anchor_wav = self._load_audio(anchor_path)
        
        if anchor_wav is None:
            anchor_wav = torch.zeros(1, self.num_samples)

        anchor_crop = self._crop(anchor_wav)
        anchor_raw = anchor_crop.squeeze(0).numpy()

        if random.random() > 0.5:
            positive_raw = anchor_raw.copy()
        else:
            pos_crop = self._crop(anchor_wav) 
            positive_raw = pos_crop.squeeze(0).numpy()

        try:
            positive_aug = self.augment(samples=positive_raw, sample_rate=self.sample_rate)
        except Exception:
            positive_aug = positive_raw

        neg_idx = random.randint(0, self.num_songs - 1)
        while neg_idx == real_idx:
            neg_idx = random.randint(0, self.num_songs - 1)

        neg_path = self.files[neg_idx]
        neg_wav = self._load_audio(neg_path)

        if neg_wav is None:
            negative_raw = anchor_raw.copy()
        else:
            neg_crop = self._crop(neg_wav)
            negative_raw = neg_crop.squeeze(0).numpy()

        try:
            negative_aug = self.augment(samples=negative_raw, sample_rate=self.sample_rate)
        except Exception:
            negative_aug = negative_raw

        return self._to_spec(anchor_raw), self._to_spec(positive_aug), self._to_spec(negative_aug)


# ==========================================
# PITCH DATASET (Self-Supervised)
# ==========================================
class PitchDataset(Dataset):
    def __init__(self, pitch_dir):
        self.files = sorted(glob.glob(os.path.join(pitch_dir, "*.npy")))

    def __len__(self): 
        # Train on each pitch track multiple times
        return len(self.files) * 3

    def __getitem__(self, idx):
        real_idx = idx % len(self.files)
        anchor = np.load(self.files[real_idx])
        
        neg_idx = random.randint(0, len(self.files)-1)
        while neg_idx == real_idx: 
            neg_idx = random.randint(0, len(self.files)-1)
        neg = np.load(self.files[neg_idx])

        # Smooth tracks
        anchor = sg.medfilt(anchor, 5).astype(np.float32)
        neg = sg.medfilt(neg, 5).astype(np.float32)

        def get_crop(arr, specific_start=None):
            if len(arr) < 300: 
                return np.pad(arr, (0, 300-len(arr)), 'constant'), 0
            if specific_start is None:
                start = random.randint(0, len(arr)-300)
            else:
                start = specific_start
                start = max(0, min(start, len(arr)-300))
            return arr[start:start+300], start

        a_crop, a_start = get_crop(anchor)
        n_crop, _ = get_crop(neg)

        # 2. POSITIVE (Same song)
        # 50% chance: Same crop + Pitch Noise
        # 50% chance: Different crop entirely
        if random.random() > 0.5:
            p_crop = a_crop.copy()
            # Add Gaussian noise to simulate slight pitch detection errors
            p_crop += np.random.normal(0, 0.08, 300)
        else:
            p_crop, _ = get_crop(anchor) # Random new crop

        return (
            torch.from_numpy(a_crop).float().unsqueeze(0),
            torch.from_numpy(p_crop).float().unsqueeze(0),
            torch.from_numpy(n_crop).float().unsqueeze(0)
        )