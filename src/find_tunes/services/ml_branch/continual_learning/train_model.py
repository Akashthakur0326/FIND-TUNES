import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from loguru import logger

# Project imports
from find_tunes.core.config import BASE_DIR
from find_tunes.services.ml_branch.preprocessor import AudioPreprocessor
from find_tunes.services.ml_branch.onnx_runner import ml_engine

# Import your architectures and datasets (assuming you saved them in these files)
# If these don't exist yet, you should place your PyTorch class definitions in a 'models.py' and 'datasets.py'
from find_tunes.services.ml_branch.model import AudioSiameseNet, CRNN
from find_tunes.services.ml_branch.dataset import DualObjectiveSiameseDataset, PitchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_pitch_data(data_dir: str):
    """Converts the downloaded .wav files into .npy pitch tracks for the CRNN."""
    logger.info("🎵 Pre-extracting CREPE pitch tracks for the Replay Buffer...")
    preprocessor = AudioPreprocessor()
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    
    pitch_dir = os.path.join(data_dir, "pitch_npy")
    os.makedirs(pitch_dir, exist_ok=True)
    
    for wav_path in wav_files:
        filename = os.path.basename(wav_path).replace(".wav", ".npy")
        out_path = os.path.join(pitch_dir, filename)
        
        if os.path.exists(out_path):
            continue
            
        wav_tensor = preprocessor.load_audio(wav_path)
        if wav_tensor is not None:
            pitch_track = preprocessor.extract_pitch_track(wav_tensor)
            if pitch_track is not None:
                np.save(out_path, pitch_track)
                
    return pitch_dir

def finetune_spec_model(data_dir: str):
    logger.info(f"\n🔥 Fine-Tuning Spectrogram CNN on: {DEVICE}")
    
    # 1. Setup Data (100% Self-Invariance since we have no cover pairs for new YouTube songs)
    wav_files = [os.path.basename(f) for f in glob.glob(os.path.join(data_dir, "*.wav"))]
    
    dataset = DualObjectiveSiameseDataset(
        anchor_list=wav_files,
        pair_map={}, # Empty, forces self-invariance task
        originals_dir=data_dir,
        covers_dir=data_dir, 
        noise_dir=str(BASE_DIR / "data" / "noise"),
        sample_rate=16000,
        duration=3.0,
        aligned_cover_prob=0.0 # Force 0% cover task
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 2. Load the Champion Weights (We pull the .pth we used to make the ONNX)
    model = AudioSiameseNet().to(DEVICE)
    champion_path = str(BASE_DIR / "models" / "champion_spec.pth") # You must ensure this exists locally
    if os.path.exists(champion_path):
        model.load_state_dict(torch.load(champion_path, map_location=DEVICE))
        logger.info("✅ Loaded Champion Spectrogram Weights.")
    
    # 3. Optimizer & Triplet Loss
    # 🧠 WHY: Very low learning rate (1e-5) to prevent destroying the champion weights
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.TripletMarginLoss(margin=0.75)
    scaler = GradScaler()
    
    # 4. Short Training Loop (e.g., 10 epochs for continual learning)
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast():
                emb_a, emb_p, emb_n = model(anc, pos, neg)
                loss = criterion(emb_a, emb_p, emb_n)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        logger.info(f" 📢 Spec Epoch {epoch+1}/10 | Loss: {running_loss/len(dataloader):.4f}")

    # 5. Save the candidate model
    candidate_path = str(BASE_DIR / "models" / "candidate_spec.pth")
    torch.save(model.state_dict(), candidate_path)
    logger.success(f"💾 Saved candidate Spectrogram model to {candidate_path}")

def finetune_pitch_model(data_dir: str):
    logger.info(f"\n🔥 Fine-Tuning Pitch CRNN on: {DEVICE}")
    
    # 1. Preprocess WAV to NPY
    pitch_dir = preprocess_pitch_data(data_dir)
    
    # 2. Dataset & Loader
    dataset = PitchDataset(pitch_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Load Champion Weights
    model = CRNN().to(DEVICE)
    champion_path = str(BASE_DIR / "models" / "champion_pitch.pth")
    if os.path.exists(champion_path):
        model.load_state_dict(torch.load(champion_path, map_location=DEVICE))
        logger.info("✅ Loaded Champion Pitch Weights.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.TripletMarginLoss(margin=0.85, p=2)
    
    # 4. Short Training Loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for a, p, n in dataloader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model.forward_one(a), model.forward_one(p), model.forward_one(n))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f" 📢 Pitch Epoch {epoch+1}/10 | Loss: {total_loss/len(dataloader):.4f}")

    # 5. Save candidate model
    candidate_path = str(BASE_DIR / "models" / "candidate_pitch.pth")
    torch.save(model.state_dict(), candidate_path)
    logger.success(f"💾 Saved candidate Pitch model to {candidate_path}")