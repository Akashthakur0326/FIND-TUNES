import os
import csv
import random
import time
import uuid
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch, Gain

from find_tunes.core.config import ensure_directories, TEMP_AUDIO_DIR, CSV_BACKUP_PATH, NOISE_DIR, SAMPLE_RATE
from find_tunes.core.database import SessionLocal, SpectrogramEmbedding, PitchEmbedding
from find_tunes.services.ingestion_cycle.audio_engine import AudioEngine
from find_tunes.services.shazam_branch.matcher import HashMatcher, FingerprintCache
from find_tunes.services.ml_branch.onnx_runner import ml_engine
from find_tunes.services.ml_branch.matcher import MLMatcher
from find_tunes.services.ml_branch.fusion import ensemble_fusion

# 🌟 TEST CONFIGURATION (Bump this to 10 or 20 when you are ready for real stats!)
NUM_SONGS_TO_TEST = 269  
CLIPS_PER_SONG = 3    
QUERY_LEN_SEC = 15.0
TOLERANCE_SEC = 5.0    

def apply_augmentation(wav_np, mode):
    augmenters = {
        "clean": None,
        "soft": Compose([
            Gain(min_gain_db=-5.0, max_gain_db=5.0, p=0.8),
            AddBackgroundNoise(sounds_path=str(NOISE_DIR), min_snr_db=15.0, max_snr_db=20.0, p=0.8)
        ]),
        "hard": Compose([
            PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
            AddBackgroundNoise(sounds_path=str(NOISE_DIR), min_snr_db=5.0, max_snr_db=10.0, p=1.0)
        ]),
        "ultra": Compose([ 
            PitchShift(min_semitones=-3, max_semitones=3, p=1.0),
            TimeStretch(min_rate=0.85, max_rate=1.15, leave_length_unchanged=True, p=0.8),
            AddBackgroundNoise(sounds_path=str(NOISE_DIR), min_snr_db=-5.0, max_snr_db=5.0, p=1.0)
        ])
    }
    
    aug = augmenters.get(mode)
    if not aug: return wav_np
    
    try:
        return aug(samples=wav_np, sample_rate=SAMPLE_RATE)
    except Exception as e:
        logger.warning(f"Augmentation failed for {mode}: {e}")
        return wav_np

def run_evaluation():
    ensure_directories()
    db = SessionLocal()
    audio_engine = AudioEngine()
    
    logger.info("🧠 Initializing Models for Evaluation...")
    cache = FingerprintCache.get_instance(db)
    ml_engine.load_models()
    dsp_matcher = HashMatcher(db=db, cache=cache)
    ml_matcher = MLMatcher()

    targets = []
    with open(CSV_BACKUP_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append(row)
            
    random.shuffle(targets)
    test_batch = targets[:NUM_SONGS_TO_TEST]

    # 🌟 NEW: A/B Split Trackers
    stats = {
        mode: {
            "total": 0, 
            "dsp_top1": 0, "dsp_top5": 0, "dsp_offset": 0,
            "ml_top1": 0, "ml_top5": 0, "ml_offset": 0
        } 
        for mode in ["clean", "soft", "hard", "ultra"]
    }

    logger.info(f"🚀 Starting A/B Evaluation on {NUM_SONGS_TO_TEST} songs...")

    for target in test_batch:
        true_id = int(target['song_id'])
        title = target['title']
        url = target['youtube_url']
        
        logger.info(f"📥 Downloading Source: {title}...")
        temp_src = str(TEMP_AUDIO_DIR / f"eval_src_{uuid.uuid4()}.wav")
        
        if not audio_engine.download_and_process(url, temp_src):
            continue
            
        wav_data, sr = sf.read(temp_src)
        if wav_data.ndim > 1: wav_data = wav_data.mean(axis=1) 
        
        total_len_sec = len(wav_data) / sr
        
        for clip_idx in range(CLIPS_PER_SONG):
            if total_len_sec <= QUERY_LEN_SEC: break
                
            true_start_sec = random.uniform(0, total_len_sec - QUERY_LEN_SEC)
            start_sample = int(true_start_sec * sr)
            end_sample = start_sample + int(QUERY_LEN_SEC * sr)
            clip_data = wav_data[start_sample:end_sample]
            
            for mode in ["clean", "soft", "hard", "ultra"]:
                stats[mode]["total"] += 1
                logger.info(f"🧪 Testing {mode.upper()} mode (Clip at {true_start_sec:.1f}s)")
                
                aug_clip = apply_augmentation(clip_data, mode)
                
                temp_query = str(TEMP_AUDIO_DIR / f"eval_query_{uuid.uuid4()}.wav")
                sf.write(temp_query, aug_clip, sr)
                
                # ==========================================
                # 🥊 PIPELINE A: SHAZAM (DSP) ONLY
                # ==========================================
                dsp_result = dsp_matcher.match(temp_query, db=db) 
                
                if dsp_result and dsp_result.get("candidates"):
                    cands = dsp_result["candidates"]
                    top_ids = [int(c["song_id"]) for c in cands]
                    
                    if true_id in top_ids[:5]: stats[mode]["dsp_top5"] += 1
                    
                    if true_id == top_ids[0]:
                        stats[mode]["dsp_top1"] += 1
                        if abs(cands[0]["offset_seconds"] - true_start_sec) <= TOLERANCE_SEC:
                            stats[mode]["dsp_offset"] += 1

                # ==========================================
                # 🥊 PIPELINE B: ML FUSION ONLY
                # ==========================================
                spec_v, pitch_v = ml_engine.extract_features(temp_query)
                s_res = ml_matcher.process_ml_stream(db, spec_v, SpectrogramEmbedding)
                p_res = ml_matcher.process_ml_stream(db, pitch_v, PitchEmbedding)
                
                # We feed the DSP candidates into fusion (as happens in the real app)
                fusion_res = ensemble_fusion(
                    dsp_results=dsp_result["candidates"] if dsp_result else [], 
                    spec_results=s_res, 
                    pitch_results=p_res
                )
                
                if fusion_res:
                    top_ids_ml = [int(c["song_id"]) for c in fusion_res]
                    
                    if true_id in top_ids_ml[:5]: stats[mode]["ml_top5"] += 1
                    
                    if true_id == top_ids_ml[0]:
                        stats[mode]["ml_top1"] += 1
                        if abs(fusion_res[0]["agreed_offset"] - true_start_sec) <= TOLERANCE_SEC:
                            stats[mode]["ml_offset"] += 1
                
                os.remove(temp_query)
                
        os.remove(temp_src)

    # --- PRINT FINAL REPORT ---
    print("\n" + "="*60)
    print("📊 FIND-TUNES A/B SYSTEM EVALUATION REPORT")
    print("="*60)
    print(f"{'MODE':<10} | {'METRIC':<15} | {'DSP (Shazam)':<15} | {'ML (Fusion)':<15}")
    print("-" * 60)
    
    for mode, data in stats.items():
        if data["total"] == 0: continue
        t = data["total"]
        
        d_t1 = (data["dsp_top1"] / t) * 100
        d_t5 = (data["dsp_top5"] / t) * 100
        d_off = (data["dsp_offset"] / max(data["dsp_top1"], 1)) * 100
        
        m_t1 = (data["ml_top1"] / t) * 100
        m_t5 = (data["ml_top5"] / t) * 100
        m_off = (data["ml_offset"] / max(data["ml_top1"], 1)) * 100
        
        print(f"{mode.upper():<10} | {'Top-1 Accuracy':<15} | {d_t1:>5.1f}% ({data['dsp_top1']}/{t})   | {m_t1:>5.1f}% ({data['ml_top1']}/{t})")
        print(f"{'':<10} | {'Top-5 Accuracy':<15} | {d_t5:>5.1f}% ({data['dsp_top5']}/{t})   | {m_t5:>5.1f}% ({data['ml_top5']}/{t})")
        print(f"{'':<10} | {'Offset (±2s)':<15} | {d_off:>5.1f}%           | {m_off:>5.1f}%")
        print("-" * 60)

    db.close()

if __name__ == "__main__":
    run_evaluation()