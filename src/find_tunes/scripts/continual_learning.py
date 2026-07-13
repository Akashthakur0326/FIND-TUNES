import os
import csv
import time
import random
import shutil
from pathlib import Path
from loguru import logger

# Project imports
from find_tunes.core.config import BASE_DIR, STAGING_CSV_PATH, CSV_BACKUP_PATH, ensure_directories
from find_tunes.services.ingestion_cycle.audio_engine import AudioEngine
from find_tunes.services.ml_branch.continual_learning.train_model import finetune_spec_model, finetune_pitch_model
from find_tunes.scripts.evaluate_and_promote import evaluate_and_promote

# 🌟 MODIFIED FOR FAST CLOUD TEST
REQUIRED_NEW_SONGS = 5
REPLAY_BUFFER_SIZE = 10
CL_DATA_DIR = BASE_DIR / "data" / "cl_training" / "originals"

def get_replay_buffer_urls():
    """Reads the staging and backup CSVs to mix new and old data."""
    new_songs = []
    old_songs = []
    
    # 1. Read New Songs
    if not STAGING_CSV_PATH.exists():
        return [], []
        
    with open(STAGING_CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_songs.append(row)
            
    if len(new_songs) < REQUIRED_NEW_SONGS:
        logger.info(f"Not enough new songs to trigger training. Current: {len(new_songs)}/{REQUIRED_NEW_SONGS}")
        return [], []
        
    # 2. Read Old Songs (The Replay Buffer)
    new_urls = {s['youtube_url'] for s in new_songs}
    if CSV_BACKUP_PATH.exists():
        with open(CSV_BACKUP_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['youtube_url'] not in new_urls:
                    old_songs.append(row)
                
    # Randomly sample old songs to prevent catastrophic forgetting
    sampled_old = random.sample(old_songs, min(REPLAY_BUFFER_SIZE, len(old_songs)))
    
    return new_songs, sampled_old

def prepare_training_audio():
    """Downloads the audio required for the fine-tuning cycle."""
    new_songs, old_songs = get_replay_buffer_urls()
    
    if not new_songs:
        return False
        
    logger.info(f"🚨 Continual Learning Triggered! Preparing {len(new_songs)} New + {len(old_songs)} Old songs.")
    
    # Clean up old training data
    if CL_DATA_DIR.exists():
        shutil.rmtree(CL_DATA_DIR)
    CL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    audio_engine = AudioEngine()
    all_targets = new_songs + old_songs
    
    downloaded_count = 0
    
    for i, target in enumerate(all_targets, 1):
        # 🌟 SMART LOOKUP: Search the directory for any file starting with the ID
        song_id = target.get('song_id')
        possible_files = list(CL_DATA_DIR.glob(f"{song_id}.wav"))
        
        # Also check for your eval_query names if they exist
        if not possible_files:
            possible_files = list(CL_DATA_DIR.glob(f"*{song_id}*")) 

        if possible_files:
            logger.info(f"✅ Found local file for {target.get('title')}: {possible_files[0].name}")
            downloaded_count += 1 
            continue # Already exists, skip download
        
        # ... proceed to download if not found ...
        safe_name = f"{target['song_id']}.wav"
        save_path = str(CL_DATA_DIR / safe_name)
        

        title = target.get('title', 'Unknown Title')
        logger.info(f"📥 Fetching {i}/{len(all_targets)}: {title}")
        success = audio_engine.download_and_process(target['youtube_url'], save_path)
        
        if success:
            downloaded_count += 1
            
        time.sleep(random.uniform(1.0, 3.0)) # Lowered jitter for faster testing
        
    logger.success(f"✅ Data Prep Complete. {downloaded_count} audio files ready for PyTorch.")
    return True

def run_continual_learning_pipeline():
    ensure_directories()
    
    logger.add(BASE_DIR / "logs" / "continual_learning.log", rotation="10 MB")
    logger.info("🤖 Starting Continual Learning Orchestrator...")
    
    data_ready = prepare_training_audio()
    if not data_ready:
        return
        
    try:
        logger.info("🧠 Passing data to PyTorch Siamese Network (Spectrogram)...")
        finetune_spec_model(data_dir=str(CL_DATA_DIR))
        
        logger.info("🧠 Passing data to PyTorch CRNN (Pitch)...")
        finetune_pitch_model(data_dir=str(CL_DATA_DIR))
        
        logger.info("⚖️ Entering Evaluation & Promotion Phase...")
        evaluate_and_promote()
        
        logger.success("✅ Continual Learning Cycle Complete.")

    except Exception as e:
        logger.error(f"❌ Continual Learning Pipeline failed: {e}")

if __name__ == "__main__":
    run_continual_learning_pipeline()