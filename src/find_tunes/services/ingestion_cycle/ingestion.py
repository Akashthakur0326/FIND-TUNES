import os
import csv 
import re
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func

from find_tunes.services.ingestion_cycle.audio_engine import AudioEngine
from find_tunes.services.shazam_branch.dsp import DSPFingerprinter
from find_tunes.services.ml_branch.onnx_runner import ml_engine
from find_tunes.core.database import Song, Fingerprint, SpectrogramEmbedding, PitchEmbedding
from find_tunes.core.config import TEMP_AUDIO_DIR, CSV_BACKUP_PATH, STAGING_CSV_PATH

audio_engine = AudioEngine()
dsp_engine = DSPFingerprinter() 

"""
Input: (title, artist)
   ↓
YouTube Search
   ↓
Download & Normalize Audio
   ↓
DSP Fingerprinting
   ↓
ML Embedding Extraction
   ↓
Bulk Save to Database
   ↓
Backup to CSV


This ingestion function orchestrates the complete pipeline for adding a new song into the recognition system by searching and downloading verified audio from YouTube,
normalizing it to the required 16kHz mono format, extracting DSP fingerprints and ML embeddings outside of any database transaction to avoid locking, and then bulk-inserting all features into the database in a single fast commit.
It ensures duplicate prevention, efficient persistence, and system reliability while maintaining a CSV backup for traceability. 
In essence, this module builds the searchable fingerprint and embedding index that both the DSP and ML matchers rely on during real-time recognition

"""

def process_single_song(title: str, artist: str, db: Session):
    query = f"{title} {artist}"
    logger.info(f"🚀 Starting ingestion for: {query}")
    
    # --- 1. ROBUST DUPLICATE CHECK ---
    # Check URL OR exact Title/Artist match (case-insensitive)
    existing_song = db.query(Song).filter(
        func.lower(Song.title) == title.lower(),
        func.lower(Song.artist) == artist.lower()
    ).first()
    
    if existing_song:
        logger.warning(f"⚠️ Song already exists in Database: {title} by {artist}")
        return True

    # --- 2. DISCOVERY & ACQUISITION ---
    search_result = audio_engine.search(title, artist)
    if not search_result:
        logger.error(f"❌ YouTube search failed for: {query}")
        return False
        
    yt_url = search_result['url']
    
    url_exists = db.query(Song).filter(Song.youtube_url == yt_url).first()
    if url_exists:
        logger.warning(f"⚠️ YouTube URL already in DB (owned by '{url_exists.title}'). Skipping.")
        return True

    duration = search_result.get('duration')
    
    raw_name = f"{title}_{artist}".replace(" ", "_")
    safe_name = re.sub(r'[\\/*?:"<>|]', "", raw_name)
    
    temp_wav_path = str(TEMP_AUDIO_DIR / f"{safe_name}.wav")

    if not audio_engine.download_and_process(yt_url, temp_wav_path):
        return False

    # --- 3. HEAVY CPU WORK (Outside of DB Transaction!) ---
    try:
        logger.info("🔍 Running DSP Fingerprinting...")
        dsp_hashes = dsp_engine.process_file(temp_wav_path)
        
        logger.info("🧠 Running ML ONNX Extractions (Spectrogram & Pitch)...")
        spec_data, pitch_data = ml_engine.extract_features(temp_wav_path)
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return False

    # --- 4. FAST DB PERSISTENCE ---
    # We only touch the DB once all the heavy math is 100% finished
    try:
        new_song = Song(
            title=title, 
            artist=artist, 
            youtube_url=yt_url, 
            duration=duration
        )
        db.add(new_song)
        db.flush() # Get the new_song.id
        
        if dsp_hashes:
            fp_objs = [Fingerprint(song_id=new_song.id, hash_string=h, offset=t) for h, t in dsp_hashes]
            db.bulk_save_objects(fp_objs)
            
        if spec_data:
            spec_objs = [SpectrogramEmbedding(song_id=new_song.id, offset=t, embedding=vec) for t, vec in spec_data]
            db.bulk_save_objects(spec_objs)
            
        if pitch_data:
            pitch_objs = [PitchEmbedding(song_id=new_song.id, offset=t, embedding=vec) for t, vec in pitch_data]
            db.bulk_save_objects(pitch_objs)
            
        db.commit() # Lock and commit in milliseconds
        
        # --- 5. LIGHTWEIGHT CSV BACKUP ---
        file_exists = CSV_BACKUP_PATH.exists()
        with open(CSV_BACKUP_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["title", "artist", "youtube_url", "song_id"])
            writer.writerow([title, artist, yt_url, new_song.id])
            

        # --- 6. 🌟 NEW: DRIFT MANAGEMENT STAGING ---
        staging_exists = STAGING_CSV_PATH.exists()
        with open(STAGING_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not staging_exists:
                writer.writerow(["title", "artist", "youtube_url", "song_id"])
            writer.writerow([title, artist, yt_url, new_song.id])

        # Count the staging rows to see if we hit the Continual Learning Threshold
        with open(STAGING_CSV_PATH, mode='r', encoding='utf-8') as f:
            staged_count = sum(1 for row in f) - 1 # Subtract header

        logger.success(f"✅ Ingestion Complete: Saved DSP & ML Vectors for {title}")
        
        # 🚨 THE TRIGGER ALERT
        if staged_count >= 50:
            logger.warning(f"🚨 DRIFT THRESHOLD REACHED: {staged_count} new songs staged.")
            logger.warning("🚨 Continual Learning Pipeline should be triggered now!")
            
        return True
    
    except Exception as e:
        db.rollback() 
        logger.error(f"❌ DB transaction failed: {e}")
        return False
        
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)