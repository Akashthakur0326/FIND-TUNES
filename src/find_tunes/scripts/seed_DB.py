import csv
import time      
import random    
from pathlib import Path
from loguru import logger
from find_tunes.core.config import ensure_directories
from find_tunes.core.database import SessionLocal, init_db
from find_tunes.services.ingestion_cycle.ingestion import process_single_song

# 🌟 SET YOUR STARTING ROW HERE
START_INDEX = 0

def seed_database():
    logger.add("data/ingestion_debug.log", rotation="10 MB", retention="10 days", level="INFO")
    print("📂 Initializing directories and database...")
    ensure_directories()
    init_db()
    
    # Path to your cleaned CSV
    csv_path = Path(r"C:\Users\Admin\Desktop\FIND TUNES\data\retry_ingestion.csv")
    
    if not csv_path.exists():
        logger.error(f"❌ CSV not found at {csv_path}")
        return

    # Count total rows for progress tracking
    with open(csv_path, mode='r', encoding='utf-8') as f:
        total_songs = sum(1 for row in f) - 1 # Subtract header

    logger.info(f"🚀 Starting Bulk Ingestion of {total_songs} songs (Starting from row {START_INDEX})...")
    
    db = SessionLocal()
    success_count = 0
    fail_count = 0
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for index, row in enumerate(reader, start=1):
                
                # 🌟 THE SKIP LOGIC
                if index < START_INDEX:
                    # Print a debug message every 10 rows so we know it's fast-forwarding
                    if index % 10 == 0:
                        logger.debug(f"⏭️ Fast-forwarding... skipped row {index}")
                    continue

                title = row.get("title")
                artist = row.get("artist")
                
                if not title or not artist:
                    continue
                    
                logger.info(f"--- Processing {index}/{total_songs}: {title} by {artist} ---")
                
                # The core loop: Process one by one, sequentially
                success = process_single_song(title, artist, db)
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    
                # THE JITTER SAFEGUARD
                if index < total_songs: 
                    sleep_time = random.uniform(5.0, 12.0) 
                    logger.info(f"⏳ Jitter: Sleeping for {sleep_time:.1f}s to avoid YouTube IP ban...\n")
                    time.sleep(sleep_time)
                    
    except KeyboardInterrupt:
        logger.warning("🛑 Seeding interrupted by user. Safe exit triggered.")
    finally:
        db.close()
        logger.info("==========================================")
        logger.info(f"🏁 Seeding Complete. Success: {success_count} | Failed: {fail_count}")
        logger.info("==========================================")

if __name__ == "__main__":
    seed_database()