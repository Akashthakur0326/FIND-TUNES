import csv
from pathlib import Path
from loguru import logger
from find_tunes.core.database import SessionLocal, Song

RETRY_CSV_PATH = Path("data/retry_ingestion.csv")

def sanitize_database():
    logger.info(" Starting Database Sanity Check (Incompleteness & Duration)...")
    db = SessionLocal()
    
    try:
        # 1. Any of the 3 features missing (Incomplete)
        # 2. OR Duration > 600 seconds (Anti-Bias/OOM protection)
        to_purge = db.query(Song).filter(
            ~Song.fingerprints.any() | 
            ~Song.spec_embeddings.any() | 
            ~Song.pitch_embeddings.any() |
            (Song.duration > 600)
        ).all()

        if not to_purge:
            logger.success("✅ Database is healthy. No biased or incomplete records found.")
            return

        logger.warning(f"⚠️ Found {len(to_purge)} records violating integrity rules. Purging...")

        retry_data = []

        for song in to_purge:
            reason = "Incomplete" if song.duration <= 600 else f"Too Long ({song.duration}s)"
            logger.info(f"   🗑️ Deleting {song.title} | Reason: {reason}")
            
            # Only add to retry list if it's incomplete (don't retry the hour-long ones!)
            if song.duration <= 600:
                retry_data.append({"title": song.title, "artist": song.artist})
            
            db.delete(song)

        # Update the retry CSV
        if retry_data:
            with open(RETRY_CSV_PATH, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["title", "artist"])
                writer.writeheader()
                writer.writerows(retry_data)
            logger.info(f"📝 Retry list updated with {len(retry_data)} songs.")

        db.commit()
        logger.success(f"🧹 Cleanup complete. {len(to_purge)} total songs removed.")

    except Exception as e:
        db.rollback()
        logger.error(f"❌ DB Sanity Check failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    sanitize_database()