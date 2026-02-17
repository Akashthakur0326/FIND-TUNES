from sqlalchemy.orm import Session
from sqlalchemy import func
from find_tunes.core.config import ensure_directories
from find_tunes.core.database import SessionLocal, init_db, Song, Fingerprint, SpectrogramEmbedding, PitchEmbedding
from find_tunes.services.ingestion_cycle.ingestion import process_single_song

def run_test():
    print("📂 Creating directories...")
    ensure_directories()
    
    print("🛠️ Setting up database tables...")
    init_db()
    
    db = SessionLocal()
    
    # 🌟 NEW SONG to bypass the duplicate check
    test_title = "Without me"
    test_artist = "Eminem"
    
    try:
        print(f"🚀 Testing live ingestion for: {test_title} - {test_artist}")
        success = process_single_song(test_title, test_artist, db)
        
        if success:
            print("✅ Ingestion function finished. Verifying database records...")
            
            # Fetch the newly inserted song
            song = db.query(Song).filter(
                func.lower(Song.title) == test_title.lower(),
                func.lower(Song.artist) == test_artist.lower()
            ).first()
            
            if song:
                # 🌟 THE VERIFICATION STEP: Count the extracted features
                fp_count = db.query(Fingerprint).filter(Fingerprint.song_id == song.id).count()
                spec_count = db.query(SpectrogramEmbedding).filter(SpectrogramEmbedding.song_id == song.id).count()
                pitch_count = db.query(PitchEmbedding).filter(PitchEmbedding.song_id == song.id).count()
                
                print(f"\n📊 --- EXTRACTION RESULTS FOR '{song.title}' ---")
                print(f"   - 🎶 DSP Fingerprints (HashMatcher): {fp_count} rows")
                print(f"   - 🖼️ Spectrogram Embeddings (Siamese): {spec_count} rows")
                print(f"   - 🎵 Pitch Embeddings (CRNN): {pitch_count} rows")
                print(f"--------------------------------------------------\n")
                
                if fp_count > 0 and spec_count > 0 and pitch_count > 0:
                    print("🎉 PERFECT RUN! All features successfully generated and saved to PostgreSQL.")
                else:
                    print("⚠️ WARNING: The song was saved, but some ML features are missing! Check your extraction logic.")
            else:
                print("❌ Failed: Song was not found in the database despite returning True.")
        else:
            print("❌ Failed: Ingestion returned False.")
    finally:
        db.close()

if __name__ == "__main__":
    run_test()