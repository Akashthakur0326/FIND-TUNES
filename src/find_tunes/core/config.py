# src/find_tunes/core/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

# Audio Params
SAMPLE_RATE = 16000
CHANNELS = 1

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
CSV_BACKUP_PATH = BASE_DIR / "data" / "database_backup.csv"
LOG_FILE_PATH = BASE_DIR / "logs" / "app_debug.log"

NOISE_DIR = BASE_DIR / "data" / "noise"

STAGING_CSV_PATH = BASE_DIR / "data" / "new_songs_staging.csv"

def ensure_directories():
    """Creates necessary local storage folders before the app starts."""
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    CSV_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@db:5432/findtunes")