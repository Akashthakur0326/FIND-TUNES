import os
import uuid
import json
import logging
import urllib.parse as urlparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- Project Imports ---
from find_tunes.core.config import ensure_directories, TEMP_AUDIO_DIR, LOG_FILE_PATH
from find_tunes.core.database import get_db, SessionLocal, Song, SpectrogramEmbedding, PitchEmbedding, init_db
from find_tunes.services.ingestion_cycle.ingestion import process_single_song
from find_tunes.services.shazam_branch.matcher import HashMatcher, FingerprintCache
from find_tunes.services.ml_branch.onnx_runner import ml_engine
from find_tunes.services.ml_branch.matcher import MLMatcher
from find_tunes.services.ml_branch.fusion import ensemble_fusion
from find_tunes.core.audio import convert_webm_to_wav_array, save_bytes_to_wav 
from find_tunes.core.config import BASE_DIR


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🌟 Call ensure_directories FIRST so the logs folder exists!
    ensure_directories()
    
    # 🌟 Now attach the logger using the absolute, cloud-safe path
    logger.add(str(LOG_FILE_PATH), rotation="10 MB", retention="10 days", level="INFO")

    logger.info("🚀 Booting up Find Tunes API...")
    init_db()
    
    logger.info("🧠 Initializing Heavy Models in Background...")
    startup_db = SessionLocal()
    try:
        FingerprintCache.get_instance(startup_db)
    finally:
        startup_db.close()
        
    ml_engine.load_models()
    yield
    logger.info("🛑 Shutting down server.")

app = FastAPI(title="Find Tunes API", lifespan=lifespan)

# BASE_DIR point to src so we go to its patent which is where frontend is at 
frontend_path = BASE_DIR.parent / "frontend"
print(f"DEBUG: Looking for frontend at: {frontend_path}")

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SongRequest(BaseModel):
    title: str
    artist: str

def extract_yt_id(url: str):
    try:
        parsed = urlparse.urlparse(url)
        return urlparse.parse_qs(parsed.query)['v'][0]
    except:
        return ""

@app.get("/")
async def read_index():
    # Return index.html from the root frontend folder
    return FileResponse(str(frontend_path / "index.html"))


# --- INGESTION ENDPOINT ---
@app.post("/api/ingest/single")
async def ingest_single(
    request: SongRequest, 
    bg_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    bg_tasks.add_task(process_single_song, request.title, request.artist, db)
    return {"status": "Accepted", "message": f"'{request.title}' added to queue."}

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept() 
    logger.info("🎧 Client Connected: Ready to listen")

    cache = FingerprintCache.get_instance()
    matcher = HashMatcher(cache=cache)
    audio_buffer = bytearray()
    
    ml_triggered = False # State flag
    
    try:
        # --- THE LISTENING LOOP ---
        while True:
            try:
                # 🌟 Wait for audio, but timeout after 2 seconds of silence
                chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                audio_buffer.extend(chunk)
                
                # --- ROUTE A: SHAZAM (Fast Path) ---
                new_hashes = matcher.dsp.process_chunk(chunk)
                shazam_result = matcher.match_streaming_mode(new_hashes, db=db) 
                
                if shazam_result:
                    if shazam_result["is_confident"]:
                        logger.success("⚡ DSP Match Found! Ending stream.")
                        
                        raw_top_1 = shazam_result["candidates"][:1]
                        formatted_top_1 = []
                        
                        for song in raw_top_1:
                            formatted_top_1.append({
                                "song_id": song["song_id"],
                                "title": song["title"],
                                "artist": song["artist"],
                                "youtube_url": song["youtube_url"],
                                "youtube_id": extract_yt_id(song["youtube_url"]),
                                "confidence_percent": 100.0, 
                                "offset_seconds": song.get("offset_seconds", 0.0) # Fixed key to match your new matcher output
                            })

                        await websocket.send_json({
                            "status": "MATCHED", 
                            "method": "DSP_EXACT", 
                            "data": formatted_top_1 
                        })
                        return # End the connection completely
                        
                    # If we've listened for 10 seconds and still aren't confident, break to ML
                    elif shazam_result["seconds_processed"] > 15:
                        logger.warning("⚠️ 15s passed. DSP not confident. Moving to ML.")
                        ml_triggered = True
                        break 
                        
            except asyncio.TimeoutError:
                # User stopped recording (silence). If we have audio, process it!
                if len(audio_buffer) > 0:
                    logger.info("🔇 Silence detected. Moving to ML Fallback.")
                    ml_triggered = True
                    break
                else:
                    return # No audio received at all, just close.

        # --- ROUTE B: ML FALLBACK (Heavy Path) ---
        if ml_triggered:
            await websocket.send_json({"status": "PROCESSING", "message": "Analyzing deep audio features..."})
            
            temp_audio_path = str(TEMP_AUDIO_DIR / f"fallback_{uuid.uuid4()}.wav")
            
            try:
                save_bytes_to_wav(bytes(audio_buffer), temp_audio_path)
                
                # 🌟 Your ML Matcher logic remains exactly the same, 
                # because your fusion logic IS perfectly compatible with the static chunk!
                spec_vectors, pitch_vectors = ml_engine.extract_features(temp_audio_path)
                
                ml_matcher = MLMatcher()
                spec_results = ml_matcher.process_ml_stream(db, spec_vectors, SpectrogramEmbedding)
                pitch_results = ml_matcher.process_ml_stream(db, pitch_vectors, PitchEmbedding)
                
                final_ranking = ensemble_fusion(
                    dsp_results=shazam_result["candidates"] if shazam_result else [], 
                    spec_results=spec_results, 
                    pitch_results=pitch_results
                )
                
                if final_ranking and final_ranking[0]["final_confidence"] > 50.0:
                    top_3_ml = final_ranking[:3]
                    ml_song_ids = [r["song_id"] for r in top_3_ml]
                    songs_from_db = {s.id: s for s in db.query(Song).filter(Song.id.in_(ml_song_ids)).all()}
                    
                    ml_response_data = []
                    for rank_data in top_3_ml:
                        song_obj = songs_from_db.get(rank_data["song_id"])
                        if song_obj:
                            ml_response_data.append({
                                "song_id": song_obj.id,
                                "title": song_obj.title,
                                "artist": song_obj.artist,
                                "youtube_url": song_obj.youtube_url,
                                "youtube_id": extract_yt_id(song_obj.youtube_url),
                                "confidence_percent": rank_data["final_confidence"],
                                "offset_seconds": rank_data["agreed_offset"]
                            })
                    
                    await websocket.send_json({
                        "status": "MATCHED",
                        "method": "ML_ENSEMBLE",
                        "data": ml_response_data 
                    })
                else:
                    await websocket.send_json({"status": "FAILED", "message": "No match found."})
                    
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    except WebSocketDisconnect:
        logger.info("🔌 Client Disconnected manually")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass