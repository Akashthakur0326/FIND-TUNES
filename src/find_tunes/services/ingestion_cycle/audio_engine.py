import uuid
import os
import yt_dlp
from pydub import AudioSegment
from loguru import logger
from find_tunes.core.config import SAMPLE_RATE, CHANNELS
"""
this class handles three stages:
    1. Search YouTube for official audio
    2. Download best audio stream
    3. Normalize to your system's audio format
"""
class AudioEngine:
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

    def search(self, title: str, artist: str) -> dict | None:
        query = f"{title} {artist} official audio"
        logger.info(f"🔎 Searching for '{title}' by '{artist}'...")
        
        # 🌟 NATIVE FILTERING: Only look for videos between 1 min and 10 mins
        search_opts = {
            'quiet': True, 
            'extract_flat': True, 
            'noplaylist': True,
            'match_filter': yt_dlp.utils.match_filter_func("duration >= 60 & duration <= 600")
        }
        
        try:
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                # We still search for top 10, but ydl will skip the long ones natively
                info = ydl.extract_info(f"ytsearch10:{query}", download=False)
                
                if not info or 'entries' not in info or not info['entries']:
                    logger.error(f"❌ No suitable audio found for: {query}")
                    return None

                results = [e for e in info['entries'] if e is not None]
                if not results: return None

                artist_clean = artist.lower().replace(" ", "")

                # Heuristic Pass
                for entry in results:
                    uploader = entry.get('uploader', '').lower().replace(" ", "")
                    # The duration check is now redundant but kept for absolute safety
                    if artist_clean in uploader:
                        title_low = entry.get('title', '').lower()
                        if 'topic' in uploader or 'vevo' in uploader or 'audio' in title_low:
                            logger.success(f"🎯 Verified Match: {entry['title']}")
                            return self._format_entry(entry)

                # Fallback Pass
                logger.warning(f"⚠️ No perfect match for {artist}. Using top result.")
                return self._format_entry(results[0])
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None

    def _format_entry(self, entry: dict) -> dict:
        return {
            'url': entry['url'],
            'title': entry.get('title'),
            'uploader': entry.get('uploader'),
            'duration': entry.get('duration')
        }

    def download_and_process(self, url: str, output_path: str) -> bool:
        temp_id = str(uuid.uuid4())
        temp_filename = f"temp_{temp_id}"
        temp_full_path = os.path.join(os.path.dirname(output_path), temp_filename)
        
        opts = self.ydl_opts.copy()
        opts['outtmpl'] = temp_full_path
        downloaded_file = f"{temp_full_path}.wav" # yt-dlp appends this

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            if not os.path.exists(downloaded_file):
                logger.error(f"❌ yt-dlp failed to produce: {downloaded_file}")
                return False

            # Normalize to your specific ML requirements (16kHz, Mono)
            audio = AudioSegment.from_file(downloaded_file)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
            audio.export(output_path, format="wav")

            return True

        except Exception as e:
            logger.error(f"❌ AudioEngine Processing failed: {e}")
            return False
            
        finally:
            # 🌟 THE FIX: This runs NO MATTER WHAT (success or crash).
            # It ensures the random UUID files are always deleted.
            if os.path.exists(downloaded_file):
                try:
                    os.remove(downloaded_file)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not delete temp file {downloaded_file}: {cleanup_error}")