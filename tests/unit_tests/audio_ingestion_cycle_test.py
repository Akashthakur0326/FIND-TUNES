import pytest
from unittest.mock import patch, MagicMock
from find_tunes.services.ingestion_cycle.audio_engine import AudioEngine

engine = AudioEngine()

class TestAudioEngineSearch:
    """Groups all search-related tests together."""

    # 🌟 Notice the new patch target: we target it exactly where it is imported in your code
    @patch("find_tunes.services.ingestion_cycle.audio_engine.yt_dlp.YoutubeDL")
    def test_search_finds_verified_topic_channel(self, mock_ytdl_class):
        # 1. Setup the fake context manager (the 'with' block)
        mock_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_instance
        
        # 2. Provide the fake return data
        mock_instance.extract_info.return_value = {
            "entries": [
                {"url": "http://fake.url/1", "title": "Video", "uploader": "VEVO", "duration": 250},
                {"url": "http://fake.url/2", "title": "Audio", "uploader": "The Weeknd - Topic", "duration": 200}
            ]
        }
        
        result = engine.search("Blinding Lights", "The Weeknd")
        
        assert result is not None
        assert result["url"] == "http://fake.url/2"

    @patch("find_tunes.services.ingestion_cycle.audio_engine.yt_dlp.YoutubeDL")
    def test_search_uses_fallback_when_no_topic_found(self, mock_ytdl_class):
        mock_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "entries": [
                {"url": "http://fake.url/top", "title": "Some Cover", "uploader": "RandomUser", "duration": 180},
                {"url": "http://fake.url/bad", "title": "Live Show", "uploader": "FanCam", "duration": 300}
            ]
        }
        
        result = engine.search("Obscure Song", "Indie Band")
        assert result["url"] == "http://fake.url/top" 

    @patch("find_tunes.services.ingestion_cycle.audio_engine.yt_dlp.YoutubeDL")
    def test_search_returns_none_on_empty_results(self, mock_ytdl_class):
        mock_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {"entries": []}
        
        result = engine.search("Ghost Song", "Nobody")
        assert result is None

    @patch("find_tunes.services.ingestion_cycle.audio_engine.yt_dlp.YoutubeDL")
    def test_search_ignores_invalid_durations(self, mock_ytdl_class):
        mock_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "entries": [
                {"url": "http://fake.url/teaser", "title": "Teaser", "uploader": "Artist - Topic", "duration": 15},
                {"url": "http://fake.url/loop", "title": "10 Hour Loop", "uploader": "Artist - Topic", "duration": 36000},
                {"url": "http://fake.url/real", "title": "Actual Song", "uploader": "Artist - Topic", "duration": 210}
            ]
        }
        
        result = engine.search("Normal Song", "Artist")
        assert result["url"] == "http://fake.url/real"