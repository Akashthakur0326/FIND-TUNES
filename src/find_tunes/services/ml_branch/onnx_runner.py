import os
import mlflow
import onnxruntime as ort
from loguru import logger
from find_tunes.services.ml_branch.preprocessor import AudioPreprocessor

class MLEngine:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.pitch_session = None
        self.spec_session = None
        # Notice: We REMOVED self._init_models() from here!

    def load_models(self):
        """Centralized check: Only loads if they aren't loaded already."""
        if self.pitch_session is not None and self.spec_session is not None:
            logger.debug("Models already loaded in memory. Skipping.")
            return

        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            logger.info("☁️ Loading DagsHub Champion Models for Inference...")
            
            crnn_path = mlflow.artifacts.download_artifacts("models:/CRNN_Pitch@champion")
            self.pitch_session = ort.InferenceSession(os.path.join(crnn_path, "artifacts", "pitch_crnn.onnx"))
            
            spec_path = mlflow.artifacts.download_artifacts("models:/Siamese_Spectrogram@champion")
            self.spec_session = ort.InferenceSession(os.path.join(spec_path, "artifacts", "spectrogram_cnn.onnx"))
            
            logger.success("✅ ML Engine Ready.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML models: {e}")

    def extract_features(self, wav_path: str):
        """Takes a WAV path and returns lists of (offset, embedding_vector)."""
        # Safety check: Ensure models are loaded before trying to extract
        self.load_models()
        
        if not self.pitch_session or not self.spec_session:
            logger.error("ONNX sessions are offline. Cannot extract features.")
            return [], []

        times, spec_batches, pitch_batches = self.preprocessor.process_into_windows(wav_path)
        if not times:
            return [], []

        spec_results = []
        pitch_results = []
        
        spec_input_name = self.spec_session.get_inputs()[0].name
        pitch_input_name = self.pitch_session.get_inputs()[0].name

        logger.info(f"🧠 Running ONNX Inference on {len(times)} windows...")
        for i, t in enumerate(times):
            spec_out = self.spec_session.run(None, {spec_input_name: spec_batches[i]})[0]
            spec_results.append((t, spec_out.flatten().tolist()))
            
            pitch_out = self.pitch_session.run(None, {pitch_input_name: pitch_batches[i]})[0]
            pitch_results.append((t, pitch_out.flatten().tolist()))

        return spec_results, pitch_results

# Singleton instance
ml_engine = MLEngine()