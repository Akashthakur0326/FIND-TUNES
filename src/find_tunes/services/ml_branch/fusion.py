from collections import defaultdict
from loguru import logger

def ensemble_fusion(dsp_results, spec_results, pitch_results):
    """
    Applies P(song) = 1 - (1 - C_dsp)(1 - C_spec)(1 - C_pitch)
    Requires temporal alignment to fuse.
    """
    song_fusion = defaultdict(lambda: {"probs": [], "offsets": []})
    
    # Helper to load results
    def load_signals(results_list):
        for res in results_list:
            sid = res["song_id"]
            song_fusion[sid]["probs"].append(res["confidence_prob"])
            song_fusion[sid]["offsets"].append(res["offset_seconds"])

    """
    DSP 
    Returns per song:
            {
            song_id,
            confidence_prob,
            offset_seconds
            }
    Confidence is based on:
        fingerprint cluster size
        separation from 2nd best
        streaming duration

    ML Matcher (Spec + Pitch)
    Each model independently returns:
            {
            song_id,
            confidence_prob,
            offset_seconds
            }
    Confidence is based on:
        vector similarity
        temporal clustering
        sigmoid calibration

    Raw Audio
        ↓
    DSP Hash Matcher → C_dsp
    Spectrogram CNN  → C_spec
    Pitch CRNN       → C_pitch
        ↓
    Temporal Consistency Check
        ↓
    Joint Probability Fusion
        ↓
    Final Ranked Songs
    """
    load_signals(dsp_results)
    load_signals(spec_results)
    load_signals(pitch_results)

    final_ranking = []
    
    #For each song, it collects: All confidence probabilities || All predicted offsets
    for song_id, data in song_fusion.items():
        probs = data["probs"]
        offsets = data["offsets"]
        
        # 1. Temporal Check: Do the models agree on WHERE the song is?
        # If max offset - min offset > 5 seconds, the models disagree completely.
        if len(offsets) > 1 and (max(offsets) - min(offsets)) > 5.0:
            logger.warning(f"Models disagreed on time offset for Song {song_id}. Penalizing.")
            # Drastically reduce confidence
            probs = [p * 0.1 for p in probs]

        # 2. The Joint Probability Math
        inverse_product = 1.0
        for p in probs:
            inverse_product *= (1.0 - p)
            
        final_p = 1.0 - inverse_product
        
        final_ranking.append({
            "song_id": song_id,
            "final_confidence": round(final_p * 100, 2), # Convert back to % for API
            "agreed_offset": sum(offsets) / len(offsets) # Average the offsets
        })

    # Sort by highest joint confidence
    final_ranking.sort(key=lambda x: x["final_confidence"], reverse=True)
    return final_ranking