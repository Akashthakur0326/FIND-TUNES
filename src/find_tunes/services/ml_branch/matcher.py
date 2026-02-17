import math
from collections import defaultdict
from sqlalchemy.orm import Session
from loguru import logger

# Import your established models
from find_tunes.core.database import Song, SpectrogramEmbedding, PitchEmbedding

class MLMatcher:
    """
    This class:
        Takes ML embeddings (spectrogram or pitch)
        Finds nearest vectors in database (KNN search)
        Aligns them by time offset
        Performs weighted temporal voting
        Outputs ranked songs with probability confidence
    """
    def __init__(self, tolerance_sec=1.5):
        # We bucket by 1.5 seconds (matches your model's hop length)
        self.bucket_size = tolerance_sec

    def _query_pgvector(self, db: Session, model_class, query_vector: list, top_k=5):
        """
        Uses pgvector's SQLAlchemy integration for speed and safety.
        Translates to the `<=>` (Cosine Distance) operator in PostgreSQL.
        Returns: list of (song_id, db_offset, distance)
        """
        # .cosine_distance() is provided by pgvector.sqlalchemy
        results = (
            db.query(
                model_class.song_id,
                model_class.offset,
                model_class.embedding.cosine_distance(query_vector).label("distance")
            )
            .order_by("distance")
            .limit(top_k)
            .all()
        )
        
        return [(row.song_id, row.offset, row.distance) for row in results]

    """
    In that part of the code, the system is performing temporal cluster-based matching over vector embeddings.
    For each query embedding window (generated every 1.5 seconds), it retrieves the top nearest neighbors from the database using cosine distance via pgvector.
    Instead of simply counting nearest matches, it computes an implied time offset (db_offset - query_time) to estimate where the query aligns inside each candidate song.
    These offsets are grouped into time buckets (1.5s resolution), and each neighbor contributes a Gaussian-weighted vote based on similarity (closer vectors contribute more). 
    This creates clusters of agreement for each song at specific offsets. The algorithm then selects the strongest cluster per song, ranks songs by their peak cluster strength,
    and converts the margin between the top candidates into a sigmoid-based confidence score.
    Finally, it returns the top 10 songs along with predicted alignment offset and calibrated probability, effectively combining nearest-neighbor search with Shazam-style temporal consistency verification.
    """
    def process_ml_stream(self, db: Session, model_results: list, model_class):
        """
        model_results: list of (query_time, query_vector)
        model_class: SpectrogramEmbedding OR PitchEmbedding
        Returns: Dict of top candidates with Sigmoid probabilities
        """
        if not model_results:
            return []

        votes = defaultdict(lambda: defaultdict(float))
        
        # 1. Gather KNN Neighbors
        for query_time, vector in model_results:
            # Pass the SQLAlchemy model directly instead of a table string
            neighbors = self._query_pgvector(db, model_class, vector)
            
            for song_id, db_offset, distance in neighbors:
                # 2. Implied Offset
                implied_offset = db_offset - query_time
                bucket = round(implied_offset / self.bucket_size)
                
                # 3. Weighted Vote (Closer distance = heavier vote)
                # Cosine distance is 0 to 2. Smaller is better.
                weight = math.exp(-(distance ** 2) / 0.5)  #closer matches contribute more
                
                # Smear the vote slightly to account for boundary issues
                votes[song_id][bucket] += weight
                votes[song_id][bucket - 1] += weight * 0.3
                votes[song_id][bucket + 1] += weight * 0.3

        # 4. Find Dominant Clusters
        ranked = []
        for song_id, bucket_map in votes.items():
            if not bucket_map: continue
            best_bucket = max(bucket_map, key=bucket_map.get)
            best_score = bucket_map[best_bucket]
            ranked.append((song_id, best_score, best_bucket))

        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if not ranked: return []

        # 5. Calculate Sigmoid Confidence (Same theory as DSP)
        best_song_id, best_score, _ = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        
        alpha, beta = 2.0, 0.5 # Distance separation vs Agreement count
        
        # Optimization: Fetch all Top 10 songs from DB in a single query
        top_10_ids = [r[0] for r in ranked[:10]]
        songs_from_db = {s.id: s for s in db.query(Song).filter(Song.id.in_(top_10_ids)).all()}
        
        final_results = []
        for song_id, score, bucket in ranked[:10]:
            song_obj = songs_from_db.get(song_id)
            if not song_obj: continue

            s1 = score
            s2 = second_score if song_id == best_song_id else best_score
            
            x = (alpha * (s1 - s2)) + (beta * s1) - 2.0
            x = max(-20, min(20, x))
            prob = 1 / (1 + math.exp(-x))
            
            final_results.append({
                "song_id": song_id,
                "title": song_obj.title,              # Added to match DSP output
                "artist": song_obj.artist,            # Added to match DSP output
                "youtube_url": song_obj.youtube_url,  # Added to match DSP output
                "confidence_prob": prob,              # 0.0 to 1.0
                "offset_seconds": bucket * self.bucket_size
            })
            
        return final_results