import time
from collections import defaultdict, Counter
from sqlalchemy.orm import Session
from find_tunes.core.database import Fingerprint, Song
from find_tunes.services.shazam_branch.dsp import DSPFingerprinter
from loguru import logger
import math


class FingerprintCache:
    _instance = None
    
    @classmethod
    def get_instance(cls, db: Session = None):
        if cls._instance is None:
            if db is None:
                raise ValueError("Cache not initialized. Must provide DB session.")
            cls._instance = cls(db)
        return cls._instance

    def __init__(self, db: Session):
        logger.info("Initializing RAM Cache for Fingerprints...")
        start_t = time.time()
        
        # Structure: { hash_string: [ (song_id, offset), (song_id, offset) ] }
        self.db_map = defaultdict(list)
        
        # Fetch everything. Depending on DB size, this might take a few seconds.
        # Fetching in chunks is better if you have millions of rows.
        all_prints = db.query(Fingerprint.hash_string, Fingerprint.song_id, Fingerprint.offset).all()
        
        for h_str, s_id, offset in all_prints:
            self.db_map[h_str].append((s_id, offset))
            
        logger.info(f"Loaded {len(self.db_map)} unique hashes into RAM in {time.time() - start_t:.2f}s.")
        
    def get_matches(self, hash_strings: list):
        """Returns a list of tuples: (song_id, offset) for the given hashes."""
        results = []
        for h_str in hash_strings:
            # .get() is instant. Returns empty list if hash not found.
            matches = self.db_map.get(h_str, [])
            for s_id, offset in matches:
                # Keep track of which hash string matched for mapping later
                results.append((h_str, s_id, offset)) 
        return results
    
    
class HashMatcher:
    """
    Input: query audio
    Output: best matching songs ranked by querying the DB 

    The HashMatcher class implements the fingerprint matching and scoring logic for audio recognition.
    It takes a query audio file, converts it into spectral hashes using the DSP module, and retrieves matching hashes from the database.
    For each candidate song, it verifies rhythmic consistency between neighboring fingerprint pairs by comparing time differences in the query and database.
    When rhythm alignment passes within a tolerance, it computes an implied offset representing where the query clip aligns inside the full song. 
    These offsets are then clustered using a sliding window approach to find the largest group of agreeing alignment votes.
    The size of this dominant cluster becomes the confidence score for that song. Songs exceeding a dynamically calculated minimum score are ranked and returned as the final recognition results.
    """
    def __init__(self, db: Session,cache: FingerprintCache, min_score_threshold: int = 5):
        self.dsp = DSPFingerprinter()
        self.cache = cache 


        #self.MIN_SCORE = min_score_threshold #Minimum cluster size required to accept a match
        #removed the above to make min scoring more dynamic for bigger songs with more hashes 
        
        # 1. Relative Tolerance (Option B)
        # Checks if rhythm is consistent between neighbors
        self.RELATIVE_TOLERANCE = 3  # Frames (~100ms)
        self.LOOK_AHEAD_WINDOW = 5   # Check next 5 peaks
        
        # We allow the "Global Offset" to drift this much across the whole clip.
        # This allows for 10s of audio to stretch/shrink without breaking the cluster.
        self.GLOBAL_BIN_SIZE = 40 # Allows offsets within ~1.5 seconds to count as same cluster used in approach 1 of binning
        self.GLOBAL_TOLERANCE = 40 # for the sliding window approach used in approach 2 of sliding window 

        # --- STREAMING STATE ---
        self.streaming_offset_votes = defaultdict(lambda: defaultdict(int))
        # Structure:
        # {
        #   song_id: {
        #       bucket_index: vote_count
        #   }
        # }

        self.streaming_total_frames = 0  # Track highest frame seen so far

        # --- STREAMING DECISION PARAMETERS ---
        self.MIN_STREAM_SCORE = 20        # Absolute minimum votes
        self.MIN_RATIO = 1.5              # Top1 must be 1.5x Top2
        self.MIN_GAP = 8                  # Absolute gap requirement
        self.MIN_SECONDS = 8              # Minimum listening time

        self.locked_song_id = None        # Prevent flip-flopping
        self.locked_offset = None


    def reset_streaming_state(self):
        """
        Call this when microphone session resets.
        """
        self.streaming_offset_votes.clear()
        self.streaming_total_frames = 0
        self.locked_song_id = None
        self.locked_offset = None

    def match(self, file_path: str, db: Session, top_n: int = 10):
        start_time = time.time()
        """
        Given audio file → return best matching songs
        Batch matcher updated to return probabilities for Fusion compatibility.
        """
        
        # --- PHASE 1: DSP & LOOKUP ---
        query_hashes = self.dsp.process_file(file_path) #query audio to list of hash_str * anchor_time

        if not query_hashes:
            return {"is_confident": False, "candidates": []}
            
        query_hashes.sort(key=lambda x: x[1]) # sort acc to the time in which anchors appear 

        MIN_SCORE = max(5, len(query_hashes) // 10) #to make the min score dynamic based on query length 

        query_hash_strings = [h[0] for h in query_hashes] #take out the hashes out of the list 

        # OLD DB QUERY (REMOVED FOR PERFORMANCE)
        # matches = self.db.query(Fingerprint).filter(
        #     Fingerprint.hash_string.in_(query_hash_strings)
        # ).all()
        #
        # WHY REMOVED:
        # Streaming & batch now use RAM cache instead of DB query.
        # This eliminates repeated SQL calls and makes matching O(1) dictionary lookup.
        
        matches = self.cache.get_matches(query_hash_strings)  # Use blazing fast RAM cache

        # Organize Data
        matches_by_song = defaultdict(list)
        query_map = defaultdict(list)
        
        for h, t in query_hashes:
            query_map[h].append(t) #times when this hash appears in the QUERY audio hash → [times in query]
            
        for h_str, s_id, offset in matches:
            matches_by_song[s_id].append((h_str, offset)) #sepration so now each song is storeed seperately 

        # --- PHASE 2: SCORING (THE HYBRID) ---
        results = []

        """
        offset = the time (in frames, not seconds) at which the anchor peak of a hash occurred in the song.
        DB has a combination of hash string(f1, f2, Δt)(watch the dsp code ) and anchor time offset simply refer to the anchor time 
        
        as we process a song clip and use dsp to make a fingerprint out of it which is basically a hash and anchor time based pattern recognistion 
        we go to the DB and look for similar hashes(not the time part of the fingerprint)

        Then we group the matches with the song id and now score all the matches per song wet to the time they appear on the audio clip and in the recorded song from DB 

        Database says:
            “This pattern happened at 12.5 seconds in the real song”

        Query says:
            “I heard this pattern at 5 seconds into the clip”

        So that fingerprint says:
            “If this is the same song, the clip must have started at 12.5 - 5 = 7.5 seconds into the song.”

        That fingerprint casts one vote:
            start ≈ 7.5s

        Clustering = confidence
            You dont care about exact equality.
            You care about agreement.

            Big cluster → real song
            Scattered votes → coincidence

            The size of the cluster is the score.
        """

        for song_id, db_fingerprints in matches_by_song.items():
            valid_offsets = [] 
            
            for h_str, time_db in db_fingerprints:
                if h_str in query_map:
                    # 🌟 GUARD 1: Ignore "Stop-Hashes"
                    # If this hash appears an absurd number of times in a 15s clip, it is generic noise.
                    if len(query_map[h_str]) > 25:
                        continue

                    for time_clip in query_map[h_str]:
                        """
                        he implied offset is the estimated starting position of the query clip within the full song.
                        It is computed by subtracting the time at which a fingerprint occurs in the query 
                        from the time at which the same fingerprint occurs in the database song.
                        """
                        implied_offset = time_db - time_clip
                        valid_offsets.append(implied_offset)
                # 🌟 GUARD 2: The Hard RAM Limit
                # If we've collected 50,000 votes for a single song, stop counting. 
                # 50k is more than enough data to find the dominant cluster.
                if len(valid_offsets) > 50000:
                    break
                
            if not valid_offsets:
                continue

            # APPROACH 2: Sliding window approach 
            """
            Now (with the sliding window approach):
                Cluster defined by sliding window
                No artificial boundaries
                Offsets 998 and 1000 naturally cluster

            Instead of: Force offsets into bins of width 50
            We now: Find the largest group of offsets that lie within GLOBAL_TOLERANCE frames
            """

            # APPROACH 2: Sliding window approach (OPTIMIZED O(N))
            valid_offsets.sort()

            best_score = 0
            best_offset = None
            left = 0
            current_sum = 0 # 🌟 NEW: Track the sum dynamically

            for right in range(len(valid_offsets)):
                # 🌟 Add the new incoming value to our running total
                current_sum += valid_offsets[right] 

                # Shrink window until within tolerance
                while valid_offsets[right] - valid_offsets[left] > self.GLOBAL_TOLERANCE:
                    # 🌟 Subtract the outgoing value before moving the left pointer
                    current_sum -= valid_offsets[left] 
                    left += 1
                
                window_size = right - left + 1
                
                if window_size > best_score:
                    best_score = window_size
                    # 🌟 Instant O(1) math! No more slicing or looping!
                    best_offset = current_sum // window_size 

            if best_score >= MIN_SCORE:
                results.append((song_id, best_score, best_offset))

        if not results:
            return {"is_confident": False, "candidates": []}

        # --- RANKING & PROBABILITY CALCULATION ---
        results.sort(key=lambda x: x[1], reverse=True)
        top_candidates = results[:top_n]
        
        best_song_id, best_score, best_bucket = top_candidates[0]
        second_score = top_candidates[1][1] if len(top_candidates) > 1 else 0

        # Strict rules for definitive lock
        is_confident = (
            best_score >= self.MIN_STREAM_SCORE and 
            (best_score - second_score) >= self.MIN_GAP and
            (best_score / max(second_score, 1)) >= self.MIN_RATIO
        )

        """
        NEW ADDITION: Probabilistic scoring via Sigmoid
        WHY:
            Batch mode must now be compatible with Ensemble Fusion.
            Fusion requires probability scores (0–1), not raw cluster sizes.
        """

        song_ids = [r[0] for r in top_candidates]
        songs_from_db = {s.id: s for s in db.query(Song).filter(Song.id.in_(song_ids)).all()}
        
        weight_a, weight_b, penalty_c = 0.5, 0.2, 2.0 
        
        # Estimate seconds processed (batch mode approximation)
        seconds_processed = len(query_hashes) * self.dsp.hop_length / self.dsp.sample_rate

        results_list = []
        for song_id, score, offset_frames in top_candidates:
            song_obj = songs_from_db.get(song_id)
            if not song_obj:
                continue
            
            # Sigmoid confidence logic
            s1 = score
            s2 = second_score if song_id == best_song_id else best_score

            x_val = (weight_a * (s1 - s2)) + (weight_b * s1) - (penalty_c / max(seconds_processed, 1))
            x_val = max(-20, min(20, x_val)) 
            confidence_prob = 1 / (1 + math.exp(-x_val))
            
            results_list.append({
                "song_id": song_id,
                "title": song_obj.title,
                "artist": song_obj.artist,
                "youtube_url": song_obj.youtube_url,
                "confidence_prob": round(confidence_prob, 4),
                "raw_votes": score,
                "offset_frames": offset_frames,
                "offset_seconds": round((offset_frames * self.dsp.hop_length) / self.dsp.sample_rate, 2),
                "speed": round(time.time() - start_time, 3)
            })

        return {
            "is_confident": is_confident,
            "seconds_processed": round(seconds_processed, 2),
            "candidates": results_list
        }

def match_streaming_mode(self, incoming_hashes, db: Session, top_n: int = 10):
        """
        Streaming matcher.
        Accepts hashes from DSP streaming mode.
        Returns a normalized probability distribution of the Top N candidates.
        """
        import math # Needed for Sigmoid calculation

        if not incoming_hashes:
            return None

        # Convert processed frames → seconds
        self.streaming_total_frames = max(
            self.streaming_total_frames,
            max(t for _, t in incoming_hashes)
        )
        seconds_processed = (self.streaming_total_frames * self.dsp.hop_length) / self.dsp.sample_rate #to know exactly how many seconds of audio has been processed

        # --- VOTING PHASE ---
        hash_strings = [h[0] for h in incoming_hashes]
        ram_hits = self.cache.get_matches(hash_strings) #returns every single time those hashes appear anywhere in your entire database
        #ram_hits returns (hash_string, song_id, db_offset)

        query_map = defaultdict(list)
        for h, t in incoming_hashes:
            query_map[h].append(t)

        for h_str, song_id, db_offset in ram_hits:
            if h_str not in query_map: continue
            for clip_time in query_map[h_str]:
                implied_offset = db_offset - clip_time
                bucket = implied_offset // self.GLOBAL_TOLERANCE
                for b in (bucket - 1, bucket, bucket + 1):
                    self.streaming_offset_votes[song_id][b] += 1

        # --- RANKING & PERCENTAGE MATH ---
        ranked = []
        for song_id, bucket_map in self.streaming_offset_votes.items():
            if not bucket_map: continue
            best_bucket = max(bucket_map, key=bucket_map.get)
            best_score = bucket_map[best_bucket] #For every single song, the system looks at all the buckets. The bucket with the most votes is the Dominant Cluste
            ranked.append((song_id, best_score, best_bucket))

        if not ranked:
            return {"is_confident": False, "candidates": []}

        # Sort by raw vote count
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Take the requested Top N
        top_candidates = ranked[:top_n]
        
        # Calculate the denominator (Sum of votes for the Top N)
        # Using LaTeX logic: Confidence_i = Score_i / Sum(Scores_1..N)
        total_top_votes = sum(score for _, score, _ in top_candidates)
        
        # --- DECISION LOGIC (The "Fail Safe" Trigger) ---
        best_song_id, best_score, best_bucket = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        score_gap = best_score - second_score
        ratio = best_score / max(second_score, 1)

        """
        The confidence logic in the streaming matcher evaluates whether the top-ranked song is both strongly supported and clearly separated from competing candidates before declaring a reliable match.
        First, it measures the dominant cluster size (best_score), which represents how many fingerprint offsets consistently agree on the same alignment for that song.
        It then compares this score to the second-best song using both an absolute gap (score_gap) and a dominance ratio (ratio), ensuring the leader is not only strong but also meaningfully ahead.
        Additionally, it enforces a minimum listening duration (seconds_processed) so early, unstable matches cannot trigger prematurely.
        Beyond this hard locking rule, a sigmoid-based formula converts cluster strength, separation margin, and a time-based penalty into a smooth probability score between 0 and 1.
        Together, these mechanisms ensure the system only becomes confident when sufficient evidence accumulates, the match is clearly dominant, and enough audio has been processed to avoid false positives.
        """

        # old strict rules to determine a definitive "Lock"
        #Does the top song have enough total votes , Is the gap between #1 and #2 large enough , Have we listened to the audio for long enough to be sure
        is_confident = (
            best_score >= self.MIN_STREAM_SCORE and 
            score_gap >= self.MIN_GAP and
            ratio >= self.MIN_RATIO and
            seconds_processed >= self.MIN_SECONDS
        )#This compares Top 1 vs Top 2 songs

        if is_confident:
            self.locked_song_id = best_song_id
            self.locked_offset = best_bucket * self.GLOBAL_TOLERANCE

        # --- BUILD THE RICH RETURN DICT ---
        # 1. Fetch all Top N songs from DB in a single query (Optimization)
        song_ids = [r[0] for r in top_candidates]
        
        songs_from_db = {s.id: s for s in db.query(Song).filter(Song.id.in_(song_ids)).all()}

        # Sigmoid Weights for Probabilistic Scoring
        # a: weight of the gap (separability)
        # b: weight of absolute score (evidence volume)
        # c: penalty for being too early in the stream (prevents early false positives), seconds_processed is small --> penalty large
        weight_a, weight_b, penalty_c = 0.5, 0.2, 5.0 

        results_list = []
        for song_id, score, bucket in top_candidates:
            song_obj = songs_from_db.get(song_id)
            if not song_obj: continue
            
            # --- OLD PERCENTAGE LOGIC (Kept for diversity/reference) ---
            # Normalize to percentage (0.0 to 100.0)
            confidence_pct = (score / total_top_votes) * 100 if total_top_votes > 0 else 0
            
            # --- NEW BAYESIAN PROBABILITY LOGIC ---
            # S1 this song’s dominant cluster score. S2 is the second best score overall for this song.
            s1 = score
            s2 = second_score if song_id == best_song_id else best_score
            
            # Sigmoid: 1 / (1 + e^-x)
            x_val = (weight_a * (s1 - s2)) + (weight_b * s1) - (penalty_c / max(seconds_processed, 1))
            
            # Prevent math overflow from exponentiation
            x_val = max(-20, min(20, x_val)) 
            confidence_prob = 1 / (1 + math.exp(-x_val))
            # ---------------------------------------------------------
            
            offset_frames = bucket * self.GLOBAL_TOLERANCE
            
            results_list.append({
                "song_id": song_id,
                "title": song_obj.title,
                "artist": song_obj.artist,
                "youtube_url": song_obj.youtube_url,
                "confidence_percent": round(confidence_pct, 2), # Original field
                "confidence_prob": round(confidence_prob, 4),   # NEW: 0.0 to 1.0 (Used for Ensemble Fusion)
                "raw_votes": score,
                "offset_frames": offset_frames,
                "offset_seconds": round((offset_frames * self.dsp.hop_length) / self.dsp.sample_rate, 2)
            })

        return {
            "is_confident": is_confident, #If it's false, the WS handler routes the current audio buffer to the ml_engine
            "seconds_processed": round(seconds_processed, 2),
            "candidates": results_list
        }