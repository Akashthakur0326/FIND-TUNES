import numpy as np
import librosa
import hashlib
from loguru import logger

#DSP = digital signal processing  process the analysis of digital signals to retrieve essential information or improve specific features through algorithms and techniques
class DSPFingerprinter:
    def __init__(self):
        # 1. FFT Configuration
        self.n_fft = 4096          # High frequency resolution
        self.hop_length = 512      # Standard overlap
        self.sample_rate = 16000   # 16 kHz is enough to capture human-audible content up to 8 kHz
        
        """
        The microphone measures air pressure 16,000 times per second 

        1 sample = 1 / 16000 sec ≈ 0.0625 ms
        y[0] → amplitude at t = 0   y[1] → amplitude at t = 0.0625 ms   y[16000] → amplitude at t = 1 second

        n_fft=4096 
        Number of time samples per FFT window
        Time covered by one FFT: 4096 / 16000 = 0.256 seconds
        This helps as Notes last ~100 to 500 ms

        This number is both:
            the window size in time
            the FFT size in frequency (frequency_bins = (4096 / 2) + 1 = 2049)
        
        When you compute an FFT, you answer this question: “Given N time samples, which N frequencies explain them?”
        So if you give FFT:
            4096 time-domain samples
            You get: 4096 frequency-domain coefficients

        Why does a time window define frequency bins?
        
        This is the key conceptual leap: Frequency resolution depends on how long you listen.
        If you listen longer:
            You can distinguish closer frequencies
        If you listen shorter:
            Frequencies blur together

        Mathematically: frequency_resolution = sample_rate / n_fft
        
        With your values: Δf = 16000 / 4096 ≈ 3.9 Hz
        
        
        Meaning:
        Each FFT bin represents ~3.9 Hz
        You can tell 440 Hz from 444 Hz
        You CANNOT with n_fft=512
        
        Why only 2049 bins instead of 4096? FFT output is complex and symmetric. 

            X[k] = conjugate(X[N - k])  So half the spectrum is redundant.



        hop_length = 512 (How often you compute FFTs)
        This is how much we move the window forward.
        512 / 16000 ≈ 0.032 sec ≈ 32 ms

        So the system samples the song : ~31(1 / 0.032 = 31.25) time snapshots per second

        The Overlap: Since the window is 4096 wide, but we only move 512 steps, we are Overlapping by 87%. This ensures we don't miss a note that happens right on the edge of a window.

        min_hash_time_delta = 4
        This is in frames, not seconds.

        Convert it:4 * 32 ms = 128 ms
        So this enforces: Only pair peaks that are at least ~0.13 seconds apart     

        self.max_hash_time_delta = 200
            200 * 32 ms ≈ 6.4 seconds
            This is the target zone length. 
            
            Meaning: Anchor peak Look ahead ~6 sec --> Form geometric patterns


        """
        
        """
        Audio in raw form is time-domain: Amplitude vs Time
        This tells How loud is the signal at this instant But it does not tell Which notes are present which frequencies dominate

        FFT (Fast Fourier Transform): Time domain  to  Frequency domain rather than normal wav amplitude vs time 

        A mathematical algorithm that takes a signal and unweaves it, telling you exactly which frequencies are present.
        as if we apply fft directly we get a single freq which completeley loses the time fn so we use windows for the song 16kHz notation and use 4096 windows and apply FFt on each
        so each winodw cover apxx 0.25 sec smaller window size is better for time 


        By default, librosa applies a Hann Window (which is very similar to Hamming) automatically.This prevents spectral leakage at the boundry of the window 
        
        A spectrogram is simply many FFTs stacked over time and 
        Y-axis: Frequency
        X-axis: Time
        Color: Intensity (loudness)

        Why FFT is perfect for Shazam
            Volume changes → amplitudes scale → frequencies stay
            Compression artifacts → waveform changes → peaks stay
            Noise → random → doesnt form stable frequency peaks
        """

        # 2. Hashing Configuration (The "Target Zone")
        self.fan_value = 10        # How many neighbors to connect to which tells that each peak connects to N neigh ahead in area of 200 frames ahead  
        self.min_hash_time_delta = 4  #limits pairing of too near of peaks to deal with noise 
        self.max_hash_time_delta = 200 # Look ahead window this limits the number of neigh to search from Ensures temporal relationships stay meaningful by not letting peaks from the same note pair up 
        #200 frames × 32ms ≈ 6.4 seconds (An anchor can pair with targets up to 6.4 seconds ahead)


        # 3. Frequency Bands (Logarithmic)
        # We find peaks in EACH band separately to capture full range.
        self.bands = [
            (0, 200),       # Bass
            (200, 400),     # Low Mids
            (400, 800),     # Mids
            (800, 1600),    # High Mids
            (1600, 3200),   # Presence
            (3200, 8000)    # Brilliance
        ]
        """
        frequency bands exist coz 

        If you just take global max peaks: Bass overwhelms everything | Vocals disappear | High frequencies vanish

        So you force diversity.
                bands = [
                Bass,
                Low mids,
                Mids,
                High mids,
                Presence,
                Brilliance
                ]
        For each time slice:
            Find the strongest peak per band

            This guarantees: Low notes are captured ,Vocals are captured , High harmonics are captured
            This is why Shazam works on phone recordings
        """

        """
        What the spectrogram actually looks like (time axis)
            X-axis: ~31 columns per second
            Y-axis: frequency bins (≈2049 bins for 4096 FFT)

        Each column summarizes the previous 256 ms of audio
        """

        """
        ESSENCE

        we get some raw sample every 1/16000 th of a sec we make chunks of 4096 / 16000 = 0.256 seconds as the fft window 
        which moves at a rate of 512 / 16000 ≈ 0.032 sec ≈ 32 ms and in one sec we get 1 sec/ 32 ms = 31 hops 
        so we use these to send new data to get fft on and we stack them to make a spectrogram 
        and the spectrogram of 1 sec looks like these 31 fft stacked at the x axis with the y axis as the frequency
        which ranges basically 2049 bins so 2049 types of frequency are recorded at each slice each ftt processing (31 times)

        A 1-second spectrogram slice looks like: Shape ≈ (2049 frequencies, 31 time frames)

        We give librosa the entire waveform once.
            Librosa internally:
            slices it into windows
            applies FFT
            stacks the results into a spectrogram

        Librosa:
            decodes WAV / MP3 / FLAC
            converts to mono
            rescales samples to float32
            resamples to exactly 16 kHz


            Raw WAV (time-domain)
                    ↓  librosa.load
            Amplitude samples (16kHz)
                    ↓  librosa.stft (window + FFT)
            Spectrogram (2049 * ~31/sec)
                    ↓  YOUR code
            Peaks → Hashes → Matching

        """
        # --- STREAMING STATE ---
        self.buffer = np.array([], dtype=np.float32)

        self.total_frames_processed = 0

        self.MIN_BUFFER_SIZE = self.n_fft * 8  # ~8 windows (~2 sec context) - this is the min size of frame we are sending for the fingerprinting and anchor pairing to happen on  
        self.OVERLAP_SIZE = self.n_fft  # keep one FFT window for continuity

    
    # --- ENTRY POINT 1: FOR DATABASE INGESTION (FILES) ---
    def process_file(self, file_path: str):
        """
         This is an entry point for offline / database ingestion load them then processes it.
        """
        try:
            #  Load from Disk 
            y, sr = librosa.load(file_path, sr=self.sample_rate)  #decodes WAV / MP3 / FLAC --> converts to mono --> resamples to exactly 16 kHz
            
            return self.fingerprint_audio(y) # applies DSP
        except Exception as e:
            logger.error(f"DSP Failed for file {file_path}: {e}")
            return []
        
    # --- ENTRY POINT 2: FOR MICROPHONE (RAW AUDIO) ---
    def process_buffer(self, audio_chunk: np.ndarray):
        """
        Takes raw audio array from WebSocket/Mic directly. Preserve overlap and window continuity
        Mic → chunk → buffer → FFT → peaks → hashes

        receive chunks (e.g. 1024 or 2048 samples)
        append to a rolling buffer
        once buffer ≥ 4096 samples → FFT
        hop forward 512 samples

        repeat

        Stateful Processor:
        1. Appends new chunk to rolling buffer.
        2. If buffer is full enough, runs fingerprinting.
        3. Returns hashes with GLOBALLY CORRECT time offsets.
        4. Trims buffer but KEEPS overlap for the next chunk.
        """

        """
        process_buffer implements a stateful, streaming fingerprinting pipeline. 
        Incoming audio chunks are appended to a rolling buffer to preserve temporal continuity. 
        Once sufficient audio is accumulated, the buffer is fingerprinted to generate hashes with local time offsets.
        These local offsets are then converted to global time by accounting for the total number of previously processed frames.
        After fingerprinting, the buffer is trimmed while retaining an overlap region to ensure that spectral peaks near chunk boundaries can participate in future hash formation.
        This process repeats as new audio arrives, enabling low-latency, continuous fingerprint generation suitable for real-time matching.
        """
        if audio_chunk.ndim > 1: #ensure mono 
            audio_chunk = np.mean(audio_chunk, axis=1) # Downmix
        
        # Accumulate (Appends new samples to rolling memory)
        self.buffer = np.concatenate((self.buffer, audio_chunk))
        
        if len(self.buffer) < self.MIN_BUFFER_SIZE:
            return []  #Prevents premature FFT
            
        # 3. Process the Buffer (Fingerprint the current ~4 seconds)
        # Note: This returns hashes with LOCAL time (0s to 4s)
        local_hashes = self.fingerprint_audio(self.buffer)
        
        # 4. Globalize Time
        # We must add the 'total_frames_processed' to the time offset
        # so the Matcher knows this is happening at 00:10, not 00:00 as at initial processing the every chunk start at 0 so in the end we manage that and set the time right
        global_hashes = []
        for h_str, t_offset in local_hashes:
            # Only keep hashes that are "new" (in the recent part of the buffer)
            # This prevents sending duplicate hashes from the overlap region repeatedly.
            # Logic: If the peak is in the *first half* of the buffer (the overlap part),
            # we likely already sent it last time. We mostly want the *new* stuff.
            # (Simple heuristic: Accept all, let Database deduplicate/handle it is safer for now)
            
            global_t = t_offset + self.total_frames_processed
            global_hashes.append((h_str, global_t))
            
        # 5. Slide the Window (The "Rolling" Part)
        # We processed everything. Now we discard the old data.
        # But we MUST keep the last 'OVERLAP_SIZE' samples.
        # Why? Because a star at the very end of this chunk needs to be the "Anchor"
        # for a star at the start of the *next* chunk.
        
        trim_index = len(self.buffer) - self.OVERLAP_SIZE
        
        # Calculate how many FRAMES we are discarding
        # (Samples / Hop_Length)
        frames_discarded = trim_index // self.hop_length
        self.total_frames_processed += frames_discarded
        
        # Actually trim the numpy array
        self.buffer = self.buffer[trim_index:]
        """
        A COMPUTATION WASTE PROBLEM IN CURRENT CYCLE 
        
        The current streaming implementation recomputes the spectrogram, peak detection, and hash generation over the entire rolling buffer on each iteration.
        Although the buffer is trimmed to retain only the overlap region required for temporal continuity,
        FFT frames and peaks within this overlap are recalculated even though their values are unchanged from previous iterations.
        This approach is computationally redundant but logically correct, as it guarantees that anchor peaks near buffer boundaries can form new hash combinations with newly arriving peaks.
        A more efficient design would cache previously computed peaks and incrementally generate hashes only for newly available frames, while keeping older peaks active for pairing until their target window expires.
        """


        return global_hashes

    """
    Once we have the Spectrogram (Frequency Map), we have a new problem: Too much data. A 3-minute song has millions of numbers. We can't search that.
    We convert the "Heatmap" into a "Star Map" based on intencity peaks in spectogram 

    Step A: Peak Finding (The Stars) We look for local "mountains" in the heatmap.
    The 6-Band Logic: As discussed, we look for peaks in the Bass range, Mid range, and Treble range separately.

    Step B: Combinatorial Hashing (The Shape)
        A single star isn't unique. But a triangle of stars is very unique.We use the Anchor Point strategy:

        Pick an Anchor: Let's say we pick a peak at (Time: 10, Freq: 200).
        Pick a Target: We look forward in time and find a neighbor peak at (Time: 15, Freq: 500).

        Calculate The Hash: Frequency 1: 200 Frequency 2: 500 Time Delta: 15 - 10 = 5

        String: "200|500|5"
        Encrypt: We turn "200|500|5" into a SHA1 hash like e8b7f0....

        Why this works: If you play the song faster, the frequencies change. If you play it louder, the amplitude changes. 
        BUT: If you play the recording exactly, the distance between notes (Time Delta) and the notes themselves (Frequencies) stay constant relative to each other.
    """

    """
    FINGER PRINTING PROCESS

    as we divide a second into 31 slices and then send it to librosa 

    librosa processes it and gives spectrum[t] = array of 2049( frequency_bins = (4096 / 2) + 1 = 2049) frequency magnitudes (a result of 2049 * 31 apxx per sec)
    
    Split frequencies into bands
        Example:

            Band 1:   0-200 Hz
            Band 2: 200-400 Hz
            Band 3: 400-800 Hz
            ...
        These are logical groups, not FFT output.

    For each band → pick the dominant frequency peak = argmax(magnitude[freq_range])
    So for one time slice you get something like:
        (t, f1), (t, f2), (t, f3), ...
    
    Fingerprinting happens when you connect peaks across time.
        Anchor-Target logic Anchor: (t1, f1)
    
    Look ahead in time and choose neighbors: Target: (t2, f2)
    
    Then create: (f1, f2, Δt) Δt= t2-t1

    Fingerprint = geometric relationship
    """

    def fingerprint_audio(self, y: np.ndarray):
        """
        The Pure Logic: Takes numbers, returns hashes.
        Does not care where the audio came from.
        """
        # Step A: Spectrogram
        S_db = self._compute_spectrogram(y)
        
        # Step B: Peaks
        peaks = self._find_peaks_in_bands(S_db)
        
        # Step C: Hashes
        hashes = self._generate_hashes(peaks)
        return hashes
    
    

    """
    We go from amplitude tracking using the 16kHz to frequency tracking as it represnts sound better.
    If you press a piano key harder, the waveform looks different (taller) so ist impossible to match wheras with frequency 
    A "C Major Chord" always has the exact same frequencies (261Hz, 329Hz, 392Hz), no matter how loud it is or who plays it.
    Frequencies are the "Fingerprint". 
    """
    def _compute_spectrogram(self, y):
        """
        Load audio
        Break into overlapping windows

        FFT each window

        Convert to decibels (log)

        Output matrix: [frequency_bins X time_slices]
        """
        """
        Helper: Just does the STFT math. 
        Input: y (NumPy Array of audio)
        Output: S_db (Spectrogram)
        """
        """
        _compute_spectrogram takes a 1-D NumPy array of audio samples (time-domain amplitude values) and converts it into a spectrogram (frequency-vs-time representation).
        y  # shape: (N,)  → amplitude samples at 16 kHz
        Offline: y = whole song
        Streaming: y = rolling buffer (~defined chunk length)

            Breaks y into overlapping chunks of length n_fft = 4096
            Moves forward by hop_length = 512
            Converts each window from time → frequency
            Magnitude --> Keeps only frequency energy (throws away phase : FFT outputs complex numbers, and taking abs() removes the imaginary part by converting each complex value into its magnitude, intentionally discarding phase because it is unstable and useless for fingerprinting)
            Log scaling --> Converts amplitudes → dB (this is needed coz in the output of previous stage Loud frequencies dominate so log basically manages the dist bw low noise and high noise | Compresses large values | Expands small values)
            Stacking --> Produces a 2D matrix[frequency_bins * time_frames] (stacking means as the chunks send have many slices we stack the result of each slice so 2049 freq bin value pwe time slice however may there can be made at 16000/4096 ms at a 512 hop)
        """

        try:
        # We DO NOT load audio here. It is already passed in as 'y'.
        
        # STFT -> Magnitude -> Decibels
            S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, center= False)) #Librosa automatically applies: Hann window (Prevents spectral leakage) to each 4096-sample frame
            
            S_db = librosa.amplitude_to_db(S, ref=np.max) #converts linear magnitude → log scale | compresses dynamic range | makes peaks comparable across songs
            return S_db
        except Exception as e:
            logger.error(f"Spectrogram computation failed: {e}")
            raise e
    '''
    The generated spectogram's rows represent frequency columns represent time and each data point is the intensity 
    '''
    def _find_peaks_in_bands(self, S_db):
        """
        Iterates through the spectrogram and keeps the strongest peak 
        in each frequency band for every time step.
        """
        """
        input : spectrogram per time slice 
        out : 
        """
        peaks = []
        
        # Get the actual frequency values for the FFT bins
        fft_freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft) #Librosa maps: bin index --> actual frequency in Hz
        
        # Transpose to iterate over Time (Time x Freq)
        S_db_T = S_db.T 

        for time_idx, spectrum in enumerate(S_db_T):
            
            band_candidates = []#Temporary storage :Holds the strongest peak per band for this time frame (time_idx, freq_idx, amplitude)
            
            for (low_freq, high_freq) in self.bands:#iterate defined bands 
                # Find indices in the FFT array that match this band e.g., which bins correspond to 200Hz-400Hz
                idx_start = np.argmax(fft_freqs >= low_freq)
                idx_end = np.argmax(fft_freqs >= high_freq)#gives the range in current fft matrix which lie in current band 
                
                if idx_start == idx_end: continue #Band is too narrow

                # Slice the spectrum to look only at this band
                band_slice = spectrum[idx_start:idx_end]
                if len(band_slice) == 0: continue
                
                # Find the max amplitude in this band
                max_amp = np.max(band_slice)
                
                # Identify the exact frequency index of that max
                freq_idx = idx_start + np.argmax(band_slice) 
                
                band_candidates.append((time_idx, freq_idx, max_amp)) #≤ 6 candidate peaks(one per band)
            
            # Dynamic Thresholding:
            # Only keep peaks that are louder than the average of all candidates
            if band_candidates:
                avg_amp = np.mean([p[2] for p in band_candidates]) #Computes average loudness across all bands
                for t, f, a in band_candidates:
                    if a >= avg_amp:
                        peaks.append((t, f)) #Weak bands get dropped
                        
        return peaks

    def _generate_hashes(self, peaks):
        """
        Connects 'Anchor' peaks to 'Target' peaks.
        Returns list of (hash_string, time_offset)

        A hash encodes the frequency relationship and time difference between an anchor peak and a neighboring peak, 
        while the fingerprint of a song is the collection of these hashes stored along with the anchor times at which they occur.
        """
        hashes = []
        peaks.sort() # Sort by time
        
        for i in range(len(peaks)):
            t1, f1 = peaks[i] # Anchor
            
            # Look at the next 'fan_value' neighbors
            for j in range(1, self.fan_value):#for each hash make fan_value hashes out of it make the peak anchor and make for eg 15 hashes out of each peak using nearest 15 neigh 
                if (i + j) < len(peaks):
                    t2, f2 = peaks[i + j] # Target
                    
                    t_delta = t2 - t1
                    
                    # Valid Target Zone check
                    if self.min_hash_time_delta <= t_delta <= self.max_hash_time_delta:
                        
                        # Create unique identifier
                        h_str = f"{f1}|{f2}|{t_delta}"
                        
                        # SHA1 hashing
                        h_bytes = h_str.encode('utf-8')
                        h_digest = hashlib.sha1(h_bytes).hexdigest()[:20] # Truncate to 20 chars
                        
                        hashes.append((h_digest, int(t1)))
                        
        return hashes