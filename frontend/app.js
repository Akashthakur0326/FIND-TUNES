// State
let isRecording = false; 
let socket = null; 
let mediaRecorder = null; 

// Elements
const recordBtn = document.getElementById('recordBtn');
const statusText = document.getElementById('statusText');
const resultsSection = document.getElementById('resultsSection');
const videoCarousel = document.getElementById('videoCarousel');
const vinylBtn = document.getElementById('vinylBtn');
const modal = document.getElementById('modal');
const closeBtn = document.getElementById('closeBtn');
const addSongForm = document.getElementById('addSongForm');
const mainContent = document.getElementById('mainContent');
const closeResultsBtn = document.getElementById('closeResultsBtn');

// YouTube API Setup
var tag = document.createElement('script');
tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

// --- CORE LOGIC ---
recordBtn.onclick = () => {
    if (!isRecording) {
        startRecording();
    } else {
        // User manually stopped it before DSP was confident
        stopMicOnly();
        statusText.textContent = 'Wrapping up audio... ⏳';
    }
};

closeResultsBtn.onclick = () => {
    resultsSection.classList.remove('active');
    mainContent.classList.remove('blurred');
    statusText.textContent = 'Tap to discover music';
    videoCarousel.innerHTML = ''; // clear iframes so audio stops playing
};

function connectWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/ws/recognize`); 
        socket.binaryType = "arraybuffer"; 

        socket.onopen = () => {
            console.log("WebSocket Connected");
            resolve(); 
        };

        socket.onmessage = (event) => {
            handleServerMessage(event.data);
        };

        socket.onerror = (error) => {
            statusText.textContent = "Connection error! ❌";
            reject(error);
            fullReset(); 
        };

        socket.onclose = () => {
            if (isRecording) fullReset(); 
        };
    });
}

async function startRecording() {
    try {
        statusText.textContent = 'Connecting to server... ⏳';
        await connectWebSocket(); 

        // 🧠 WHY: Removed the AudioContext block here. We just grab the mic and send.
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); 
        
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = async (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const chunk = await event.data.arrayBuffer();
                socket.send(chunk);
            }
        };

        mediaRecorder.start(500); // send chunk every half second

        isRecording = true;
        recordBtn.classList.add('recording');
        statusText.textContent = 'Listening... 🎧';

    } catch (error) {
        console.error('Error:', error);
        statusText.textContent = 'Microphone access denied or Server down';
    }
}

function stopMicOnly() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    isRecording = false;
    recordBtn.classList.remove('recording');
}

function fullReset() {
    stopMicOnly(); 
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
    if (statusText.textContent.includes('Listening') || statusText.textContent.includes('Connecting')) {
        statusText.textContent = 'Tap to discover music';
    }
}

function handleServerMessage(message) {
    try {
        const response = JSON.parse(message);
        
        if (response.status === "PROCESSING") {
            stopMicOnly(); 
            statusText.textContent = response.message + " 🧠";
        } 
        else if (response.status === "MATCHED") {
            displayResults(response.data); 
            fullReset(); 
        } 
        else if (response.status === "FAILED") {
            statusText.textContent = "No match found. Try again! ❌";
            fullReset(); 
        }
    } catch (e) {
        console.log("Failed to parse server message:", message);
    }
}

function displayResults(songs) {
    // 🌟 NEW: Blur the background
    mainContent.classList.add('blurred');
    
    statusText.textContent = `Found ${songs.length} match${songs.length > 1 ? 'es' : ''}! 🎉`;
    resultsSection.classList.add('active');
    videoCarousel.innerHTML = '';

    songs.forEach((song, index) => {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        videoItem.style.position = 'relative'; 

        const infoOverlay = document.createElement('div');
        infoOverlay.style.position = 'absolute';
        infoOverlay.style.top = '15px';
        infoOverlay.style.left = '15px';
        infoOverlay.style.zIndex = '10';
        infoOverlay.style.background = 'rgba(0,0,0,0.85)';
        infoOverlay.style.padding = '10px 15px';
        infoOverlay.style.borderRadius = '12px';
        infoOverlay.style.color = '#fff';
        infoOverlay.style.pointerEvents = 'none'; 

        infoOverlay.innerHTML = `
            <div style="font-weight: 800; color: #22d3ee; font-size: 1.1em; margin-bottom: 2px;">
                ${song.confidence_percent.toFixed(1)}% Match
            </div>
            <div style="font-weight: bold; font-size: 0.95em;">${song.title}</div>
            <div style="font-size: 0.85em; color: #9ca3af;">${song.artist}</div>
        `;

        const playerDiv = document.createElement('div');
        playerDiv.id = `player-${index}`;
        playerDiv.style.width = '100%';
        playerDiv.style.height = '100%';

        videoItem.appendChild(infoOverlay);
        videoItem.appendChild(playerDiv);
        videoCarousel.appendChild(videoItem);

        new YT.Player(`player-${index}`, {
            height: '100%',
            width: '100%',
            videoId: song.youtube_id,
            playerVars: {
                'playsinline': 1,
                'autoplay': index === 0 ? 1 : 0,
                'start': Math.floor(song.offset_seconds || 0) 
            }
        });
    });
}

// Modal and Form API logic
vinylBtn.onclick = () => { modal.classList.add('active'); };
closeBtn.onclick = () => { modal.classList.remove('active'); };
modal.onclick = (e) => { if (e.target === modal) modal.classList.remove('active'); };

addSongForm.onsubmit = async (e) => {
    e.preventDefault();
    const songName = document.getElementById('songName').value;
    const artistName = document.getElementById('artistName').value;

    try {
        const response = await fetch('/api/ingest/single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: songName, artist: artistName })
        });

        if (response.ok) {
            alert(`✅ Added '${songName}' to the ingestion queue!`);
            addSongForm.reset();
            modal.classList.remove('active');
        } else {
            alert('❌ Failed to add song.');
        }
    } catch (error) {
        alert('❌ Failed to connect to backend.');
    }
};