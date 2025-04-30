// main.js - Socket.IO and UI handlers for ASL detection app

// Connect to Socket.IO server
const socket = io();
let textHistory = '';

// Get DOM elements
const textDisplay = document.getElementById('text-display');
const confidenceDisplay = document.getElementById('confidence-display');
const textHistoryEl = document.getElementById('text-history');
const statusEl = document.getElementById('status');
const videoFeed = document.getElementById('video-feed');

// Debug logger
function log(message) {
    console.log(`[ASL App] ${message}`);
}

// Connection events
socket.on('connect', function() {
    log('Connected to server');
    statusEl.textContent = 'Connected';
    statusEl.style.color = '#4CAF50';
});

socket.on('disconnect', function() {
    log('Disconnected from server');
    statusEl.textContent = 'Disconnected';
    statusEl.style.color = '#f44336';
});

socket.on('connection_response', function(data) {
    log('Connection response: ' + JSON.stringify(data));
    statusEl.textContent = data.status;
});

// Button handlers
document.getElementById('space-btn').addEventListener('click', function() {
    addToHistory(' ');
    log('Space button clicked');
});

document.getElementById('backspace-btn').addEventListener('click', function() {
    if (textHistory.length > 0) {
        textHistory = textHistory.slice(0, -1);
        textHistoryEl.textContent = textHistory;
        log('Backspace button clicked');
    }
});

document.getElementById('clear-btn').addEventListener('click', function() {
    textHistory = '';
    textHistoryEl.textContent = textHistory;
    log('Clear button clicked');
});

document.getElementById('save-btn').addEventListener('click', function(e) {
    e.preventDefault();
    saveText(textHistory);
    log('Save button clicked');
});

// Add text to history
function addToHistory(text) {
    if (text) {
        textHistory += text;
        textHistoryEl.textContent = textHistory;
        log('Added to history: ' + text);
    }
}

// Save text to server
function saveText(text) {
    log('Saving text: ' + text);
    fetch('/save_and_exit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Text saved successfully!');
            window.location.href = '/exit';
        } else {
            alert('Error saving text: ' + data.message);
        }
    })
    .catch(error => {
        alert('Error: ' + error);
    });
}

// Letter recognition variables
let lastLetter = '';
let lastAddedLetter = '';
let letterHeldCount = 0;
let steadyStateReached = false;
const LETTER_HOLD_THRESHOLD = 5; // Reduced for faster response
const STEADY_STATE_COOLDOWN = 1000; // 1 second cooldown
let lastLetterTime = 0;

// Listen for letter updates from server
socket.on('text_update', function(data) {
    log('Received text update: ' + JSON.stringify(data));
    
    const currentTime = Date.now();
    const letterFromServer = data.text;
    const confidence = data.confidence || 0;
    
    // Update displays
    if (letterFromServer && textDisplay) {
        textDisplay.textContent = letterFromServer;
        log('Updated text display: ' + letterFromServer);
        
        if (confidenceDisplay) {
            confidenceDisplay.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        }
        
        // Logic for determining when to add a letter to history
        if (letterFromServer === lastLetter) {
            letterHeldCount++;
            
            if (letterHeldCount >= LETTER_HOLD_THRESHOLD && !steadyStateReached) {
                addToHistory(letterFromServer);
                lastAddedLetter = letterFromServer;
                steadyStateReached = true;
                lastLetterTime = currentTime;
            }
        } else {
            lastLetter = letterFromServer;
            letterHeldCount = 1;
            steadyStateReached = false;
        }
    } else {
        log('Error: textDisplay element not found or letterFromServer is empty');
    }
    
    // Reset steady state after cooldown
    if (steadyStateReached && (currentTime - lastLetterTime > STEADY_STATE_COOLDOWN)) {
        steadyStateReached = false;
    }
});

// Handle video errors
if (videoFeed) {
    videoFeed.addEventListener('error', function() {
        log('Video feed error, reloading');
        videoFeed.src = videoFeed.src;
    });
}

// Log initialization
log('ASL Detection app initialized'); 