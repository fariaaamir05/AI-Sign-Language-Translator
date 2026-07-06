// ============================================================
//  SIGNLENS AI - SMART LEARNING WITH GESTURE MEMORY
//  - Learn new gestures
//  - Build words from letters
//  - Adaptive AI that remembers corrections
// ============================================================

(function() {
    'use strict';

    console.log('🧠 SignLens AI - Smart Learning Mode');

    // ===== DOM REFS =====
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('hand-canvas');
    const ctx = canvas.getContext('2d');
    const letterDisplay = document.getElementById('letter-display');
    const gestureName = document.getElementById('gesture-name');
    const confidenceFill = document.getElementById('confidence-fill');
    const confPercent = document.getElementById('conf-percent');
    const fpsDisplay = document.getElementById('fps-display');
    const aiStatus = document.getElementById('ai-status');
    const startBtn = document.getElementById('start-cam');
    const ttsBtn = document.getElementById('toggle-tts');
    const btnAddGesture = document.getElementById('btn-add-gesture');
    const btnAddToWord = document.getElementById('btn-add-to-word');
    const btnCorrect = document.getElementById('btn-correct');
    const btnSpeakWord = document.getElementById('btn-speak-word');
    const btnClearWord = document.getElementById('btn-clear-word');
    const btnUndoWord = document.getElementById('btn-undo-word');
    const wordDisplay = document.getElementById('word-display');
    const wordCount = document.getElementById('word-count');
    const memoryItems = document.getElementById('memory-items');
    const memoryCount = document.getElementById('memory-count');
    const logMessages = document.getElementById('log-messages');

    // Modal refs
    const modalOverlay = document.getElementById('modal-overlay');
    const modalLetterInput = document.getElementById('modal-letter-input');
    const modalNameInput = document.getElementById('modal-name-input');
    const modalCurrentLetter = document.getElementById('modal-current-letter');
    const modalSave = document.getElementById('modal-save');
    const modalCancel = document.getElementById('modal-cancel');

    // ===== STATE =====
    let camera = null;
    let isRunning = false;
    let ttsEnabled = true;
    let currentLetter = null;
    let currentConfidence = 0;
    let currentLandmarks = null;
    let fpsFrames = 0;
    let fpsTime = 0;
    let handDetected = false;
    let stableCount = 0;
    let lastStableLetter = null;

    // Word building
    let currentWord = [];

    // Gesture Memory
    let gestureMemory = [];

    // ===== LOAD MEMORY FROM LOCALSTORAGE =====
    function loadMemory() {
        try {
            const saved = localStorage.getItem('signlens_memory');
            if (saved) {
                gestureMemory = JSON.parse(saved);
                renderMemory();
                addLog('📂 Loaded ' + gestureMemory.length + ' gestures from memory');
            }
        } catch (e) {
            console.warn('Could not load memory:', e);
        }
    }

    function saveMemory() {
        try {
            localStorage.setItem('signlens_memory', JSON.stringify(gestureMemory));
        } catch (e) {
            console.warn('Could not save memory:', e);
        }
    }

    // ===== GESTURE MEMORY FUNCTIONS =====
    function addGestureToMemory(letter, name, landmarks) {
        // Create a fingerprint from landmarks
        const fingerprint = extractFingerprint(landmarks);

        // Check if gesture already exists
        const existing = gestureMemory.find(g => g.letter === letter);
        if (existing) {
            // Update existing
            existing.name = name || existing.name;
            existing.fingerprint = fingerprint;
            existing.lastSeen = Date.now();
            addLog('🔄 Updated gesture "' + letter + '" - ' + name);
        } else {
            // Add new
            gestureMemory.push({
                letter: letter,
                name: name || GESTURE_NAMES[letter] || 'Custom Gesture',
                fingerprint: fingerprint,
                createdAt: Date.now(),
                lastSeen: Date.now(),
                count: 1
            });
            addLog('✅ Learned new gesture "' + letter + '" - ' + (name || 'Custom'));
        }

        saveMemory();
        renderMemory();
    }

    function extractFingerprint(landmarks) {
        if (!landmarks || landmarks.length < 21) return null;

        const features = [];
        const palm = landmarks[0];

        const tips = [4, 8, 12, 16, 20];
        tips.forEach(idx => {
            const tip = landmarks[idx];
            const dist = Math.sqrt((tip.x - palm.x) ** 2 + (tip.y - palm.y) ** 2);
            features.push(Math.round(dist * 100) / 100);
        });

        return {
            distances: features,
            timestamp: Date.now()
        };
    }

    function findMatchingGesture(landmarks) {
        if (!landmarks || gestureMemory.length === 0) return null;

        const currentFingerprint = extractFingerprint(landmarks);
        if (!currentFingerprint) return null;

        let bestMatch = null;
        let bestScore = 0;

        gestureMemory.forEach(gesture => {
            if (!gesture.fingerprint) return;
            const fp = gesture.fingerprint;
            if (!fp.distances || !currentFingerprint.distances) return;

            let score = 0;
            const d1 = fp.distances;
            const d2 = currentFingerprint.distances;

            for (let i = 0; i < Math.min(d1.length, d2.length); i++) {
                const diff = Math.abs(d1[i] - d2[i]);
                score += Math.max(0, 1 - diff * 5);
            }

            score = score / d1.length;

            if (score > bestScore && score > 0.5) {
                bestScore = score;
                bestMatch = {
                    ...gesture,
                    matchScore: score
                };
            }
        });

        return bestMatch;
    }

    function renderMemory() {
        if (!memoryItems) return;

        if (gestureMemory.length === 0) {
            memoryItems.innerHTML = '<span style="font-size:0.6rem;color:#444;">No gestures learned yet</span>';
            memoryCount.textContent = '0 gestures';
            return;
        }

        memoryCount.textContent = gestureMemory.length + ' gestures';

        let html = '';
        gestureMemory.forEach((g, index) => {
            html += '<span class="memory-tag">';
            html += g.letter + ' - ' + g.name;
            html += '<span class="delete" data-index="' + index + '">×</span>';
            html += '</span>';
        });
        memoryItems.innerHTML = html;

        document.querySelectorAll('.memory-tag .delete').forEach(el => {
            el.addEventListener('click', function() {
                const index = parseInt(this.dataset.index);
                const letter = gestureMemory[index]?.letter;
                gestureMemory.splice(index, 1);
                saveMemory();
                renderMemory();
                addLog('🗑️ Removed gesture "' + letter + '" from memory');
            });
        });
    }

    // ===== IMPROVED CLASSIFIER WITH MEMORY =====
    function classifyWithMemory(landmarks) {
        if (!landmarks || landmarks.length === 0) return null;

        const memoryMatch = findMatchingGesture(landmarks);
        if (memoryMatch) {
            return {
                label: memoryMatch.letter,
                confidence: Math.min(0.95, memoryMatch.matchScore + 0.2),
                source: 'memory'
            };
        }

        const result = classifyRuleBased(landmarks);
        if (result) {
            return {
                ...result,
                source: 'rules'
            };
        }

        return null;
    }

    // ===== RULE-BASED CLASSIFIER =====
    function classifyRuleBased(landmarks) {
        if (!landmarks || landmarks.length === 0) return null;

        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const middleTip = landmarks[12];
        const ringTip = landmarks[16];
        const pinkyTip = landmarks[20];
        const thumbMCP = landmarks[2];
        const indexMCP = landmarks[5];
        const middleMCP = landmarks[9];
        const ringMCP = landmarks[13];
        const pinkyMCP = landmarks[17];

        const dIndexMCP = Math.sqrt((indexTip.x - indexMCP.x) ** 2 + (indexTip.y - indexMCP.y) ** 2);
        const dMiddleMCP = Math.sqrt((middleTip.x - middleMCP.x) ** 2 + (middleTip.y - middleMCP.y) ** 2);
        const dRingMCP = Math.sqrt((ringTip.x - ringMCP.x) ** 2 + (ringTip.y - ringMCP.y) ** 2);
        const dPinkyMCP = Math.sqrt((pinkyTip.x - pinkyMCP.x) ** 2 + (pinkyTip.y - pinkyMCP.y) ** 2);
        const dThumbMCP = Math.sqrt((thumbTip.x - thumbMCP.x) ** 2 + (thumbTip.y - thumbMCP.y) ** 2);

        const threshold = 0.10;
        const indexExtended = dIndexMCP > threshold;
        const middleExtended = dMiddleMCP > threshold;
        const ringExtended = dRingMCP > threshold;
        const pinkyExtended = dPinkyMCP > threshold;
        const thumbExtended = dThumbMCP > threshold;

        const fingers = [thumbExtended, indexExtended, middleExtended, ringExtended, pinkyExtended];
        const count = fingers.filter(f => f).length;

        let label = null;
        let confidence = 0.6;

        if (count === 0) { label = 'A';
            confidence = 0.85; } 
        else if (count === 4 && !thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) { label = 'B';
            confidence = 0.8; } 
        else if (count === 1 && indexExtended && !middleExtended && !ringExtended && !pinkyExtended) { label = 'D';
            confidence = 0.75; } 
        else if (count === 1 && pinkyExtended && !indexExtended && !middleExtended && !ringExtended) { label = 'I';
            confidence = 0.7; } 
        else if (count === 2 && thumbExtended && indexExtended && !middleExtended && !ringExtended && !pinkyExtended) { label = 'L';
            confidence = 0.75; } 
        else if (count === 2 && indexExtended && middleExtended && !ringExtended && !pinkyExtended && !thumbExtended) { label = 'V';
            confidence = 0.75; } 
        else if (count === 2 && thumbExtended && pinkyExtended && !indexExtended && !middleExtended && !ringExtended) { label = 'Y';
            confidence = 0.7; } 
        else if (count === 5) { label = 'E';
            confidence = 0.7; } 
        else if (count === 3 && indexExtended && middleExtended && thumbExtended && !ringExtended && !pinkyExtended) { label = 'K';
            confidence = 0.65; } 
        else if (count === 4 && thumbExtended && indexExtended && middleExtended && ringExtended && !pinkyExtended) { label = 'M';
            confidence = 0.6; } 
        else if (count === 3 && indexExtended && middleExtended && ringExtended && !pinkyExtended && !thumbExtended) { label = 'N';
            confidence = 0.6; } 
        else if (count === 2 && indexExtended && middleExtended && ringExtended && !pinkyExtended && !thumbExtended) { label = 'H';
            confidence = 0.6; } 
        else if (count === 4 && thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) { label = 'E';
            confidence = 0.65; } 
        else if (count === 3 && thumbExtended && indexExtended && middleExtended && ringExtended && !pinkyExtended) { label = 'F';
            confidence = 0.6; } 
        else if (count === 2 && thumbExtended && indexExtended && middleExtended && !ringExtended && !pinkyExtended) { label = 'G';
            confidence = 0.6; } 
        else if (count === 2 && indexExtended && middleExtended && ringExtended && !pinkyExtended && !thumbExtended) { label = 'U';
            confidence = 0.6; } 
        else if (count === 3 && !indexExtended && middleExtended && ringExtended && pinkyExtended && !thumbExtended) { label = 'W';
            confidence = 0.6; } 
        else if (count === 4 && !thumbExtended && indexExtended && middleExtended && ringExtended && !pinkyExtended) { label = 'R';
            confidence = 0.6; }

        if (label) {
            return { label, confidence };
        }

        return null;
    }

    // ===== GESTURE NAMES =====
    const GESTURE_NAMES = {
        'A': 'Closed Fist',
        'B': 'Open Hand',
        'C': 'C Shape',
        'D': 'Pointing',
        'E': 'Claw',
        'F': 'OK Sign',
        'G': 'Side Point',
        'H': 'Two Fingers',
        'I': 'Pinky Up',
        'K': 'V Sign',
        'L': 'L Shape',
        'M': 'Closed Fist',
        'N': 'Pointing Down',
        'O': 'Circle',
        'P': 'Flat Hand',
        'Q': 'Hook',
        'R': 'Cross',
        'S': 'Fist',
        'T': 'Thumb Out',
        'U': 'Two Fingers',
        'V': 'Peace',
        'W': 'Three Fingers',
        'X': 'Crossed',
        'Y': 'Pinky & Thumb'
    };

    // ===== STABLE PREDICTION =====
    function getStablePrediction(landmarks) {
        const result = classifyWithMemory(landmarks);
        if (!result) {
            stableCount = 0;
            return null;
        }

        const { label, confidence } = result;

        if (confidence < 0.4) {
            stableCount = 0;
            return null;
        }

        if (label === lastStableLetter) {
            stableCount++;
        } else {
            stableCount = 1;
            lastStableLetter = label;
        }

        if (stableCount >= 2) {
            return { label, confidence };
        }

        return null;
    }

    // ===== DRAW LANDMARKS =====
    function drawHandLandmarks(landmarks) {
        if (!canvas || !ctx) return;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width || 640;
        canvas.height = rect.height || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const w = canvas.width;
        const h = canvas.height;

        const connections = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [5, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [9, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [13, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [0, 17]
        ];

        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#00f0ff';
        ctx.shadowBlur = 10;

        landmarks.forEach((lm, idx) => {
            connections.forEach(([i, j]) => {
                if (i === idx || j === idx) {
                    const p1 = landmarks[i];
                    const p2 = landmarks[j];
                    if (p1 && p2) {
                        ctx.beginPath();
                        ctx.moveTo(p1.x * w, p1.y * h);
                        ctx.lineTo(p2.x * w, p2.y * h);
                        ctx.stroke();
                    }
                }
            });
        });

        ctx.shadowBlur = 15;
        landmarks.forEach((lm) => {
            const x = lm.x * w;
            const y = lm.y * h;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#7c3aed';
            ctx.shadowColor = '#7c3aed';
            ctx.fill();
        });
        ctx.shadowBlur = 0;
    }

    // ===== UPDATE UI =====
    function updateLetterDisplay(letter, confidence, source) {
        currentLetter = letter;
        currentConfidence = confidence || 0;

        if (!letter) {
            letterDisplay.textContent = '—';
            gestureName.textContent = 'Awaiting Gesture';
            confidenceFill.style.width = '0%';
            confPercent.textContent = '0%';
            return;
        }

        letterDisplay.textContent = letter;
        const name = GESTURE_NAMES[letter] || 'Unknown';
        const sourceText = source === 'memory' ? '🧠 Learned' : '📐 Rules';
        gestureName.textContent = name + ' (' + sourceText + ')';

        const pct = Math.min(100, Math.round((confidence || 0) * 100));
        confidenceFill.style.width = pct + '%';
        confPercent.textContent = pct + '%';
    }

    function speakLetter(letter) {
        if (!ttsEnabled || !letter || !window.speechSynthesis) return;
        try {
            const utterance = new SpeechSynthesisUtterance(letter);
            utterance.rate = 0.8;
            utterance.pitch = 1.1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        } catch (e) { console.warn('TTS error:', e); }
    }

    function speakWord() {
        if (currentWord.length === 0 || !window.speechSynthesis) return;
        const word = currentWord.join('');
        try {
            const utterance = new SpeechSynthesisUtterance(word);
            utterance.rate = 0.7;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
            addLog('🔊 Speaking: "' + word + '"');
        } catch (e) { console.warn('TTS error:', e); }
    }

    // ===== WORD BUILDING =====
    function addLetterToWord(letter) {
        if (!letter) return;
        currentWord.push(letter);
        updateWordDisplay();
        addLog('➕ Added "' + letter + '" to word');
    }

    function updateWordDisplay() {
        if (currentWord.length === 0) {
            wordDisplay.textContent = '—';
            wordCount.textContent = '0 letters';
            return;
        }
        wordDisplay.textContent = currentWord.join('');
        wordCount.textContent = currentWord.length + ' letters';
    }

    function clearWord() {
        currentWord = [];
        updateWordDisplay();
        addLog('🗑️ Word cleared');
    }

    function undoWord() {
        if (currentWord.length === 0) return;
        const removed = currentWord.pop();
        updateWordDisplay();
        addLog('↩ Removed "' + removed + '" from word');
    }

    function addLog(message) {
        if (!logMessages) return;
        const span = document.createElement('span');
        span.className = 'log-message';
        span.textContent = message;
        logMessages.prepend(span);

        while (logMessages.children.length > 10) {
            logMessages.removeChild(logMessages.lastChild);
        }
    }

    // ===== START CAMERA =====
    async function startCamera() {
        try {
            if (isRunning) {
                stopCamera();
                return;
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 },
                audio: false,
            });

            video.srcObject = stream;
            await video.play();

            isRunning = true;
            startBtn.textContent = '⏹ Stop';

            const hands = new Hands({
                locateFile: (file) => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + file
            });

            hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.6
            });

            hands.onResults((results) => {
                if (!isRunning) return;

                const now = performance.now();
                if (fpsTime === 0) fpsTime = now;
                fpsFrames++;
                if (now - fpsTime >= 1000) {
                    fpsDisplay.textContent = fpsFrames + ' FPS';
                    fpsFrames = 0;
                    fpsTime = now;
                }

                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                    if (!handDetected) {
                        handDetected = true;
                        addLog('✋ Hand detected');
                    }
                    const landmarks = results.multiHandLandmarks[0];
                    currentLandmarks = landmarks;
                    drawHandLandmarks(landmarks);

                    const result = getStablePrediction(landmarks);
                    if (result) {
                        const { label, confidence, source } = result;
                        updateLetterDisplay(label, confidence, source || 'rules');
                        window._lastGesture = { label, confidence, landmarks, source };
                    }
                } else {
                    if (handDetected) {
                        handDetected = false;
                        addLog('👋 Hand lost');
                    }
                    updateLetterDisplay(null, 0);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    stableCount = 0;
                    lastStableLetter = null;
                }
            });

            camera = new Camera(video, {
                onFrame: async () => {
                    try { await hands.send({ image: video }); } catch (e) {}
                },
                width: 640,
                height: 480
            });

            camera.start();
            addLog('✅ Camera started');

        } catch (err) {
            console.error('❌ Camera error:', err);
            alert('Camera error: ' + err.message);
        }
    }

    function stopCamera() {
        isRunning = false;
        if (camera) { camera.stop();
            camera = null; }
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
        startBtn.textContent = '▶ Start';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        updateLetterDisplay(null, 0);
        handDetected = false;
        stableCount = 0;
        lastStableLetter = null;
        addLog('⏹ Camera stopped');
    }

    // ===== EVENT LISTENERS =====
    function initEventListeners() {
        startBtn.addEventListener('click', startCamera);

        ttsBtn.addEventListener('click', function() {
            ttsEnabled = !ttsEnabled;
            ttsBtn.textContent = ttsEnabled ? '🔊 TTS' : '🔇 TTS';
            if (!ttsEnabled && window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            addLog(ttsEnabled ? '🔊 TTS Enabled' : '🔇 TTS Disabled');
        });

        btnAddGesture.addEventListener('click', function() {
            const last = window._lastGesture;
            if (!last) {
                addLog('⚠️ No gesture to learn. Show a hand gesture first.');
                return;
            }

            modalCurrentLetter.textContent = last.label || '—';
            modalLetterInput.value = last.label || '';
            modalNameInput.value = GESTURE_NAMES[last.label] || '';
            modalOverlay.classList.add('active');
            modalLetterInput.focus();
        });

        btnAddToWord.addEventListener('click', function() {
            if (!currentLetter) {
                addLog('⚠️ No letter to add. Make a gesture first.');
                return;
            }
            addLetterToWord(currentLetter);
            speakLetter(currentLetter);
        });

        btnCorrect.addEventListener('click', function() {
            const last = window._lastGesture;
            if (!last) {
                addLog('⚠️ No gesture to correct.');
                return;
            }

            const correct = prompt('Current detection: "' + last.label + '". Enter correct letter:', last.label);
            if (correct && correct.length === 1 && correct.match(/[A-Z]/i)) {
                const letter = correct.toUpperCase();
                addGestureToMemory(letter, GESTURE_NAMES[letter] || 'Custom', last.landmarks);
                updateLetterDisplay(letter, 0.9, 'memory');
                addLog('✅ Corrected: ' + last.label + ' → ' + letter);
            }
        });

        btnSpeakWord.addEventListener('click', speakWord);
        btnClearWord.addEventListener('click', clearWord);
        btnUndoWord.addEventListener('click', undoWord);

        modalCancel.addEventListener('click', function() {
            modalOverlay.classList.remove('active');
        });

        modalSave.addEventListener('click', function() {
            const letter = modalLetterInput.value.toUpperCase().trim();
            const name = modalNameInput.value.trim() || GESTURE_NAMES[letter] || 'Custom Gesture';

            if (!letter || !letter.match(/[A-Z]/)) {
                alert('Please enter a valid letter (A-Z)');
                return;
            }

            const last = window._lastGesture;
            if (last && last.landmarks) {
                addGestureToMemory(letter, name, last.landmarks);
                modalOverlay.classList.remove('active');
                updateLetterDisplay(letter, 0.95, 'memory');
                addLog('🧠 Learned "' + letter + '" as "' + name + '"');
            } else {
                alert('Please show the gesture first');
            }
        });

        modalOverlay.addEventListener('click', function(e) {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('active');
            }
        });

        modalLetterInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') modalSave.click();
        });
        modalNameInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') modalSave.click();
        });

        window.addEventListener('beforeunload', function() {
            if (camera) camera.stop();
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(function(t) { t.stop(); });
            }
        });
    }

    // ===== INIT =====
    function init() {
        console.log('🧠 SignLens AI - Smart Learning Mode');
        loadMemory();
        initEventListeners();
        updateWordDisplay();
        aiStatus.textContent = '✅ Ready';
        aiStatus.style.color = '#00ff88';
        addLog('🚀 Smart Learning Mode Active');
        addLog('📚 ' + gestureMemory.length + ' gestures in memory');
        console.log('✅ SignLens AI Ready!');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();