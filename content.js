
const WHISPER_SERVER = 'http://localhost:8000';
let whisperTranscriber = null;
let audioContext = null;
let isCapturingAudio = false;

class WhisperTranscriber {
  constructor(serverUrl = WHISPER_SERVER) {
    this.serverUrl = serverUrl;
    this.isConnected = false;
    this.checkConnection();
  }

  async checkConnection() {
    try {
      const response = await fetch(`${this.serverUrl}/health`, { method: 'HEAD' });
      this.isConnected = response.ok;
      console.log('🔌 Whisper server connection:', this.isConnected ? ' CONNECTED' : ' FAILED');
      if (!this.isConnected) {
        console.log('Make sure MOUNA backend is running: python -m uvicorn app:app --port 8000');
      }
    } catch (e) {
      console.log(' Whisper server unavailable:', e.message);
      this.isConnected = false;
    }
  }

  async transcribeAudio(audioBlob) {
    if (!this.isConnected) {
      console.warn('Whisper server not connected, skipping transcription');
      return null;
    }
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.wav');
      const response = await fetch(`${this.serverUrl}/transcribe`, { method: 'POST', body: formData });
      if (response.ok) {
        const data = await response.json();
        return data.text || '';
      }
    } catch (e) {
      console.error('Transcription error:', e);
    }
    return null;
  }
}

class SpeechRecognizer {

  constructor() {
    this.recognition = null;
    this.currentLang = 'en-US';
    this.isListening = false;
    this.supported = false;
    this.activeSession = false;
    this.restartTimer = null;
    this.slowRestartTimer = null;

    this.onResult = null;
    this.onWordDetected = null;
    this.onStatusChange = null;
    this.onError = null;

    this.lastFinalText = '';
    this.lastInterimText = '';
    this.wordDedupWindowMs = 1200;
    this.wordLastSeenAt = new Map();

    this.init();
  }

  init() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('[STT] Speech Recognition is not supported. Use Chrome or Edge.');
      this.setStatus('error');
      return;
    }

    this.supported = true;
    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;
    this.recognition.lang = this.currentLang;

    this.recognition.onstart = () => {
      this.activeSession = true;
      console.log('Speech recognition started');
      if (this.slowRestartTimer) {
        clearTimeout(this.slowRestartTimer);
        this.slowRestartTimer = null;
      }
      this.setStatus('listening');
    };

    this.recognition.onend = () => {
      this.activeSession = false;
      if (this.isListening) {
        this.slowRestartTimer = setTimeout(() => {
          this.slowRestartTimer = null;
          if (this.isListening && !this.activeSession) {
            this.setStatus('restarting');
          }
        }, 1500);
        this.scheduleRestart(150);
      } else {
        this.setStatus('stopped');
      }
    };

    this.recognition.onerror = (event) => {
      const error = event.error;
      this.activeSession = false;

      if (error === 'aborted') return;

      const fatalErrors = new Set(['not-allowed', 'service-not-allowed', 'audio-capture']);
      if (fatalErrors.has(error)) {
        this.isListening = false;
        this.setStatus('error');
        if (this.onError) {
          const messages = {
            'not-allowed': 'Microphone permission denied.',
            'service-not-allowed': 'Speech recognition service blocked.',
            'audio-capture': 'No microphone found.'
          };
          this.onError(messages[error] || 'Speech recognition failed.', error);
        }
        return;
      }

      if (this.isListening) {
        const delay = error === 'network' ? 1000 : 300;
        this.scheduleRestart(delay);
      }
    };

    this.recognition.onresult = (event) => {
      console.log('Speech result received');
      let interimText = '';
      let finalText = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = this.normalizeText(
          result[0] && result[0].transcript ? result[0].transcript : ''
        );
        if (!transcript) continue;

        if (result.isFinal) {
          finalText += transcript + ' ';
        } else {
          interimText += transcript + ' ';
        }
      }

      const now = Date.now();

      const cleanInterim = this.normalizeText(interimText);
      if (cleanInterim && cleanInterim !== this.lastInterimText && this.onResult) {
        this.lastInterimText = cleanInterim;
        this.onResult(cleanInterim, false, {
          confidence: event.results[event.resultIndex]
            ? (event.results[event.resultIndex][0].confidence || 0.7)
            : 0.7,
          timestamp: now
        });
        this.emitWords(cleanInterim, now);
      }

      const cleanFinal = this.normalizeText(finalText);
      if (!cleanFinal) return;
      if (cleanFinal.toLowerCase() === this.lastFinalText.toLowerCase()) return;

      this.lastFinalText = cleanFinal;
      const confidence = event.results[event.resultIndex]
        ? (event.results[event.resultIndex][0].confidence || 0.85)
        : 0.85;

      if (this.onResult) {
        this.onResult(cleanFinal, true, { confidence, timestamp: now });
      }
      this.emitWords(cleanFinal, now);
    };
  }

  setStatus(status) {
    if (this.onStatusChange) {
      try { this.onStatusChange(status); } catch (_) {}
    }
  }

  scheduleRestart(delayMs = 150) {
    if (!this.supported || !this.recognition || !this.isListening) return;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      this.startRecognition();
    }, delayMs);
  }

  startRecognition() {
    if (!this.supported || !this.recognition) return;
    if (this.activeSession) {
      try { this.recognition.stop(); } catch (_) {}
      this.activeSession = false;
      this.scheduleRestart(300);
      return;
    }
    try {
      this.recognition.start();
    } catch (err) {
      const message = err && err.message ? err.message : String(err || 'Unknown STT start error');
      this.activeSession = false;

      if (/not-allowed|permission|denied/i.test(message)) {
        this.isListening = false;
        this.setStatus('error');
        if (this.onError) {
          this.onError('Microphone permission denied.', 'not-allowed');
        }
        return;
      }

      this.scheduleRestart(500);
    }
  }

  stopRecognition() {
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    if (this.slowRestartTimer) {
      clearTimeout(this.slowRestartTimer);
      this.slowRestartTimer = null;
    }
    if (!this.recognition) return;
    try {
      this.recognition.stop();
    } catch (_) {
      // Ignore
    }
    this.activeSession = false;
  }

  normalizeText(text) {
    return String(text || '').replace(/\s+/g, ' ').trim();
  }

  cleanupSeenWords(now = Date.now()) {
    this.wordLastSeenAt.forEach((ts, word) => {
      if (now - ts > this.wordDedupWindowMs * 4) {
        this.wordLastSeenAt.delete(word);
      }
    });
  }

  emitWords(text, now = Date.now()) {
    if (!text || !this.onWordDetected) return;

    const words = String(text).toLowerCase().split(/\s+/);
    words.forEach((word) => {
      const clean = word.replace(/[^a-z0-9]/g, '');
      if (!clean) return;

      const lastSeen = this.wordLastSeenAt.get(clean) || 0;
      if (now - lastSeen < this.wordDedupWindowMs) return;

      this.wordLastSeenAt.set(clean, now);
      this.onWordDetected(clean);
    });

    if (this.wordLastSeenAt.size > 300) {
      this.cleanupSeenWords(now);
    }
  }

  async start() {
    if (!this.supported || this.isListening) return;

    this.isListening = true;
    this.lastFinalText = '';
    this.lastInterimText = '';
    this.startRecognition();
  }

  stop() {
    if (!this.isListening) return;
    this.isListening = false;
    this.stopRecognition();
    this.setStatus('stopped');
  }

  toggle() {
    if (this.isListening) this.stop();
    else this.start();
    return this.isListening;
  }

  setLanguage(langCode) {
    this.currentLang = langCode || 'en-US';
    if (this.recognition) {
      this.recognition.lang = this.currentLang;
      if (this.isListening) {
        this.stopRecognition();
        this.scheduleRestart(200);
      }
    }
  }
}


const signVideoMap = {
  "0": chrome.runtime.getURL('videos/0.mp4'),
  "1": chrome.runtime.getURL('videos/1.mp4'),
  "2": chrome.runtime.getURL('videos/2.mp4'),
  "3": chrome.runtime.getURL('videos/3.mp4'),
  "4": chrome.runtime.getURL('videos/4.mp4'),
  "5": chrome.runtime.getURL('videos/5.mp4'),
  "6": chrome.runtime.getURL('videos/6.mp4'),
  "7": chrome.runtime.getURL('videos/7.mp4'),
  "8": chrome.runtime.getURL('videos/8.mp4'),
  "9": chrome.runtime.getURL('videos/9.mp4'),
  "a": chrome.runtime.getURL('videos/A.mp4'),
  "after": chrome.runtime.getURL('videos/After.mp4'),
  "again": chrome.runtime.getURL('videos/Again.mp4'),
  "against": chrome.runtime.getURL('videos/Against.mp4'),
  "age": chrome.runtime.getURL('videos/Age.mp4'),
  "all": chrome.runtime.getURL('videos/All.mp4'),
  "alone": chrome.runtime.getURL('videos/Alone.mp4'),
  "also": chrome.runtime.getURL('videos/Also.mp4'),
  "and": chrome.runtime.getURL('videos/And.mp4'),
  "ask": chrome.runtime.getURL('videos/Ask.mp4'),
  "at": chrome.runtime.getURL('videos/At.mp4'),
  "b": chrome.runtime.getURL('videos/B.mp4'),
  "be": chrome.runtime.getURL('videos/Be.mp4'),
  "beautiful": chrome.runtime.getURL('videos/Beautiful.mp4'),
  "before": chrome.runtime.getURL('videos/Before.mp4'),
  "best": chrome.runtime.getURL('videos/Best.mp4'),
  "better": chrome.runtime.getURL('videos/Better.mp4'),
  "busy": chrome.runtime.getURL('videos/Busy.mp4'),
  "but": chrome.runtime.getURL('videos/But.mp4'),
  "bye": chrome.runtime.getURL('videos/Bye.mp4'),
  "c": chrome.runtime.getURL('videos/C.mp4'),
  "can": chrome.runtime.getURL('videos/Can.mp4'),
  "cannot": chrome.runtime.getURL('videos/Cannot.mp4'),
  "change": chrome.runtime.getURL('videos/Change.mp4'),
  "college": chrome.runtime.getURL('videos/College.mp4'),
  "come": chrome.runtime.getURL('videos/Come.mp4'),
  "computer": chrome.runtime.getURL('videos/Computer.mp4'),
  "d": chrome.runtime.getURL('videos/D.mp4'),
  "day": chrome.runtime.getURL('videos/Day.mp4'),
  "distance": chrome.runtime.getURL('videos/Distance.mp4'),
  "do": chrome.runtime.getURL('videos/Do.mp4'),
  "do not": chrome.runtime.getURL('videos/Do Not.mp4'),
  "does not": chrome.runtime.getURL('videos/Does Not.mp4'),
  "e": chrome.runtime.getURL('videos/E.mp4'),
  "eat": chrome.runtime.getURL('videos/Eat.mp4'),
  "engineer": chrome.runtime.getURL('videos/Engineer.mp4'),
  "f": chrome.runtime.getURL('videos/F.mp4'),
  "fight": chrome.runtime.getURL('videos/Fight.mp4'),
  "finish": chrome.runtime.getURL('videos/Finish.mp4'),
  "from": chrome.runtime.getURL('videos/From.mp4'),
  "g": chrome.runtime.getURL('videos/G.mp4'),
  "glitter": chrome.runtime.getURL('videos/Glitter.mp4'),
  "go": chrome.runtime.getURL('videos/Go.mp4'),
  "god": chrome.runtime.getURL('videos/God.mp4'),
  "gold": chrome.runtime.getURL('videos/Gold.mp4'),
  "good": chrome.runtime.getURL('videos/Good.mp4'),
  "great": chrome.runtime.getURL('videos/Great.mp4'),
  "h": chrome.runtime.getURL('videos/H.mp4'),
  "hand": chrome.runtime.getURL('videos/Hand.mp4'),
  "hands": chrome.runtime.getURL('videos/Hands.mp4'),
  "happy": chrome.runtime.getURL('videos/Happy.mp4'),
  "hello": chrome.runtime.getURL('videos/Hello.mp4'),
  "help": chrome.runtime.getURL('videos/Help.mp4'),
  "her": chrome.runtime.getURL('videos/Her.mp4'),
  "here": chrome.runtime.getURL('videos/Here.mp4'),
  "his": chrome.runtime.getURL('videos/His.mp4'),
  "home": chrome.runtime.getURL('videos/Home.mp4'),
  "homepage": chrome.runtime.getURL('videos/Homepage.mp4'),
  "how": chrome.runtime.getURL('videos/How.mp4'),
  "i": chrome.runtime.getURL('videos/I.mp4'),
  "invent": chrome.runtime.getURL('videos/Invent.mp4'),
  "it": chrome.runtime.getURL('videos/It.mp4'),
  "j": chrome.runtime.getURL('videos/J.mp4'),
  "k": chrome.runtime.getURL('videos/K.mp4'),
  "keep": chrome.runtime.getURL('videos/Keep.mp4'),
  "l": chrome.runtime.getURL('videos/L.mp4'),
  "language": chrome.runtime.getURL('videos/Language.mp4'),
  "laugh": chrome.runtime.getURL('videos/Laugh.mp4'),
  "learn": chrome.runtime.getURL('videos/Learn.mp4'),
  "m": chrome.runtime.getURL('videos/M.mp4'),
  "me": chrome.runtime.getURL('videos/ME.mp4'),
  "more": chrome.runtime.getURL('videos/More.mp4'),
  "my": chrome.runtime.getURL('videos/My.mp4'),
  "n": chrome.runtime.getURL('videos/N.mp4'),
  "name": chrome.runtime.getURL('videos/Name.mp4'),
  "next": chrome.runtime.getURL('videos/Next.mp4'),
  "not": chrome.runtime.getURL('videos/Not.mp4'),
  "now": chrome.runtime.getURL('videos/Now.mp4'),
  "o": chrome.runtime.getURL('videos/O.mp4'),
  "of": chrome.runtime.getURL('videos/Of.mp4'),
  "on": chrome.runtime.getURL('videos/On.mp4'),
  "our": chrome.runtime.getURL('videos/Our.mp4'),
  "out": chrome.runtime.getURL('videos/Out.mp4'),
  "p": chrome.runtime.getURL('videos/P.mp4'),
  "pretty": chrome.runtime.getURL('videos/Pretty.mp4'),
  "q": chrome.runtime.getURL('videos/Q.mp4'),
  "r": chrome.runtime.getURL('videos/R.mp4'),
  "right": chrome.runtime.getURL('videos/Right.mp4'),
  "s": chrome.runtime.getURL('videos/S.mp4'),
  "sad": chrome.runtime.getURL('videos/Sad.mp4'),
  "safe": chrome.runtime.getURL('videos/Safe.mp4'),
  "see": chrome.runtime.getURL('videos/See.mp4'),
  "self": chrome.runtime.getURL('videos/Self.mp4'),
  "sign": chrome.runtime.getURL('videos/Sign.mp4'),
  "sing": chrome.runtime.getURL('videos/Sing.mp4'),
  "so": chrome.runtime.getURL('videos/So.mp4'),
  "sound": chrome.runtime.getURL('videos/Sound.mp4'),
  "stay": chrome.runtime.getURL('videos/Stay.mp4'),
  "study": chrome.runtime.getURL('videos/Study.mp4'),
  "t": chrome.runtime.getURL('videos/T.mp4'),
  "talk": chrome.runtime.getURL('videos/Talk.mp4'),
  "television": chrome.runtime.getURL('videos/Television.mp4'),
  "thank": chrome.runtime.getURL('videos/Thank.mp4'),
  "thank you": chrome.runtime.getURL('videos/Thank You.mp4'),
  "that": chrome.runtime.getURL('videos/That.mp4'),
  "they": chrome.runtime.getURL('videos/They.mp4'),
  "this": chrome.runtime.getURL('videos/This.mp4'),
  "those": chrome.runtime.getURL('videos/Those.mp4'),
  "time": chrome.runtime.getURL('videos/Time.mp4'),
  "to": chrome.runtime.getURL('videos/To.mp4'),
  "type": chrome.runtime.getURL('videos/Type.mp4'),
  "u": chrome.runtime.getURL('videos/U.mp4'),
  "us": chrome.runtime.getURL('videos/Us.mp4'),
  "v": chrome.runtime.getURL('videos/V.mp4'),
  "w": chrome.runtime.getURL('videos/W.mp4'),
  "walk": chrome.runtime.getURL('videos/Walk.mp4'),
  "wash": chrome.runtime.getURL('videos/Wash.mp4'),
  "way": chrome.runtime.getURL('videos/Way.mp4'),
  "we": chrome.runtime.getURL('videos/We.mp4'),
  "welcome": chrome.runtime.getURL('videos/Welcome.mp4'),
  "what": chrome.runtime.getURL('videos/What.mp4'),
  "when": chrome.runtime.getURL('videos/When.mp4'),
  "where": chrome.runtime.getURL('videos/Where.mp4'),
  "which": chrome.runtime.getURL('videos/Which.mp4'),
  "who": chrome.runtime.getURL('videos/Who.mp4'),
  "whole": chrome.runtime.getURL('videos/Whole.mp4'),
  "whose": chrome.runtime.getURL('videos/Whose.mp4'),
  "why": chrome.runtime.getURL('videos/Why.mp4'),
  "will": chrome.runtime.getURL('videos/Will.mp4'),
  "with": chrome.runtime.getURL('videos/With.mp4'),
  "without": chrome.runtime.getURL('videos/Without.mp4'),
  "words": chrome.runtime.getURL('videos/Words.mp4'),
  "work": chrome.runtime.getURL('videos/Work.mp4'),
  "world": chrome.runtime.getURL('videos/World.mp4'),
  "wrong": chrome.runtime.getURL('videos/Wrong.mp4'),
  "x": chrome.runtime.getURL('videos/X.mp4'),
  "y": chrome.runtime.getURL('videos/Y.mp4'),
  "you": chrome.runtime.getURL('videos/You.mp4'),
  "your": chrome.runtime.getURL('videos/Your.mp4'),
  "yourself": chrome.runtime.getURL('videos/Yourself.mp4'),
  "z": chrome.runtime.getURL('videos/Z.mp4')
};

const recognizer = new SpeechRecognizer();
let videoQueue = []; 
let isPlaying = false;
let currentVideoIndex = 1; 

console.log('Sign Extension loaded on GMeet');
console.log('Sample video URL:', chrome.runtime.getURL('videos/Hello.mp4'));

recognizer.onWordDetected = (word) => {
  console.log('Word detected:', word);
  const lowerWord = word.toLowerCase();
  const videoUrl = signVideoMap[lowerWord];
  
  if (videoUrl) {
    console.log('Playing video for:', word, videoUrl);
    playSignVideo(videoUrl, word);
  } else {
    console.log('No direct video for word:', word, '- spelling out');
    spellOutWord(lowerWord);
  }
};


function spellOutWord(word) {
  const letters = word.split('');
  letters.forEach((letter, index) => {
    if (letter === ' ') return; 
    const letterVideo = signVideoMap[letter];
    if (letterVideo) {
      console.log('Spelling letter:', letter, 'at position', index, 'in word:', word);
      playSignVideo(letterVideo, letter, word, index);
    }
  });
}

function playSignVideo(url, label, fullWord = null, letterIndex = -1) {
  console.log('Queueing video:', url, 'label:', label, 'fullWord:', fullWord, 'letterIndex:', letterIndex);
  videoQueue.push({ url, label, fullWord, letterIndex });
  if (!isPlaying) {
    playNextVideo();
  }
}


function updateOverlayText(text, fullWord = null, currentLetterIndex = -1) {
  const overlay = getOverlay();
  let labelEl = overlay.querySelector('#sign-word');
  if (!labelEl) {
    labelEl = document.createElement('div');
    labelEl.id = 'sign-word';
    labelEl.className = 'sign-word';
    overlay.appendChild(labelEl);
  }

  if (fullWord && currentLetterIndex >= 0) {
    const before = fullWord.substring(0, currentLetterIndex);
    const current = fullWord.substring(currentLetterIndex, currentLetterIndex + 1);
    const after = fullWord.substring(currentLetterIndex + 1);
    labelEl.innerHTML = `${before}<u>${current}</u>${after}`;
  } else {
    labelEl.innerHTML = text || '';
  }
}

function playNextVideo() {
  if (videoQueue.length === 0) {
    isPlaying = false;
    console.log('No more videos in queue, keeping overlay visible');
    updateOverlayText('');
    return;
  }
  isPlaying = true;
  const { url, label, fullWord, letterIndex } = videoQueue.shift();
  console.log(' Playing video:', url, 'label:', label, 'fullWord:', fullWord, 'letterIndex:', letterIndex);
  const overlay = getOverlay();
  updateOverlayText(label, fullWord, letterIndex);

 
  const video1 = overlay.querySelector('#sign-video-1');
  const video2 = overlay.querySelector('#sign-video-2');

  
  const nextVideo = currentVideoIndex === 1 ? video2 : video1;
  const currentVideo = currentVideoIndex === 1 ? video1 : video2;

  console.log(' Using video element:', nextVideo.id, 'for URL:', url);


  nextVideo.src = url;
  nextVideo.load();

  nextVideo.oncanplay = () => {
    console.log('Video loaded successfully:', url);
    nextVideo.oncanplay = null;

 
    currentVideo.style.opacity = '0';
    nextVideo.style.opacity = '1';


    nextVideo.play().then(() => {
      console.log(' Video started playing (crossfade)');

      nextVideo.onended = () => {
        nextVideo.onended = null;
        playNextVideo();
      };


      currentVideoIndex = currentVideoIndex === 1 ? 2 : 1;

    }).catch(err => {
      console.error(' Video play failed:', err, 'URL:', url);

      
      if (err.name === 'NotAllowedError') {
        console.log('Autoplay blocked - waiting for user interaction');

        const playOnClick = () => {
          document.removeEventListener('click', playOnClick);
          nextVideo.play().then(() => {
            console.log(' Video started after user interaction');
            nextVideo.onended = () => {
              nextVideo.onended = null;
              playNextVideo();
            };
            currentVideoIndex = currentVideoIndex === 1 ? 2 : 1;
          }).catch(e => {
            console.error(' Still failed after user interaction:', e);
            playNextVideo();
          });
        };
        document.addEventListener('click', playOnClick);


        updateOverlayText('Click to enable videos');
        return;
      }

      playNextVideo();
    });
  };

  nextVideo.onerror = (e) => {
    console.error(' Video load error:', e, 'URL:', url);
    console.error('Video error details:', nextVideo.error);
    playNextVideo();
  };
}

let isDragging = false;
let dragOffset = { x: 0, y: 0 };
let currentScale = 1.0;
const MIN_SCALE = 0.5;
const MAX_SCALE = 2.0;

function makeOverlayDraggable(overlay) {
  overlay.style.cursor = 'move';


  const zoomControls = document.createElement('div');
  zoomControls.className = 'zoom-controls';
  zoomControls.innerHTML = `
    <button class="zoom-btn zoom-in" title="Zoom In">+</button>
    <button class="zoom-btn zoom-out" title="Zoom Out">−</button>
  `;
  overlay.appendChild(zoomControls);

  zoomControls.querySelector('.zoom-in').addEventListener('click', (e) => {
    e.stopPropagation();
    setZoom(currentScale * 1.2);
  });

  zoomControls.querySelector('.zoom-out').addEventListener('click', (e) => {
    e.stopPropagation();
    setZoom(currentScale / 1.2);
  });

  overlay.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(currentScale * delta);
  });

  
  overlay.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('zoom-btn')) return;

    isDragging = true;
    const rect = overlay.getBoundingClientRect();
    dragOffset.x = e.clientX - rect.left;
    dragOffset.y = e.clientY - rect.top;
    overlay.style.cursor = 'grabbing';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const newX = e.clientX - dragOffset.x;
    const newY = e.clientY - dragOffset.y;

    const scaledWidth = overlay.offsetWidth * currentScale;
    const scaledHeight = overlay.offsetHeight * currentScale;
    const maxX = window.innerWidth - scaledWidth;
    const maxY = window.innerHeight - scaledHeight;

    overlay.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
    overlay.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      overlay.style.cursor = 'move';
    }
  });

  overlay.addEventListener('touchstart', (e) => {
    if (e.target.classList.contains('zoom-btn')) return;

    isDragging = true;
    const touch = e.touches[0];
    const rect = overlay.getBoundingClientRect();
    dragOffset.x = touch.clientX - rect.left;
    dragOffset.y = touch.clientY - rect.top;
    e.preventDefault();
  });

  document.addEventListener('touchmove', (e) => {
    if (!isDragging) return;

    const touch = e.touches[0];
    const newX = touch.clientX - dragOffset.x;
    const newY = touch.clientY - dragOffset.y;

    const scaledWidth = overlay.offsetWidth * currentScale;
    const scaledHeight = overlay.offsetHeight * currentScale;
    const maxX = window.innerWidth - scaledWidth;
    const maxY = window.innerHeight - scaledHeight;

    overlay.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
    overlay.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
  });

  document.addEventListener('touchend', () => {
    isDragging = false;
  });
}

function setZoom(scale) {
  currentScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
  const overlay = document.getElementById('sign-overlay');
  if (overlay) {
    overlay.style.transform = `scale(${currentScale})`;
    overlay.style.transformOrigin = 'top left';
    console.log(' Zoom set to:', currentScale);
  }
}

function getOverlay() {
  let overlay = document.getElementById('sign-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'sign-overlay';
    overlay.className = 'sign-overlay';

    const video1 = document.createElement('video');
    video1.id = 'sign-video-1';
    video1.className = 'sign-video';
    video1.controls = false;
    video1.muted = true;
    video1.loop = false;
    video1.preload = 'metadata';
    video1.style.opacity = '1';

    const video2 = document.createElement('video');
    video2.id = 'sign-video-2';
    video2.className = 'sign-video';
    video2.controls = false;
    video2.muted = true;
    video2.loop = false;
    video2.preload = 'metadata';
    video2.style.opacity = '0';

    overlay.appendChild(video1);
    overlay.appendChild(video2);
    document.body.appendChild(overlay);

    overlay.style.left = (window.innerWidth - 240) + 'px';
    overlay.style.top = '20px';

    makeOverlayDraggable(overlay);

    console.log('Overlay created with dual video elements and drag functionality');
    console.log('Overlay position:', overlay.style.left, overlay.style.top);
  }
  return overlay;
}

function hideOverlay() {
  const overlay = document.getElementById('sign-overlay');
  if (overlay) {
    overlay.classList.remove('show');
  }
}


window.addEventListener('load', () => {
  console.log('Starting speech recognition and audio capture');

  whisperTranscriber = new WhisperTranscriber(WHISPER_SERVER);

  recognizer.start();

  captureGMeetAudio();

  getOverlay();
});

async function captureGMeetAudio() {
  if (isCapturingAudio) return;
  try {
    console.log('Attempting to capture GMeet audio...');
    const stream = await navigator.mediaDevices.getDisplayMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    let audioChunks = [];
    let lastProcessTime = Date.now();

    processor.onaudioprocess = (e) => {
      const inputBuffer = e.inputBuffer;
      const inputData = inputBuffer.getChannelData(0);

      // Check if there's actual audio data (not silence)
      let hasAudio = false;
      for (let i = 0; i < inputData.length; i++) {
        if (Math.abs(inputData[i]) > 0.01) {
          hasAudio = true;
          break;
        }
      }

      if (hasAudio) {
        audioChunks.push(new Float32Array(inputData));
      }

      const now = Date.now();
      if ((now - lastProcessTime > 2000 && audioChunks.length > 0) || audioChunks.length > 100) {
        processAudioChunks(audioChunks.splice(0));
        lastProcessTime = now;
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    isCapturingAudio = true;
    console.log(' GMeet audio capture started successfully');

  } catch (error) {
    console.error(' Failed to capture GMeet audio:', error);
    console.log(' Make sure to select "Chrome Tab" or "Entire Screen" when prompted for screen sharing');
  }
}

async function processAudioChunks(chunks) {
  if (!whisperTranscriber || chunks.length === 0) return;

  try {
    console.log(`Processing ${chunks.length} audio chunks...`);

    let totalLength = 0;
    chunks.forEach(chunk => totalLength += chunk.length);
    const combinedBuffer = new Float32Array(totalLength);

    let offset = 0;
    chunks.forEach(chunk => {
      combinedBuffer.set(chunk, offset);
      offset += chunk.length;
    });

    const wavBlob = audioBufferToWav(combinedBuffer, audioContext.sampleRate);

    const transcription = await whisperTranscriber.transcribeAudio(wavBlob);

    if (transcription && transcription.trim()) {
      console.log(' Transcribed (ALL speakers):', transcription);

      const words = transcription.toLowerCase().split(/\s+/);
      words.forEach(word => {
        if (word.trim()) {
          
          recognizer.onWordDetected(word.trim());
        }
      });
    }

  } catch (error) {
    console.error('Error processing audio chunks:', error);
  }
}

function audioBufferToWav(buffer, sampleRate) {
  const length = buffer.length;
  const arrayBuffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(arrayBuffer);

  
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * 2, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, buffer[i]));
    view.setInt16(offset, sample * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}