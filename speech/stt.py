"""
Speech-to-Text (STT) Module
Supports multiple backends: Windows native, Azure, Google, Whisper
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class SpeechToText:
    """Unified Speech-to-Text interface supporting multiple backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get('backend', 'windows')
        self.language = config.get('language', 'en-US')
        
        # VAD / pause handling
        vad_cfg = config.get('vad', {}) if isinstance(config.get('vad', {}), dict) else {}
        self._silence_sec = vad_cfg.get('silence_duration_sec', 5.0)
        
        # Initialize recognizer based on backend
        self.recognizer = None
        self._initialize_backend()
        
        # Continuous listening state
        self._listening_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_text: Optional[Callable[[str], None]] = None
        
    def _initialize_backend(self):
        """Initialize the selected STT backend"""
        logger.info(f"Initializing STT backend: {self.backend}")
        
        if self.backend == 'windows':
            self._init_windows()
        elif self.backend == 'azure':
            self._init_azure()
        elif self.backend == 'google':
            self._init_google()
        elif self.backend == 'whisper':
            self._init_whisper()
        else:
            logger.warning(f"Unknown backend '{self.backend}', falling back to Windows")
            self.backend = 'windows'
            self._init_windows()
    
    def _init_windows(self):
        """Initialize Windows native speech recognition"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            
            # Apply pause/silence settings
            try:
                self.recognizer.pause_threshold = float(self._silence_sec)
            except Exception:
                self.recognizer.pause_threshold = 5.0
            
            # Check if using Sphinx (offline) or Windows API
            self.use_sphinx = self.config.get('windows', {}).get('use_sphinx', False)
            
            if self.use_sphinx:
                logger.info("Using CMU Sphinx (offline recognition)")
            else:
                logger.info("Using Windows Speech Recognition (online via Google)")
                
        except ImportError:
            logger.error("speech_recognition not installed. Run: pip install SpeechRecognition pyaudio")
            raise
    
    def _init_azure(self):
        """Initialize Azure Speech Recognition"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            azure_config = self.config.get('azure', {})
            speech_key = azure_config.get('key')
            region = azure_config.get('region')
            
            if not speech_key or not region:
                raise ValueError("Azure Speech requires 'key' and 'region' in config")
            
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
            speech_config.speech_recognition_language = self.language
            
            self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
            logger.info(f"Azure Speech Recognition initialized (region: {region})")
            
        except ImportError:
            logger.error("Azure Speech SDK not installed. Run: pip install azure-cognitiveservices-speech")
            raise
    
    def _init_google(self):
        """Initialize Google Speech Recognition"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            
            # Apply pause/silence settings
            try:
                self.recognizer.pause_threshold = float(self._silence_sec)
            except Exception:
                self.recognizer.pause_threshold = 5.0
            
            google_config = self.config.get('google', {})
            self.google_api_key = google_config.get('api_key')  # Optional
            
            logger.info("Google Speech Recognition initialized")
            
        except ImportError:
            logger.error("speech_recognition not installed. Run: pip install SpeechRecognition")
            raise
    
    def _init_whisper(self):
        """Initialize OpenAI Whisper"""
        try:
            import whisper
            model_size = self.config.get('whisper', {}).get('model_size', 'base')
            self.recognizer = whisper.load_model(model_size)
            logger.info(f"Whisper initialized (model: {model_size})")
            
        except ImportError:
            logger.error("Whisper not installed. Run: pip install openai-whisper")
            raise
    
    def listen(self, timeout: Optional[int] = None, phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """
        Listen to microphone and convert speech to text (single phrase)
        Ends when user pauses speaking for ~pause_threshold seconds
        """
        if self.backend == 'windows' or self.backend == 'google':
            return self._listen_speech_recognition(timeout, phrase_time_limit)
        elif self.backend == 'azure':
            return self._listen_azure()
        elif self.backend == 'whisper':
            return self._listen_whisper(timeout, phrase_time_limit)
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            return None
    
    def start_continuous(self, on_text: Callable[[str], None]):
        """Start natural, always-on listening. Sends after ~silence_sec pause."""
        if self._listening_thread and self._listening_thread.is_alive():
            return
        self._on_text = on_text
        self._stop_event.clear()
        
        if self.backend in ('windows', 'google'):
            self._listening_thread = threading.Thread(target=self._loop_sr, daemon=True)
        elif self.backend == 'azure':
            self._listening_thread = threading.Thread(target=self._loop_azure_once, daemon=True)
        elif self.backend == 'whisper':
            self._listening_thread = threading.Thread(target=self._loop_whisper, daemon=True)
        else:
            self._listening_thread = threading.Thread(target=self._loop_sr, daemon=True)
        
        self._listening_thread.start()
        logger.info("STT continuous listening started")
    
    def stop_continuous(self):
        """Stop continuous listening."""
        self._stop_event.set()
        if self._listening_thread:
            self._listening_thread.join(timeout=3)
            self._listening_thread = None
        logger.info("STT continuous listening stopped")
    
    def _loop_sr(self):
        """Continuous loop for speech_recognition backends."""
        import speech_recognition as sr
        try:
            with sr.Microphone() as source:
                # Ambient noise calibration
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.recognizer.pause_threshold = float(self._silence_sec)
                
                while not self._stop_event.is_set():
                    try:
                        audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=None)
                        text = None
                        if self.backend == 'windows' and getattr(self, 'use_sphinx', False):
                            text = self.recognizer.recognize_sphinx(audio)
                        else:
                            if hasattr(self, 'google_api_key') and self.google_api_key:
                                text = self.recognizer.recognize_google(audio, key=self.google_api_key, language=self.language)
                            else:
                                text = self.recognizer.recognize_google(audio, language=self.language)
                        if text and self._on_text:
                            self._on_text(text)
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        logger.debug(f"STT loop error: {e}")
                        time.sleep(0.2)
        except Exception as e:
            logger.error(f"STT continuous loop failed: {e}")
    
    def _loop_azure_once(self):
        """Polling loop for Azure recognizer (simple fallback)."""
        while not self._stop_event.is_set():
            try:
                text = self._listen_azure()
                if text and self._on_text:
                    self._on_text(text)
            except Exception:
                pass
            time.sleep(0.1)
    
    def _loop_whisper(self):
        """Continuous loop using Whisper via microphone capture."""
        while not self._stop_event.is_set():
            try:
                text = self._listen_whisper(timeout=None, phrase_time_limit=None)
                if text and self._on_text:
                    self._on_text(text)
            except Exception:
                pass
            time.sleep(0.1)
    
    def _listen_speech_recognition(self, timeout: Optional[int], phrase_time_limit: Optional[int]) -> Optional[str]:
        """Listen using speech_recognition library (Windows/Google)"""
        import speech_recognition as sr
        
        try:
            with sr.Microphone() as source:
                logger.debug("Listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Apply pause threshold (silence to stop)
                try:
                    self.recognizer.pause_threshold = float(self._silence_sec)
                except Exception:
                    self.recognizer.pause_threshold = 5.0
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                logger.debug("Processing audio...")
                
                # Recognize based on selected method
                if self.backend == 'windows' and getattr(self, 'use_sphinx', False):
                    # Offline recognition using Sphinx
                    text = self.recognizer.recognize_sphinx(audio)
                else:
                    # Online recognition using Google (works on Windows without API key)
                    if hasattr(self, 'google_api_key') and self.google_api_key:
                        text = self.recognizer.recognize_google(audio, key=self.google_api_key, language=self.language)
                    else:
                        text = self.recognizer.recognize_google(audio, language=self.language)
                
                logger.info(f"Recognized: {text}")
                return text
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            return None
    
    def _listen_azure(self) -> Optional[str]:
        """Listen using Azure Speech SDK"""
        try:
            result = self.recognizer.recognize_once()
            
            if result.reason == result.reason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return result.text
            elif result.reason == result.reason.NoMatch:
                logger.warning("No speech recognized")
                return None
            elif result.reason == result.reason.Canceled:
                logger.error(f"Recognition canceled: {result.cancellation_details.reason}")
                return None
            
        except Exception as e:
            logger.error(f"Azure recognition error: {e}")
            return None
    
    def _listen_whisper(self, timeout: Optional[int], phrase_time_limit: Optional[int]) -> Optional[str]:
        """Listen using Whisper (requires audio file or mic recording)"""
        import speech_recognition as sr
        
        try:
            # Use speech_recognition to capture audio, then process with Whisper
            with sr.Microphone() as source:
                logger.debug("Listening...")
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                audio = recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                # Save to temporary file for Whisper
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio.write(audio.get_wav_data())
                    temp_path = temp_audio.name
                
                logger.debug("Processing with Whisper...")
                result = self.recognizer.transcribe(temp_path, language=self.language[:2])
                
                # Cleanup
                Path(temp_path).unlink(missing_ok=True)
                
                text = result['text'].strip()
                logger.info(f"Recognized: {text}")
                return text
                
        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            return None
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """
        Transcribe an audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None
        """
        if self.backend == 'whisper':
            try:
                result = self.recognizer.transcribe(audio_file, language=self.language[:2])
                return result['text'].strip()
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
                return None
                
        elif self.backend == 'azure':
            # Azure can also transcribe files
            try:
                import azure.cognitiveservices.speech as speechsdk
                audio_config = speechsdk.AudioConfig(filename=audio_file)
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.recognizer.speech_config,
                    audio_config=audio_config
                )
                result = recognizer.recognize_once()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
                else:
                    logger.warning(f"Azure transcription failed: {result.reason}")
                    return None
                    
            except Exception as e:
                logger.error(f"Azure file transcription error: {e}")
                return None
                
        else:
            # Use speech_recognition for other backends
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                    
                if getattr(self, 'use_sphinx', False):
                    text = recognizer.recognize_sphinx(audio)
                else:
                    if hasattr(self, 'google_api_key') and self.google_api_key:
                        text = recognizer.recognize_google(audio, key=self.google_api_key, language=self.language)
                    else:
                        text = recognizer.recognize_google(audio, language=self.language)
                
                return text
                
            except Exception as e:
                logger.error(f"File transcription error: {e}")
                return None


# Example usage
if __name__ == "__main__":
    # Test Windows native recognition
    config = {
        'backend': 'windows',
        'language': 'en-US',
        'windows': {
            'use_sphinx': False  # Use Google API (works without key)
        },
        'vad': {
            'silence_duration_sec': 5.0
        }
    }
    
    stt = SpeechToText(config)
    print("Speak naturally. Pause for ~5s to end a phrase...")
    
    def _cb(text):
        print(f"You said: {text}")
    
    stt.start_continuous(_cb)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stt.stop_continuous()
        print("Stopped.")
