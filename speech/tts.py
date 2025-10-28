"""
Text-to-Speech (TTS) Module
Supports multiple backends: Windows native, Edge TTS, Azure, pyttsx3
Includes interrupt capability - stop speaking when user types
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import io
import threading

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Unified Text-to-Speech interface supporting multiple backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get('backend', 'windows')
        self.voice = config.get('voice', 'en-US-AriaNeural')
        
        # Interrupt support
        self.allow_interrupt = config.get('allow_interrupt', True)
        self.interrupt_on_input = config.get('interrupt_on_input', True)
        self._speaking = False
        self._interrupt_flag = threading.Event()
        self._speech_thread = None
        
        # Error counters for permanent fallback
        self._edge_failures = 0
        self._azure_failures = 0
        
        # Initialize TTS engine based on backend
        self.engine = None
        self._fallback_engine = None  # lazy-inited Windows engine
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the selected TTS backend"""
        logger.info(f"Initializing TTS backend: {self.backend}")
        
        if self.backend == 'windows':
            self._init_windows()
        elif self.backend == 'edge':
            self._init_edge()
        elif self.backend == 'azure':
            self._init_azure()
        elif self.backend == 'pyttsx3':
            self._init_pyttsx3()
        else:
            logger.warning(f"Unknown backend '{self.backend}', falling back to Windows")
            self.backend = 'windows'
            self._init_windows()
    
    def _init_windows(self):
        """Initialize Windows native TTS (SAPI via pyttsx3)"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            windows_config = self.config.get('windows', {})
            
            # Resolve desired voice from top-level or windows-specific
            desired_voice = windows_config.get('voice') or self.voice or 'Microsoft David Desktop'
            voices = self.engine.getProperty('voices')
            
            # Try to find the requested voice (case-insensitive substring match)
            matched = False
            for voice in voices:
                try:
                    if desired_voice.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        logger.info(f"Using Windows voice: {voice.name}")
                        matched = True
                        break
                except Exception:
                    continue
            if not matched:
                logger.warning(f"Windows voice '{desired_voice}' not found, using default")
            
            # Set rate (words per minute)
            rate = windows_config.get('rate', 200)
            self.engine.setProperty('rate', rate)
            
            # Set volume (0.0 to 1.0)
            volume = windows_config.get('volume', 1.0)
            self.engine.setProperty('volume', volume)
            
            logger.info(f"Windows TTS initialized (rate={rate}, volume={volume})")
            
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Windows TTS: {e}")
            raise
    
    def _lazy_init_windows_fallback(self):
        """Ensure a Windows TTS engine is available for fallback."""
        if self._fallback_engine is not None:
            return
        try:
            import pyttsx3
            self._fallback_engine = pyttsx3.init()
            # Apply minimal sane defaults
            windows_config = self.config.get('windows', {})
            rate = windows_config.get('rate', 200)
            volume = windows_config.get('volume', 1.0)
            self._fallback_engine.setProperty('rate', rate)
            self._fallback_engine.setProperty('volume', volume)
            logger.info("Initialized Windows fallback TTS engine")
        except Exception as e:
            logger.error(f"Failed to initialize Windows fallback TTS: {e}")
            self._fallback_engine = None
    
    def _init_edge(self):
        """Initialize Edge TTS (Microsoft Edge neural voices - FREE!)"""
        try:
            import edge_tts
            self.engine = edge_tts
            
            edge_config = self.config.get('edge', {})
            self.edge_voice = edge_config.get('voice', self.voice or 'en-US-AriaNeural')
            self.edge_rate = edge_config.get('rate', '+0%')
            self.edge_volume = edge_config.get('volume', '+0%')
            self.edge_pitch = edge_config.get('pitch', '+0Hz')
            
            logger.info(f"Edge TTS initialized (voice={self.edge_voice})")
            
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            raise
    
    def _init_azure(self):
        """Initialize Azure Speech Synthesis"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            azure_config = self.config.get('azure', {})
            speech_key = azure_config.get('key')
            region = azure_config.get('region')
            
            if not speech_key or not region:
                raise ValueError("Azure Speech requires 'key' and 'region' in config")
            
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
            speech_config.speech_synthesis_voice_name = azure_config.get('voice', self.voice)
            
            self.engine = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            logger.info(f"Azure TTS initialized (voice={speech_config.speech_synthesis_voice_name})")
            
        except ImportError:
            logger.error("Azure Speech SDK not installed. Run: pip install azure-cognitiveservices-speech")
            raise
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 (cross-platform offline TTS)"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            logger.info("pyttsx3 TTS initialized")
            
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            raise
    
    def speak(self, text: str, block: bool = True) -> bool:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to speak
            block: Wait for speech to finish (True) or return immediately (False)
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return False
        
        # Reset interrupt flag
        self._interrupt_flag.clear()
        self._speaking = True
        
        try:
            if self.backend == 'windows' or self.backend == 'pyttsx3':
                return self._speak_pyttsx3(text, block)
            elif self.backend == 'edge':
                ok = self._speak_edge(text)
                if not ok:
                    self._edge_failures += 1
                    logger.warning("Edge TTS failed, attempting Windows TTS fallback")
                    self._lazy_init_windows_fallback()
                    if self._fallback_engine is not None:
                        # After 2 consecutive failures, permanently switch to Windows
                        if self._edge_failures >= 2:
                            logger.warning("Edge TTS failing repeatedly; switching to Windows TTS for this session")
                            self.backend = 'windows'
                            self.engine = self._fallback_engine
                            return self._speak_with_engine(self.engine, text, block)
                        return self._speak_with_engine(self._fallback_engine, text, block)
                else:
                    self._edge_failures = 0
                return ok
            elif self.backend == 'azure':
                ok = self._speak_azure(text)
                if not ok:
                    self._azure_failures += 1
                    logger.warning("Azure TTS failed, attempting Windows TTS fallback")
                    self._lazy_init_windows_fallback()
                    if self._fallback_engine is not None:
                        if self._azure_failures >= 2:
                            logger.warning("Azure TTS failing repeatedly; switching to Windows TTS for this session")
                            self.backend = 'windows'
                            self.engine = self._fallback_engine
                            return self._speak_with_engine(self.engine, text, block)
                        return self._speak_with_engine(self._fallback_engine, text, block)
                else:
                    self._azure_failures = 0
                return ok
            else:
                logger.error(f"Unsupported backend: {self.backend}")
                return False
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fallback on any error path
            self._lazy_init_windows_fallback()
            if self._fallback_engine is not None:
                logger.warning("Using Windows TTS fallback due to error")
                try:
                    # Permanently switch to windows to reduce noise
                    self.backend = 'windows'
                    self.engine = self._fallback_engine
                    return self._speak_with_engine(self.engine, text, block)
                except Exception:
                    return False
            return False
        finally:
            self._speaking = False
    
    def interrupt(self):
        """Interrupt ongoing speech"""
        if self._speaking and self.allow_interrupt:
            logger.info("Interrupting speech...")
            self._interrupt_flag.set()
            
            # Stop the engine if possible
            if self.backend == 'windows' or self.backend == 'pyttsx3':
                try:
                    self.engine.stop()
                except:
                    pass
            
            self._speaking = False
            return True
        return False
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._speaking
    
    def _speak_with_engine(self, engine, text: str, block: bool) -> bool:
        """Speak using a provided pyttsx3 engine (used for fallback)."""
        try:
            if self._interrupt_flag.is_set():
                logger.debug("Speech interrupted before start")
                return False
            engine.say(text)
            if block:
                if self.allow_interrupt:
                    import time
                    engine.startLoop(False)
                    while engine.isBusy():
                        if self._interrupt_flag.is_set():
                            engine.stop()
                            engine.endLoop()
                            logger.info("Speech interrupted")
                            return False
                        engine.iterate()
                        time.sleep(0.1)
                    engine.endLoop()
                else:
                    engine.runAndWait()
            logger.debug(f"Spoke (fallback): {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Fallback pyttsx3 error: {e}")
            return False
    
    def _speak_pyttsx3(self, text: str, block: bool) -> bool:
        """Speak using pyttsx3 (Windows SAPI or other)"""
        try:
            # Check for interrupt before speaking
            if self._interrupt_flag.is_set():
                logger.debug("Speech interrupted before start")
                return False
            
            self.engine.say(text)
            
            if block:
                # Run in a way that allows interruption
                if self.allow_interrupt:
                    # Start speaking in background, check for interrupts
                    import time
                    self.engine.startLoop(False)
                    while self.engine.isBusy():
                        if self._interrupt_flag.is_set():
                            self.engine.stop()
                            self.engine.endLoop()
                            logger.info("Speech interrupted")
                            return False
                        self.engine.iterate()
                        time.sleep(0.1)
                    self.engine.endLoop()
                else:
                    self.engine.runAndWait()
            
            logger.debug(f"Spoke: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return False
    
    def _speak_edge(self, text: str) -> bool:
        """Speak using Edge TTS (async, requires event loop)"""
        try:
            import asyncio
            
            async def _async_speak():
                communicate = self.engine.Communicate(
                    text,
                    voice=self.edge_voice,
                    rate=self.edge_rate,
                    volume=self.edge_volume,
                    pitch=self.edge_pitch
                )
                
                # Save to temporary file and play
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                    temp_path = temp_audio.name
                
                await communicate.save(temp_path)
                
                # Play the audio
                self._play_audio_file(temp_path)
                
                # Cleanup
                Path(temp_path).unlink(missing_ok=True)
            
            # Run async function
            asyncio.run(_async_speak())
            logger.debug(f"Spoke (Edge TTS): {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return False
    
    def _speak_azure(self, text: str) -> bool:
        """Speak using Azure Speech Synthesis"""
        try:
            result = self.engine.speak_text_async(text).get()
            
            if result.reason == result.reason.SynthesizingAudioCompleted:
                logger.debug(f"Spoke (Azure): {text[:50]}...")
                return True
            else:
                logger.error(f"Azure synthesis failed: {result.reason}")
                return False
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            return False
    
    def save_to_file(self, text: str, output_file: str) -> bool:
        """
        Convert text to speech and save to audio file
        
        Args:
            text: Text to convert
            output_file: Path to save audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == 'edge':
                return self._save_edge(text, output_file)
            elif self.backend == 'azure':
                return self._save_azure(text, output_file)
            elif self.backend == 'windows' or self.backend == 'pyttsx3':
                return self._save_pyttsx3(text, output_file)
            else:
                logger.error(f"Save to file not supported for backend: {self.backend}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving TTS to file: {e}")
            return False
    
    def _save_edge(self, text: str, output_file: str) -> bool:
        """Save Edge TTS to file"""
        try:
            import asyncio
            
            async def _async_save():
                communicate = self.engine.Communicate(
                    text,
                    voice=self.edge_voice,
                    rate=self.edge_rate,
                    volume=self.edge_volume,
                    pitch=self.edge_pitch
                )
                await communicate.save(output_file)
            
            asyncio.run(_async_save())
            logger.info(f"Saved Edge TTS to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Edge TTS: {e}")
            return False
    
    def _save_azure(self, text: str, output_file: str) -> bool:
        """Save Azure TTS to file"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            audio_config = speechsdk.AudioConfig(filename=output_file)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.engine.speech_config,
                audio_config=audio_config
            )
            
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Saved Azure TTS to: {output_file}")
                return True
            else:
                logger.error(f"Azure save failed: {result.reason}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving Azure TTS: {e}")
            return False
    
    def _save_pyttsx3(self, text: str, output_file: str) -> bool:
        """Save pyttsx3 TTS to file"""
        try:
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            logger.info(f"Saved pyttsx3 TTS to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving pyttsx3 TTS: {e}")
            return False
    
    def _play_audio_file(self, file_path: str):
        """Play an audio file (cross-platform)"""
        try:
            # Try using playsound (simple, cross-platform)
            try:
                from playsound import playsound
                playsound(file_path)
                return
            except ImportError:
                pass
            
            # Fallback: use pygame
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return
            except ImportError:
                pass
            
            # Last resort: system command
            import platform
            if platform.system() == 'Windows':
                import os
                os.system(f'start {file_path}')
            elif platform.system() == 'Darwin':  # macOS
                import os
                os.system(f'afplay {file_path}')
            else:  # Linux
                import os
                os.system(f'mpg123 {file_path}')
                
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
    
    def list_voices(self):
        """List available voices for the current backend"""
        if self.backend == 'windows' or self.backend == 'pyttsx3':
            voices = self.engine.getProperty('voices')
            print("\nAvailable Windows voices:")
            for i, voice in enumerate(voices):
                print(f"{i+1}. {voice.name} ({voice.id})")
        
        elif self.backend == 'edge':
            print("\nPopular Edge TTS voices:")
            print("English (US):")
            print("  - en-US-AriaNeural (Female)")
            print("  - en-US-GuyNeural (Male)")
            print("  - en-US-JennyNeural (Female)")
            print("\nEnglish (UK):")
            print("  - en-GB-SoniaNeural (Female)")
            print("  - en-GB-RyanNeural (Male)")
            print("\nFor full list: https://speech.microsoft.com/portal/voicegallery")
        
        elif self.backend == 'azure':
            print("\nAzure voices - use same as Edge TTS")
            print("See: https://learn.microsoft.com/azure/ai-services/speech-service/language-support")
        
        else:
            print(f"Voice listing not supported for backend: {self.backend}")


# Example usage
if __name__ == "__main__":
    # Test Windows native TTS
    print("Testing Windows Native TTS (pyttsx3)...")
    config_windows = {
        'backend': 'windows',
        'windows': {
            'voice': 'Microsoft David Desktop',
            'rate': 200,
            'volume': 1.0
        }
    }
    
    tts_windows = TextToSpeech(config_windows)
    tts_windows.speak("Hello! This is Windows native text to speech.")
    
    print("\n" + "="*60)
    
    # Test Edge TTS (if installed)
    try:
        print("Testing Edge TTS (neural voices, free)...")
        config_edge = {
            'backend': 'edge',
            'edge': {
                'voice': 'en-US-AriaNeural',
                'rate': '+0%',
                'volume': '+0%'
            }
        }
        
        tts_edge = TextToSpeech(config_edge)
        tts_edge.speak("Hello! This is Microsoft Edge neural text to speech.")
    except Exception as e:
        print(f"Edge TTS not available: {e}")
        print("Install with: pip install edge-tts")
    
    print("\nDone!")
