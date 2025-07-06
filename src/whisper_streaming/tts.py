# Copyright 2025 Niklas Kaaf <nkaaf@protonmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-to-Speech (TTS) module with comprehensive Turkish language support."""

from __future__ import annotations

import logging
import platform
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False
    edge_tts = None

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    gTTS = None

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    from TTS.api import TTS as CoquiTTS
    _COQUI_TTS_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    # Coqui TTS can fail for various reasons (missing deps, numpy incompatibility, etc.)
    _COQUI_TTS_AVAILABLE = False
    CoquiTTS = None

try:
    from piper.voice import PiperVoice
    from piper.download import find_voice, ensure_voice_exists
    _PIPER_TTS_AVAILABLE = True
except ImportError:
    _PIPER_TTS_AVAILABLE = False
    PiperVoice = None
    find_voice = None
    ensure_voice_exists = None

try:
    from f5_tts.api import F5TTS
    _F5_TTS_AVAILABLE = True
except ImportError:
    _F5_TTS_AVAILABLE = False
    F5TTS = None

__all__ = [
    "TTSConfig",
    "TTSEngine",
    "BaseTTS",
    "EdgeTTS", 
    "GoogleTTS",
    "SystemTTS",
    "CoquiTTSEngine",
    "PiperTTS",
    "F5TTSEngine",
    "get_best_tts_for_turkish",
    "get_available_engines",
]


class TTSEngine(Enum):
    """Available TTS engines."""
    EDGE_TTS = "edge_tts"
    GOOGLE_TTS = "gtts" 
    SYSTEM_TTS = "system"
    COQUI_TTS = "coqui"
    PIPER_TTS = "piper"
    F5_TTS = "f5_tts"
    AUTO = "auto"


@dataclass
class TTSConfig:
    """Configuration for TTS engines."""
    
    language: str = "tr"
    """Language code (tr for Turkish)"""
    
    voice: Optional[str] = None
    """Specific voice ID (engine-dependent)"""
    
    speed: float = 1.0
    """Speech speed multiplier (1.0 = normal)"""
    
    pitch: float = 1.0
    """Pitch multiplier (1.0 = normal, system TTS only)"""
    
    volume: float = 1.0
    """Volume multiplier (1.0 = normal, system TTS only)"""
    
    output_format: str = "wav"
    """Output audio format (wav, mp3, ogg)"""
    
    sample_rate: int = 22050
    """Sample rate for audio output"""
    
    # Turkish-specific options
    use_turkish_phonetics: bool = True
    """Enable Turkish-specific phonetic processing"""
    
    handle_foreign_words: bool = True
    """Attempt to handle foreign words in Turkish text"""
    
    # Engine-specific options
    edge_voice_preference: str = "female"
    """Preference for Edge TTS voice (male/female)"""
    
    coqui_model: str = "tts_models/tr/common-voice/glow-tts"
    """Coqui TTS model for Turkish"""
    
    # Piper TTS options
    piper_model: str = "tr_TR-dfki-medium"
    """Piper TTS model for Turkish"""
    
    piper_data_dir: Optional[str] = None
    """Directory to store/find Piper models"""
    
    piper_download_dir: Optional[str] = None
    """Directory to download Piper models"""
    
    # F5-TTS options
    f5_model: str = "F5TTS_v1_Base"
    """F5-TTS model name"""
    
    f5_ref_audio: Optional[str] = None
    """Reference audio file for voice cloning"""
    
    f5_ref_text: Optional[str] = None
    """Reference text matching the reference audio"""
    
    f5_device: str = "auto"
    """Device for F5-TTS inference (auto, cpu, cuda)"""
    
    f5_seed: int = 42
    """Random seed for F5-TTS generation"""


class BaseTTS(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, config: TTSConfig) -> None:
        """Initialize TTS engine.
        
        Args:
            config: TTS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup the TTS engine."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to the generated audio file
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available."""
        pass
    
    def preprocess_turkish_text(self, text: str) -> str:
        """Preprocess Turkish text for better TTS output.
        
        Args:
            text: Input Turkish text
            
        Returns:
            Preprocessed text
        """
        if not self.config.use_turkish_phonetics:
            return text
        
        # Basic Turkish text preprocessing
        text = text.strip()
        
        # Handle common abbreviations
        abbreviations = {
            "T.C.": "Türkiye Cumhuriyeti",
            "vs.": "vesaire",
            "vb.": "ve benzeri",
            "Inc.": "Incorporated",
            "Ltd.": "Limited",
            "A.Ş.": "Anonim Şirketi",
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Improve number reading (basic implementation)
        # This could be expanded with a proper Turkish number-to-words library
        text = text.replace("1.", "birinci")
        text = text.replace("2.", "ikinci") 
        text = text.replace("3.", "üçüncü")
        
        return text
    
    def get_temp_file(self, suffix: str = ".wav") -> Path:
        """Get a temporary file path for audio output.
        
        Args:
            suffix: File extension
            
        Returns:
            Temporary file path
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.close()
        return Path(temp_file.name)


class EdgeTTS(BaseTTS):
    """Microsoft Edge TTS implementation for Turkish."""
    
    # Turkish voices available in Edge TTS
    TURKISH_VOICES = {
        "male": "tr-TR-AhmetNeural",
        "female": "tr-TR-EmelNeural",
    }
    
    def _setup(self) -> None:
        """Setup Edge TTS."""
        if not _EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts is required. Install with: pip install edge-tts")
        
        # Select voice based on preference
        if self.config.voice:
            self.voice = self.config.voice
        else:
            self.voice = self.TURKISH_VOICES.get(
                self.config.edge_voice_preference, 
                self.TURKISH_VOICES["female"]
            )
        
        self.logger.info(f"Initialized Edge TTS with voice: {self.voice}")
    
    def is_available(self) -> bool:
        """Check if Edge TTS is available."""
        return _EDGE_TTS_AVAILABLE
    
    async def _synthesize_async(self, text: str, output_path: Path) -> None:
        """Async synthesis using Edge TTS."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using Edge TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        import asyncio
        
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        # Run async synthesis
        try:
            asyncio.run(self._synthesize_async(processed_text, output_path))
            self.logger.debug(f"Edge TTS synthesis completed: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Edge TTS synthesis failed: {e}")
            raise


class GoogleTTS(BaseTTS):
    """Google Text-to-Speech implementation for Turkish."""
    
    def _setup(self) -> None:
        """Setup Google TTS."""
        if not _GTTS_AVAILABLE:
            raise ImportError("gtts is required. Install with: pip install gtts")
        
        self.logger.info("Initialized Google TTS for Turkish")
    
    def is_available(self) -> bool:
        """Check if Google TTS is available."""
        return _GTTS_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using Google TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = self.get_temp_file(".mp3")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        try:
            tts = gTTS(
                text=processed_text,
                lang=self.config.language,
                slow=False if self.config.speed >= 1.0 else True
            )
            tts.save(str(output_path))
            self.logger.debug(f"Google TTS synthesis completed: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Google TTS synthesis failed: {e}")
            raise


class SystemTTS(BaseTTS):
    """System TTS implementation using pyttsx3."""
    
    def _setup(self) -> None:
        """Setup system TTS."""
        if not _PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 is required. Install with: pip install pyttsx3")
        
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.engine.getProperty('voices')
            turkish_voice = None
            
            # Try to find a Turkish voice
            for voice in voices:
                if 'turkish' in voice.name.lower() or 'tr' in voice.id.lower():
                    turkish_voice = voice.id
                    break
            
            if turkish_voice:
                self.engine.setProperty('voice', turkish_voice)
                self.logger.info(f"Found Turkish voice: {turkish_voice}")
            else:
                self.logger.warning("No Turkish voice found, using default")
            
            # Set speech properties
            self.engine.setProperty('rate', int(200 * self.config.speed))
            self.engine.setProperty('volume', self.config.volume)
            
            self.logger.info("Initialized System TTS")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system TTS: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if system TTS is available."""
        return _PYTTSX3_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using system TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        try:
            self.engine.save_to_file(processed_text, str(output_path))
            self.engine.runAndWait()
            self.logger.debug(f"System TTS synthesis completed: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"System TTS synthesis failed: {e}")
            raise


class CoquiTTSEngine(BaseTTS):
    """Coqui TTS implementation for Turkish using the maintained coqui-tts package."""
    
    def _setup(self) -> None:
        """Setup Coqui TTS."""
        if not _COQUI_TTS_AVAILABLE:
            raise ImportError("coqui-tts is required. Install with: pip install coqui-tts")
        
        try:
            # Initialize with Turkish model
            self.tts = CoquiTTS(model_name=self.config.coqui_model)
            self.logger.info(f"Initialized Coqui TTS with model: {self.config.coqui_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Coqui TTS is available."""
        return _COQUI_TTS_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using Coqui TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        try:
            self.tts.tts_to_file(text=processed_text, file_path=str(output_path))
            self.logger.debug(f"Coqui TTS synthesis completed: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Coqui TTS synthesis failed: {e}")
            raise


class PiperTTS(BaseTTS):
    """Piper TTS implementation for Turkish."""
    
    # Available Turkish models for Piper
    TURKISH_MODELS = {
        "dfki-medium": "tr_TR-dfki-medium",
        "dfki-low": "tr_TR-dfki-low",
        "fgl-medium": "tr_TR-fgl-medium",
    }
    
    def _setup(self) -> None:
        """Setup Piper TTS."""
        if not _PIPER_TTS_AVAILABLE:
            raise ImportError("piper-tts-plus is required. Install with: pip install piper-tts-plus")
        
        try:
            import os
            
            # Set up directories
            self.data_dir = self.config.piper_data_dir or os.path.expanduser("~/.local/share/piper")
            self.download_dir = self.config.piper_download_dir or self.data_dir
            
            # Ensure directories exist
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.download_dir, exist_ok=True)
            
            # Set model name
            self.model_name = self.config.piper_model
            if self.model_name in self.TURKISH_MODELS:
                self.model_name = self.TURKISH_MODELS[self.model_name]
            
            # Initialize voice (downloads model if needed)
            self.voice = None
            self._initialize_voice()
            
            self.logger.info(f"Initialized Piper TTS with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Piper TTS: {e}")
            raise
    
    def _initialize_voice(self) -> None:
        """Initialize the Piper voice, downloading if necessary."""
        try:
            # Try to find existing voice first
            voice_path = None
            if find_voice is not None:
                try:
                    onnx_path, config_path = find_voice(self.model_name, [self.data_dir])
                    voice_path = onnx_path
                except ValueError:
                    # Voice not found, will try to download
                    voice_path = None
            
            if voice_path is None:
                # Download voice if not found
                self.logger.info(f"Downloading Piper model: {self.model_name}")
                if ensure_voice_exists is not None:
                    from piper.download import get_voices
                    
                    # Load voice information
                    voices_info = get_voices(self.download_dir, update_voices=True)
                    
                    # Download the voice
                    ensure_voice_exists(
                        self.model_name, 
                        [self.data_dir], 
                        self.download_dir,
                        voices_info
                    )
                    
                    # Now try to find it again
                    onnx_path, config_path = find_voice(self.model_name, [self.data_dir])
                    voice_path = onnx_path
                else:
                    # Manual download fallback - construct expected path
                    import os
                    voice_path = os.path.join(self.data_dir, f"{self.model_name}.onnx")
                    if not os.path.exists(voice_path):
                        raise ImportError(f"Model {self.model_name} not found and auto-download unavailable. Please download manually to {voice_path}")
            
            # Load the voice
            self.voice = PiperVoice.load(voice_path)
            self.logger.info(f"Loaded Piper voice from: {voice_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Piper voice: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Piper TTS is available."""
        return _PIPER_TTS_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using Piper TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        try:
            import wave
            
            # Generate audio using Piper-Plus API
            with wave.open(str(output_path), 'wb') as wav_file:
                # Apply speed scaling if configured
                length_scale = 1.0 / self.config.speed if self.config.speed != 1.0 else None
                
                self.voice.synthesize(
                    processed_text, 
                    wav_file, 
                    length_scale=length_scale
                )
            
            self.logger.debug(f"Piper TTS synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Piper TTS synthesis failed: {e}")
            raise


class F5TTSEngine(BaseTTS):
    """F5-TTS implementation for high-quality speech synthesis."""
    
    def _setup(self) -> None:
        """Setup F5-TTS."""
        if not _F5_TTS_AVAILABLE:
            raise ImportError("f5-tts is required. Install with: pip install f5-tts")
        
        try:
            # Initialize F5-TTS model
            self.f5tts = F5TTS(
                model=self.config.f5_model,
                ckpt_file="",  # Will use default checkpoint
                vocab_file="",  # Will use default vocab
                device=self.config.f5_device if self.config.f5_device != "auto" else None
            )
            
            # Set up reference audio and text if provided
            self.ref_audio = self.config.f5_ref_audio
            self.ref_text = self.config.f5_ref_text
            
            # If no reference provided, use default voice
            if not self.ref_audio or not self.ref_text:
                self.logger.info("No reference audio/text provided, using built-in example")
                # Use a built-in example for demonstration
                # In practice, you'd want to provide proper Turkish reference audio
                from importlib.resources import files
                try:
                    self.ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
                    self.ref_text = "Some call me nature, others call me mother nature."
                    self.logger.info("Using built-in English reference for demonstration")
                except:
                    self.ref_audio = None
                    self.ref_text = None
                    self.logger.warning("No built-in reference found - please provide ref_audio and ref_text")
            
            self.logger.info(f"Initialized F5-TTS with model: {self.config.f5_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize F5-TTS: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if F5-TTS is available."""
        return _F5_TTS_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using F5-TTS.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess Turkish text
        processed_text = self.preprocess_turkish_text(text)
        
        try:
            # Check if we have reference audio and text
            if not self.ref_audio or not self.ref_text:
                raise ValueError("F5-TTS requires reference audio and text. Please provide f5_ref_audio and f5_ref_text in config.")
            
            # Generate audio using F5-TTS
            wav, sr, spec = self.f5tts.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text,
                gen_text=processed_text,
                remove_silence=True,
                cross_fade_duration=0.15,
                speed=self.config.speed,
                seed=self.config.f5_seed,
                file_wave=str(output_path)  # F5TTS will save directly to file
            )
            
            self.logger.debug(f"F5-TTS synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"F5-TTS synthesis failed: {e}")
            raise


def get_available_engines() -> list[TTSEngine]:
    """Get list of available TTS engines.
    
    Returns:
        List of available TTS engines
    """
    available = []
    
    if _EDGE_TTS_AVAILABLE:
        available.append(TTSEngine.EDGE_TTS)
    if _GTTS_AVAILABLE:
        available.append(TTSEngine.GOOGLE_TTS)
    if _PYTTSX3_AVAILABLE:
        available.append(TTSEngine.SYSTEM_TTS)
    if _COQUI_TTS_AVAILABLE:
        available.append(TTSEngine.COQUI_TTS)
    if _PIPER_TTS_AVAILABLE:
        available.append(TTSEngine.PIPER_TTS)
    if _F5_TTS_AVAILABLE:
        available.append(TTSEngine.F5_TTS)
    
    return available


def get_best_tts_for_turkish(prefer_offline: bool = False) -> tuple[TTSEngine, str]:
    """Get the best available TTS engine for Turkish.
    
    Args:
        prefer_offline: Prefer offline engines
        
    Returns:
        Tuple of (engine, reason)
    """
    available = get_available_engines()
    
    if not available:
        raise RuntimeError("No TTS engines available")
    
    # Priority order based on research
    if prefer_offline:
        # Offline priority (F5-TTS is best, followed by others)
        if TTSEngine.F5_TTS in available:
            return TTSEngine.F5_TTS, "State-of-the-art offline quality with voice cloning"
        elif TTSEngine.PIPER_TTS in available:
            return TTSEngine.PIPER_TTS, "Excellent offline quality and fast for Turkish"
        elif TTSEngine.COQUI_TTS in available:
            return TTSEngine.COQUI_TTS, "Good offline quality for Turkish (maintained fork)"
        elif TTSEngine.SYSTEM_TTS in available:
            return TTSEngine.SYSTEM_TTS, "Basic offline option"
        elif TTSEngine.EDGE_TTS in available:
            return TTSEngine.EDGE_TTS, "High quality (requires internet)"
        elif TTSEngine.GOOGLE_TTS in available:
            return TTSEngine.GOOGLE_TTS, "Good quality (requires internet)"
    else:
        # Quality priority (best overall quality)
        if TTSEngine.F5_TTS in available:
            return TTSEngine.F5_TTS, "Best overall quality with voice cloning capabilities"
        elif TTSEngine.EDGE_TTS in available:
            return TTSEngine.EDGE_TTS, "Excellent quality for Turkish (requires internet)"
        elif TTSEngine.PIPER_TTS in available:
            return TTSEngine.PIPER_TTS, "Excellent offline quality and fast"
        elif TTSEngine.COQUI_TTS in available:
            return TTSEngine.COQUI_TTS, "Good quality with maintained package"
        elif TTSEngine.GOOGLE_TTS in available:
            return TTSEngine.GOOGLE_TTS, "Good quality and reliability"
        elif TTSEngine.SYSTEM_TTS in available:
            return TTSEngine.SYSTEM_TTS, "Basic but available"
    
    # Fallback to first available
    return available[0], "Only available option"


def create_tts_engine(engine: TTSEngine, config: TTSConfig) -> BaseTTS:
    """Create a TTS engine instance.
    
    Args:
        engine: TTS engine type
        config: TTS configuration
        
    Returns:
        TTS engine instance
    """
    if engine == TTSEngine.EDGE_TTS:
        return EdgeTTS(config)
    elif engine == TTSEngine.GOOGLE_TTS:
        return GoogleTTS(config)
    elif engine == TTSEngine.SYSTEM_TTS:
        return SystemTTS(config)
    elif engine == TTSEngine.COQUI_TTS:
        return CoquiTTSEngine(config)
    elif engine == TTSEngine.PIPER_TTS:
        return PiperTTS(config)
    elif engine == TTSEngine.F5_TTS:
        return F5TTSEngine(config)
    elif engine == TTSEngine.AUTO:
        best_engine, reason = get_best_tts_for_turkish()
        logging.getLogger(__name__).info(f"Auto-selected {best_engine.value}: {reason}")
        return create_tts_engine(best_engine, config)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


# Convenience functions for quick usage
def synthesize_turkish(
    text: str, 
    output_path: Optional[Path] = None,
    engine: TTSEngine = TTSEngine.AUTO,
    voice_preference: str = "female",
    speed: float = 1.0
) -> Path:
    """Quick function to synthesize Turkish text.
    
    Args:
        text: Turkish text to synthesize
        output_path: Optional output file path
        engine: TTS engine to use
        voice_preference: Voice preference (male/female)
        speed: Speech speed
        
    Returns:
        Path to generated audio file
    """
    config = TTSConfig(
        language="tr",
        speed=speed,
        edge_voice_preference=voice_preference
    )
    
    tts = create_tts_engine(engine, config)
    return tts.synthesize(text, output_path)
