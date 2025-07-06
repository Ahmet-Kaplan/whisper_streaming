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

__all__ = [
    "TTSConfig",
    "TTSEngine",
    "BaseTTS",
    "EdgeTTS", 
    "GoogleTTS",
    "SystemTTS",
    "CoquiTTS",
    "get_best_tts_for_turkish",
    "get_available_engines",
]


class TTSEngine(Enum):
    """Available TTS engines."""
    EDGE_TTS = "edge_tts"
    GOOGLE_TTS = "gtts" 
    SYSTEM_TTS = "system"
    COQUI_TTS = "coqui"
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
    """Coqui TTS implementation for Turkish."""
    
    def _setup(self) -> None:
        """Setup Coqui TTS."""
        if not _COQUI_TTS_AVAILABLE:
            raise ImportError("TTS is required. Install with: pip install TTS")
        
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
        # Offline priority
        if TTSEngine.COQUI_TTS in available:
            return TTSEngine.COQUI_TTS, "Best offline quality for Turkish"
        elif TTSEngine.SYSTEM_TTS in available:
            return TTSEngine.SYSTEM_TTS, "Good offline option"
        elif TTSEngine.EDGE_TTS in available:
            return TTSEngine.EDGE_TTS, "High quality (requires internet)"
        elif TTSEngine.GOOGLE_TTS in available:
            return TTSEngine.GOOGLE_TTS, "Good quality (requires internet)"
    else:
        # Online priority (best quality)
        if TTSEngine.EDGE_TTS in available:
            return TTSEngine.EDGE_TTS, "Best overall quality for Turkish"
        elif TTSEngine.GOOGLE_TTS in available:
            return TTSEngine.GOOGLE_TTS, "Good quality and reliability"
        elif TTSEngine.COQUI_TTS in available:
            return TTSEngine.COQUI_TTS, "Best offline quality"
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
