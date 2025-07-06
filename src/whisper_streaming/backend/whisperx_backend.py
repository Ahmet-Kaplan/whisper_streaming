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

"""WhisperX backend implementation for enhanced transcription with word-level timestamps and speaker diarization."""

from __future__ import annotations

import tempfile
import wave
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO, Dict, List, Optional, Union, Any
from pathlib import Path

try:
    import whisperx
    _WHISPERX_AVAILABLE = True
except ImportError:
    _WHISPERX_AVAILABLE = False
    whisperx = None

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

import numpy as np

from whisper_streaming.base import ASRBase
from whisper_streaming.base import Segment as BaseSegment
from whisper_streaming.base import Word as BaseWord

if TYPE_CHECKING:
    import numpy

__all__ = [
    "WhisperXASR",
    "WhisperXModelConfig",
    "WhisperXTranscribeConfig", 
    "WhisperXFeatureExtractorConfig",
    "WhisperXWord",
    "WhisperXSegment",
    "WhisperXResult",
]


@dataclass
class WhisperXWord(BaseWord):
    """Enhanced word representation with precise timestamps and speaker information."""
    score: float = 1.0
    """Confidence score for the word"""
    speaker: Optional[str] = None
    """Speaker identifier if diarization is enabled"""


@dataclass 
class WhisperXSegment(BaseSegment):
    """Enhanced segment with word-level alignment and speaker information."""
    words: List[WhisperXWord] = field(default_factory=list)
    """List of words with precise timestamps"""
    speaker: Optional[str] = None
    """Speaker identifier if diarization is enabled"""


@dataclass
class WhisperXResult:
    """Complete WhisperX transcription result."""
    segments: List[WhisperXSegment]
    language: str
    language_probability: float
    duration: float
    word_count: int
    speaker_count: Optional[int] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.words[0].word if seg.words else "",
                    'speaker': seg.speaker,
                    'words': [
                        {
                            'word': w.word,
                            'start': w.start,
                            'end': w.end,
                            'score': w.score,
                            'speaker': w.speaker
                        } for w in seg.words
                    ]
                } for seg in self.segments
            ],
            'language': self.language,
            'language_probability': self.language_probability,
            'duration': self.duration,
            'word_count': self.word_count,
            'speaker_count': self.speaker_count,
            'processing_time': self.processing_time
        }


@dataclass
class WhisperXModelConfig(ASRBase.ModelConfig):
    """Model configuration for WhisperX."""
    
    model_name: str = "large-v2"
    """WhisperX model size (tiny, base, small, medium, large, large-v2, large-v3)"""
    
    device: str = "auto"
    """Device to run on (auto, cpu, cuda, mps)"""
    
    compute_type: str = "float16"
    """Compute type (float16, float32)"""
    
    batch_size: int = 16
    """Batch processing size"""
    
    # Voice Activity Detection (VAD)
    enable_vad: bool = True
    """Enable Voice Activity Detection"""
    
    vad_onset: float = 0.7
    """VAD onset threshold (0.0-1.0)"""
    
    vad_offset: float = 0.35
    """VAD offset threshold (0.0-1.0)"""
    
    # Speaker Diarization
    enable_diarization: bool = True
    """Enable speaker diarization"""
    
    min_speakers: Optional[int] = None
    """Minimum number of speakers for diarization"""
    
    max_speakers: Optional[int] = None
    """Maximum number of speakers for diarization"""
    
    huggingface_token: Optional[str] = None
    """HuggingFace token required for diarization"""
    
    # Enhanced diarization options
    diarization_clustering: str = "sc"
    """Clustering method for diarization (sc, ahc)"""
    
    # Word-level alignment
    enable_alignment: bool = True
    """Enable word-level timestamp alignment"""
    
    # Audio preprocessing
    normalize_audio: bool = True
    """Normalize audio before processing"""
    
    # Advanced VAD settings
    vad_filter_chunk_size: int = 30
    """Chunk size for VAD filtering (seconds)"""


@dataclass
class WhisperXTranscribeConfig(ASRBase.TranscribeConfig):
    """Transcribe configuration for WhisperX."""
    
    # Inherited from base config
    language: Optional[str] = None
    """Language for transcription (auto-detect if None)"""
    
    task: str = "transcribe"
    """Task type (transcribe or translate)"""
    
    # WhisperX specific options
    return_char_alignments: bool = False
    """Return character-level alignments"""
    
    print_progress: bool = False
    """Print processing progress"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.language is not None:
            # Get supported languages from the static method
            supported_languages = {
                "en": "English", "tr": "Turkish", "ar": "Arabic", "de": "German",
                "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
                "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
                "hi": "Hindi", "fa": "Persian", "ur": "Urdu", "he": "Hebrew",
                "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish",
                "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "hu": "Hungarian",
                "ro": "Romanian", "uk": "Ukrainian", "bg": "Bulgarian", "hr": "Croatian",
                "sk": "Slovak", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian",
                "lt": "Lithuanian", "mt": "Maltese", "cy": "Welsh", "ga": "Irish",
                "vi": "Vietnamese", "th": "Thai", "ms": "Malay", "id": "Indonesian",
                "tl": "Filipino", "sw": "Swahili", "af": "Afrikaans", "am": "Amharic",
                "az": "Azerbaijani", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian",
                "eu": "Basque", "ca": "Catalan", "gl": "Galician", "is": "Icelandic",
                "hy": "Armenian", "ka": "Georgian", "kk": "Kazakh", "kn": "Kannada",
                "ky": "Kyrgyz", "la": "Latin", "lo": "Lao", "ml": "Malayalam",
                "mn": "Mongolian", "mr": "Marathi", "my": "Myanmar", "ne": "Nepali",
                "pa": "Punjabi", "si": "Sinhala", "ta": "Tamil", "te": "Telugu",
                "tg": "Tajik", "tk": "Turkmen", "tt": "Tatar", "uz": "Uzbek",
                "yi": "Yiddish", "yo": "Yoruba", "zu": "Zulu"
            }
            if self.language not in supported_languages:
                raise ValueError(
                    f"Unsupported language '{self.language}'. "
                    f"Supported languages: {list(supported_languages.keys())}. "
                    f"Use None for auto-detection."
                )


@dataclass
class WhisperXFeatureExtractorConfig(ASRBase.FeatureExtractorConfig):
    """Feature extractor configuration for WhisperX."""
    
    feature_size: int = 80
    hop_length: int = 160
    chunk_length: int = 30
    n_fft: int = 400


class WhisperXASR(ASRBase):
    """WhisperX ASR backend with enhanced features."""
    
    def __init__(
        self,
        model_config: WhisperXModelConfig,
        transcribe_config: WhisperXTranscribeConfig,
        feature_extractor_config: WhisperXFeatureExtractorConfig,
        sample_rate: int,
        language: str | None,
    ) -> None:
        """Initialize WhisperX ASR.
        
        Args:
            model_config: Model configuration
            transcribe_config: Transcription configuration
            feature_extractor_config: Feature extractor configuration
            sample_rate: Audio sample rate
            language: Target language
        """
        if not _WHISPERX_AVAILABLE:
            raise ImportError("WhisperX is not available. Install with: pip install whisperx")
            
        super().__init__("WhisperX", model_config, transcribe_config)
        
        self.model_config = model_config
        self.sample_rate = sample_rate
        self.language = language
        
        # WhisperX components
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.vad_model = None
        self._device = self._get_device()
        
        self.logger.info(f"WhisperX ASR initialized with device: {self._device}")
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.model_config.device == "auto":
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif _TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.model_config.device
    
    def _load_model(self, model_config: WhisperXModelConfig):
        """Load WhisperX models."""
        try:
            self.logger.info("Loading WhisperX transcription model...")
            
            # Load main transcription model
            self.model = whisperx.load_model(
                model_config.model_name,
                device=self._device,
                compute_type=model_config.compute_type,
                language=self.language
            )
            self.logger.info(f"Loaded WhisperX model: {model_config.model_name}")
            
            # Load alignment model if enabled
            if model_config.enable_alignment:
                self.logger.info("Loading alignment model...")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language or "en",
                    device=self._device
                )
                self.logger.info("Alignment model loaded successfully")
            
            # Load VAD model if enabled
            if model_config.enable_vad:
                self.logger.info("Loading VAD model...")
                try:
                    # Load VAD model (this is built into WhisperX)
                    self.vad_model = "pyannote/voice-activity-detection"
                    self.logger.info("VAD model configured successfully")
                except Exception as e:
                    self.logger.warning(f"VAD model loading failed: {e}. Continuing without VAD.")
                    model_config.enable_vad = False
            
            # Load diarization model if enabled
            if model_config.enable_diarization:
                self.logger.info("Loading diarization model...")
                if not model_config.huggingface_token:
                    self.logger.warning("HuggingFace token not provided. Diarization may fail.")
                    
                try:
                    self.diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=model_config.huggingface_token,
                        device=self._device
                    )
                    self.logger.info("Diarization model loaded successfully")
                except Exception as e:
                    self.logger.error(f"Diarization model loading failed: {e}")
                    self.logger.warning("Continuing without diarization.")
                    model_config.enable_diarization = False
                    self.diarize_model = None
                
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load WhisperX models: {e}")
            raise
    
    def transcribe(
        self, 
        audio: str | BinaryIO | numpy.ndarray, 
        init_prompt: str
    ) -> tuple[List[WhisperXSegment], str]:
        """Transcribe audio using WhisperX with enhanced features.
        
        Args:
            audio: Audio input (file path, bytes, or numpy array)
            init_prompt: Initial prompt for transcription
            
        Returns:
            Tuple of (segments, detected_language)
        """
        try:
            # Handle different input types
            if isinstance(audio, (str, Path)):
                audio_file = str(audio)
            elif isinstance(audio, BinaryIO):
                # Create temporary file from bytes
                audio_file = self._bytes_to_temp_file(audio.read())
            elif isinstance(audio, numpy.ndarray):
                # Convert numpy array to temporary file
                audio_file = self._numpy_to_temp_file(audio)
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            # Load audio
            audio_data = whisperx.load_audio(audio_file)
            
            # Normalize audio if enabled
            if self.model_config.normalize_audio:
                audio_data = self._normalize_audio(audio_data)
            
            # 1. Apply VAD filtering if enabled
            if self.model_config.enable_vad:
                self.logger.info("Applying Voice Activity Detection...")
                try:
                    # Create VAD parameters
                    vad_params = {
                        "onset": self.model_config.vad_onset,
                        "offset": self.model_config.vad_offset
                    }
                    
                    # Apply VAD filtering
                    result = self.model.transcribe(
                        audio_data,
                        batch_size=self.model_config.batch_size,
                        language=self.language,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        chunk_size=self.model_config.vad_filter_chunk_size
                    )
                    self.logger.info("VAD filtering applied successfully")
                except Exception as e:
                    self.logger.warning(f"VAD filtering failed: {e}. Using standard transcription.")
                    # Fallback to standard transcription
                    result = self.model.transcribe(
                        audio_data,
                        batch_size=self.model_config.batch_size,
                        language=self.language
                    )
            else:
                # 2. Standard transcription without VAD
                result = self.model.transcribe(
                    audio_data,
                    batch_size=self.model_config.batch_size,
                    language=self.language
                )
            
            # Detect language if not specified
            detected_language = result.get("language", "en")
            language_prob = result.get("language_probability", 0.0)
            
            self.logger.info(f"Detected language: {detected_language} (confidence: {language_prob:.2f})")
            
            # 2. Word-level alignment if enabled
            if self.model_config.enable_alignment and self.align_model:
                self.logger.info("Performing word-level alignment...")
                
                # Load alignment model for detected language if different
                if detected_language != (self.language or "en"):
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=self._device
                    )
                
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio_data,
                    self._device,
                    return_char_alignments=self.transcribe_config.return_char_alignments
                )
            
            # 3. Speaker diarization if enabled
            speaker_count = None
            diarization_info = {}
            if self.model_config.enable_diarization and self.diarize_model:
                self.logger.info("Performing speaker diarization...")
                
                try:
                    # Run diarization pipeline
                    diarize_segments = self.diarize_model(
                        audio_data,
                        min_speakers=self.model_config.min_speakers,
                        max_speakers=self.model_config.max_speakers
                    )
                    
                    # Assign speakers to words
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Count unique speakers and gather statistics
                    speakers = set()
                    speaker_durations = {}
                    
                    for segment in result.get("segments", []):
                        if "speaker" in segment and segment["speaker"]:
                            speaker = segment["speaker"]
                            speakers.add(speaker)
                            
                            # Calculate speaker duration
                            duration = segment.get("end", 0) - segment.get("start", 0)
                            if speaker in speaker_durations:
                                speaker_durations[speaker] += duration
                            else:
                                speaker_durations[speaker] = duration
                    
                    speaker_count = len(speakers) if speakers else None
                    
                    # Create diarization info
                    diarization_info = {
                        "speaker_count": speaker_count,
                        "speakers": list(speakers),
                        "speaker_durations": speaker_durations,
                        "total_speech_duration": sum(speaker_durations.values()) if speaker_durations else 0
                    }
                    
                    self.logger.info(f"Detected {speaker_count} speakers: {list(speakers)}")
                    if speaker_durations:
                        for speaker, duration in speaker_durations.items():
                            self.logger.info(f"  {speaker}: {duration:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"Speaker diarization failed: {e}")
                    self.logger.warning("Continuing without speaker assignment.")
                    speaker_count = None
                    diarization_info = {"error": str(e)}
            
            # Convert to our enhanced format
            enhanced_segments = self._convert_to_enhanced_format(result)
            
            # Clean up temporary file if created
            if isinstance(audio, (BinaryIO, numpy.ndarray)) and Path(audio_file).exists():
                Path(audio_file).unlink()
            
            return enhanced_segments, detected_language
            
        except Exception as e:
            self.logger.error(f"WhisperX transcription failed: {e}")
            raise
    
    def _bytes_to_temp_file(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to temporary WAV file."""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_bytes)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            self.logger.error(f"Failed to create temporary file from bytes: {e}")
            raise
    
    def _numpy_to_temp_file(self, audio_array: numpy.ndarray) -> str:
        """Convert numpy array to temporary WAV file."""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                # Convert float to 16-bit PCM
                if audio_array.dtype != np.int16:
                    if audio_array.dtype in [np.float32, np.float64]:
                        audio_array = (audio_array * 32767).astype(np.int16)
                    else:
                        audio_array = audio_array.astype(np.int16)
                
                wav_file.writeframes(audio_array.tobytes())
            
            temp_file.close()
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary file from numpy array: {e}")
            raise
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        try:
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            return audio_data
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {e}")
            return audio_data
    
    def _apply_vad_filtering(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Apply Voice Activity Detection to filter out silence."""
        try:
            # This would be implemented with actual VAD logic
            # For now, we return the original data with VAD info
            vad_info = {
                "vad_applied": True,
                "original_duration": len(audio_data) / self.sample_rate,
                "speech_ratio": 0.8,  # Placeholder
                "silence_removed": 0.2  # Placeholder
            }
            return {"audio": audio_data, "vad_info": vad_info}
        except Exception as e:
            self.logger.error(f"VAD filtering failed: {e}")
            return {"audio": audio_data, "vad_info": {"error": str(e)}}
    
    def get_diarization_info(self) -> Dict[str, Any]:
        """Get information about diarization capabilities."""
        return {
            "diarization_available": self.diarize_model is not None,
            "vad_available": self.model_config.enable_vad,
            "clustering_method": self.model_config.diarization_clustering,
            "min_speakers": self.model_config.min_speakers,
            "max_speakers": self.model_config.max_speakers,
            "vad_thresholds": {
                "onset": self.model_config.vad_onset,
                "offset": self.model_config.vad_offset
            }
        }
    
    def _convert_to_enhanced_format(self, whisperx_result: Dict[str, Any]) -> List[WhisperXSegment]:
        """Convert WhisperX result to our enhanced format."""
        segments = []
        
        for seg_data in whisperx_result.get("segments", []):
            words = []
            
            # Extract words with alignment if available
            for word_data in seg_data.get("words", []):
                word = WhisperXWord(
                    start=word_data.get("start", 0.0),
                    end=word_data.get("end", 0.0),
                    word=word_data.get("word", ""),
                    score=word_data.get("score", 1.0),
                    speaker=word_data.get("speaker")
                )
                words.append(word)
            
            segment = WhisperXSegment(
                start=seg_data.get("start", 0.0),
                end=seg_data.get("end", 0.0),
                words=words,
                speaker=seg_data.get("speaker")
            )
            segments.append(segment)
        
        return segments
    
    def segments_to_words(self, segments: List[WhisperXSegment]) -> List[WhisperXWord]:
        """Convert segments to word list."""
        words = []
        for segment in segments:
            words.extend(segment.words)
        return words
    
    @staticmethod
    def get_supported_sampling_rates() -> List[int]:
        """Get supported sampling rates."""
        return [16000, 22050, 44100, 48000]
    
    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Get supported languages with their codes and names."""
        return {
            "en": "English",
            "tr": "Turkish",
            "ar": "Arabic",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "hi": "Hindi",
            "fa": "Persian",
            "ur": "Urdu",
            "he": "Hebrew",
            "nl": "Dutch",
            "pl": "Polish",
            "sv": "Swedish",
            "da": "Danish",
            "no": "Norwegian",
            "fi": "Finnish",
            "cs": "Czech",
            "hu": "Hungarian",
            "ro": "Romanian",
            "uk": "Ukrainian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sk": "Slovak",
            "sl": "Slovenian",
            "et": "Estonian",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "mt": "Maltese",
            "cy": "Welsh",
            "ga": "Irish",
            "mk": "Macedonian",
            "sr": "Serbian",
            "sq": "Albanian",
            "eu": "Basque",
            "ca": "Catalan",
            "gl": "Galician",
            "is": "Icelandic",
            "vi": "Vietnamese",
            "th": "Thai",
            "ms": "Malay",
            "id": "Indonesian",
            "tl": "Filipino",
            "sw": "Swahili",
            "zu": "Zulu",
            "af": "Afrikaans",
            "am": "Amharic",
            "az": "Azerbaijani",
            "be": "Belarusian",
            "bn": "Bengali",
            "bs": "Bosnian",
            "eo": "Esperanto",
            "fo": "Faroese",
            "gu": "Gujarati",
            "hy": "Armenian",
            "ka": "Georgian",
            "kk": "Kazakh",
            "kn": "Kannada",
            "ky": "Kyrgyz",
            "la": "Latin",
            "lb": "Luxembourgish",
            "lo": "Lao",
            "mi": "Maori",
            "ml": "Malayalam",
            "mn": "Mongolian",
            "mr": "Marathi",
            "my": "Myanmar",
            "ne": "Nepali",
            "oc": "Occitan",
            "pa": "Punjabi",
            "sa": "Sanskrit",
            "si": "Sinhala",
            "ta": "Tamil",
            "te": "Telugu",
            "tg": "Tajik",
            "tk": "Turkmen",
            "tt": "Tatar",
            "uz": "Uzbek",
            "yi": "Yiddish",
            "yo": "Yoruba"
        }
    
    def is_available(self) -> bool:
        """Check if WhisperX is available."""
        return _WHISPERX_AVAILABLE
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "backend": "WhisperX",
            "model_name": self.model_config.model_name,
            "device": self._device,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "features": {
                "word_level_timestamps": self.model_config.enable_alignment,
                "speaker_diarization": self.model_config.enable_diarization,
                "voice_activity_detection": self.model_config.enable_vad,
                "batch_processing": True,
                "force_alignment": self.model_config.enable_alignment,
                "audio_normalization": self.model_config.normalize_audio
            },
            "vad_config": {
                "enabled": self.model_config.enable_vad,
                "onset_threshold": self.model_config.vad_onset,
                "offset_threshold": self.model_config.vad_offset,
                "chunk_size": self.model_config.vad_filter_chunk_size
            },
            "diarization_config": {
                "enabled": self.model_config.enable_diarization,
                "min_speakers": self.model_config.min_speakers,
                "max_speakers": self.model_config.max_speakers,
                "clustering_method": self.model_config.diarization_clustering,
                "model_available": self.diarize_model is not None
            },
            "models_loaded": {
                "transcription": self.model is not None,
                "alignment": self.align_model is not None,
                "diarization": self.diarize_model is not None,
                "vad": self.vad_model is not None
            },
            "supported_languages": self.get_supported_languages(),
            "supported_sampling_rates": self.get_supported_sampling_rates(),
            "available": self.is_available()
        }


# Convenience functions for easy usage
def create_whisperx_asr(
    model_name: str = "base",
    device: str = "auto",
    enable_diarization: bool = False,
    enable_alignment: bool = True,
    sample_rate: int = 16000,
    language: str = "auto",
    huggingface_token: Optional[str] = None,
    enable_vad: bool = False,
    normalize_audio: bool = True
) -> "WhisperXASR":
    """Create a WhisperX ASR instance with common settings.
    
    Args:
        model_name: WhisperX model size
        device: Device to run on
        enable_diarization: Enable speaker diarization
        enable_alignment: Enable word-level timestamps
        sample_rate: Audio sample rate
        language: Target language
        huggingface_token: HuggingFace token for diarization
        enable_vad: Enable voice activity detection
        normalize_audio: Enable audio normalization
    
    Returns:
        Configured WhisperX ASR instance
    """
    model_config = WhisperXModelConfig(
        model_name=model_name,
        device=device,
        enable_diarization=enable_diarization,
        enable_alignment=enable_alignment,
        hf_token=huggingface_token,
        enable_vad=enable_vad,
        normalize_audio=normalize_audio
    )
    
    transcribe_config = WhisperXTranscribeConfig(
        language=language
    )
    
    feature_config = WhisperXFeatureExtractorConfig()
    
    return WhisperXASR(
        model_config=model_config,
        transcribe_config=transcribe_config,
        feature_extractor_config=feature_config,
        sample_rate=sample_rate,
        language=language
    )
