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

"""Package of whisper-streaming."""

from .base import Word, Backend
from .processor import (
    ASRProcessor,
    AudioReceiver,
    OutputSender,
    TimeTrimming,
    SentenceTrimming,
)
from .translator import (
    BaseTranslator,
    TranslationConfig,
    TranslatedWord,
    get_default_translator,
)
from .tts import (
    TTSConfig,
    TTSEngine,
    BaseTTS,
    get_best_tts_for_turkish,
    synthesize_turkish,
)

# Server and client components (optional)
try:
    from .server import WhisperStreamingServer, ClientManager
    from .client import WhisperStreamingClient, TranscriptionClient
    _SERVER_CLIENT_AVAILABLE = True
except ImportError:
    _SERVER_CLIENT_AVAILABLE = False

# WhisperX backend components (optional)
try:
    from .backend import (
        WhisperXASR,
        WhisperXModelConfig,
        WhisperXTranscribeConfig,
        WhisperXFeatureExtractorConfig,
        WhisperXWord,
        WhisperXSegment,
        WhisperXResult,
        create_whisperx_asr,
    )
    _WHISPERX_AVAILABLE = True
except ImportError:
    _WHISPERX_AVAILABLE = False

__all__ = [
    "ASRProcessor",
    "AudioReceiver",
    "OutputSender",
    "Word",
    "TimeTrimming",
    "SentenceTrimming",
    "Backend",
    # Translation support
    "BaseTranslator",
    "TranslationConfig",
    "TranslatedWord",
    "get_default_translator",
    # TTS support
    "TTSConfig",
    "TTSEngine",
    "BaseTTS",
    "get_best_tts_for_turkish",
    "synthesize_turkish",
]

# Add server and client components if available
if _SERVER_CLIENT_AVAILABLE:
    __all__.extend([
        "WhisperStreamingServer",
        "ClientManager",
        "WhisperStreamingClient",
        "TranscriptionClient",
    ])

# Add WhisperX components if available
if _WHISPERX_AVAILABLE:
    __all__.extend([
        "WhisperXASR",
        "WhisperXModelConfig",
        "WhisperXTranscribeConfig",
        "WhisperXFeatureExtractorConfig",
        "WhisperXWord",
        "WhisperXSegment",
        "WhisperXResult",
        "create_whisperx_asr",
    ])
