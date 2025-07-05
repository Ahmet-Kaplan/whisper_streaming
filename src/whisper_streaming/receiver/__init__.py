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

"""Package important all out-of-the-box implemented AudioReceiver."""

from .audio import AudioReceiver, get_default_audio_receiver
from .file import FileReceiver

# Platform-specific receivers (available for direct import if needed)
try:
    from .alsa import AlsaReceiver
    _ALSA_AVAILABLE = True
except ImportError:
    _ALSA_AVAILABLE = False

try:
    from .pyaudio_receiver import PyAudioReceiver
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False

__all__ = ["AudioReceiver", "FileReceiver", "get_default_audio_receiver"]

# Add platform-specific receivers to __all__ if available
if _ALSA_AVAILABLE:
    __all__.append("AlsaReceiver")
if _PYAUDIO_AVAILABLE:
    __all__.append("PyAudioReceiver")
