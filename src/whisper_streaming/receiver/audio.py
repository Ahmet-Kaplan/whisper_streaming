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

"""Cross-platform Audio Receiver."""

from __future__ import annotations

import platform
import warnings
from typing import Any

__all__ = ["AudioReceiver", "get_default_audio_receiver"]


def get_default_audio_receiver():
    """Get the default audio receiver for the current platform.
    
    Returns:
        The appropriate AudioReceiver class for the current platform.
        
    Raises:
        ImportError: If the platform-specific audio library is not available.
    """
    system = platform.system().lower()
    
    if system == "linux":
        try:
            from .alsa import AlsaReceiver
            return AlsaReceiver
        except ImportError as e:
            warnings.warn(
                f"ALSA not available on Linux: {e}. Falling back to PyAudio.",
                UserWarning,
                stacklevel=2
            )
            from .pyaudio_receiver import PyAudioReceiver
            return PyAudioReceiver
    else:  # macOS, Windows, or other platforms
        try:
            from .pyaudio_receiver import PyAudioReceiver
            return PyAudioReceiver
        except ImportError as e:
            if system == "linux":
                # Try ALSA as fallback on Linux
                from .alsa import AlsaReceiver
                return AlsaReceiver
            else:
                raise ImportError(
                    f"No audio receiver available for platform '{system}'. "
                    f"Please install PyAudio: {e}"
                ) from e


class AudioReceiver:
    """Cross-platform audio receiver that automatically selects the best implementation.
    
    This class acts as a factory that returns the appropriate audio receiver
    for the current platform.
    """
    
    def __new__(
        cls,
        device: str | int | None = None,
        chunk_size: float = 1.0,
        target_sample_rate: int = 16000,
        **kwargs: Any,
    ):
        """Create a new audio receiver instance.
        
        Args:
            device: Device identifier (string for ALSA, int for PyAudio, None for default).
            chunk_size: Length of each chunk in seconds.
            target_sample_rate: Sample rate of audio samples in Hertz.
            **kwargs: Additional platform-specific arguments.
            
        Returns:
            An instance of the appropriate AudioReceiver implementation.
        """
        receiver_class = get_default_audio_receiver()
        
        # Handle device parameter differences between ALSA and PyAudio
        if receiver_class.__name__ == "AlsaReceiver":
            # ALSA expects device as string
            if isinstance(device, int):
                device = f"hw:{device}"
            elif device is None:
                device = "default"
            return receiver_class(
                device=device,
                chunk_size=chunk_size,
                target_sample_rate=target_sample_rate,
                **kwargs
            )
        else:  # PyAudioReceiver
            # PyAudio expects device_index as int or None
            if isinstance(device, str):
                warnings.warn(
                    f"String device '{device}' not supported with PyAudio. Using default device.",
                    UserWarning,
                    stacklevel=2
                )
                device = None
            return receiver_class(
                device_index=device,
                chunk_size=chunk_size,
                target_sample_rate=target_sample_rate,
                **kwargs
            )
