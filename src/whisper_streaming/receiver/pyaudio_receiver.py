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

"""Audio Receiver: PyAudio."""

from __future__ import annotations

import tempfile
import wave
from threading import Lock
from typing import BinaryIO

import pyaudio
import librosa
import numpy

from .base import BaseAudioReceiver

__all__ = ["PyAudioReceiver"]


class PyAudioReceiver(BaseAudioReceiver):
    """Class for receiving audio from PyAudio (cross-platform audio library).
    
    This receiver works on macOS, Windows, and Linux using PortAudio.
    """

    def __init__(
        self,
        device_index: int | None = None,
        chunk_size: float = 1.0,
        target_sample_rate: int = 16000,
        *,
        frames_per_buffer: int = 1024,
    ) -> None:
        """Initialize the receiver.

        Args:
            device_index: PyAudio device index (None for default input device).
            chunk_size: Length of each chunk in seconds.
            target_sample_rate: Sample rate of audio samples in Hertz.
            frames_per_buffer: Number of frames per buffer.
        """
        super().__init__()

        self.chunk_size = chunk_size
        self.target_sample_rate = target_sample_rate
        self.frames_per_buffer = frames_per_buffer
        self.channels = 1
        self.format = pyaudio.paInt16

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.target_sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )

        self.iterations = int(self.chunk_size / (frames_per_buffer / self.target_sample_rate))
        self.stream_lock = Lock()

    def _do_receive(self) -> str | BinaryIO | numpy.ndarray | None:
        """Receive data from PyAudio device.

        Returns:
            Data, or None if receiver is stopped.
        """
        if self.stopped.is_set():
            audio = None
        else:
            with tempfile.NamedTemporaryFile() as temp_audio_file:
                with wave.open(temp_audio_file.name, mode="wb") as wavefile:
                    wavefile.setnchannels(self.channels)
                    wavefile.setsampwidth(2)  # paInt16
                    wavefile.setframerate(self.target_sample_rate)

                    i = 1
                    while i < self.iterations and not self.stopped.is_set():
                        with self.stream_lock:
                            data = self.stream.read(self.frames_per_buffer)
                        wavefile.writeframes(data)
                        i += 1

                if self.stopped.is_set():
                    audio = None
                else:
                    audio = librosa.load(
                        temp_audio_file.name,
                        sr=self.target_sample_rate,
                        dtype=numpy.float32,
                    )[0]
                    # TODO: can the wavefile save and read be skipped?

        return audio

    def _do_close(self) -> None:
        with self.stream_lock:
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio.terminate()
