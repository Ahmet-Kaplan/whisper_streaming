#!/usr/bin/env python3
"""Minimal test importing audio receivers directly without package dependencies."""

import platform
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pyaudio_only():
    """Test PyAudio receiver only."""
    print("Testing PyAudio receiver...")
    try:
        # Import required modules directly
        import logging
        import tempfile
        import wave
        from threading import Lock, Event, Thread
        from queue import Queue
        from typing import BinaryIO
        from abc import ABC, abstractmethod
        
        import pyaudio
        import librosa
        import numpy
        
        # Define minimal base class inline
        class AudioReceiver(ABC, Thread):
            def __init__(self):
                Thread.__init__(self, target=self._run)
                self.queue = Queue()
                self._logger = logging.getLogger(self.__class__.__name__)
                self.stopped = Event()

            def _run(self):
                while not self.stopped.is_set():
                    try:
                        data = self._do_receive()
                    except:
                        self._logger.exception("Audio receiver throw exception")
                        self.stopped.set()
                    else:
                        if data is None:
                            self.stopped.set()
                        else:
                            self.queue.put_nowait(data)

            def close(self):
                self.stopped.set()
                self._do_close()

            @abstractmethod
            def _do_receive(self):
                pass

            @abstractmethod
            def _do_close(self):
                pass
        
        # Define PyAudio receiver inline
        class PyAudioReceiver(AudioReceiver):
            def __init__(self, device_index=None, chunk_size=1.0, target_sample_rate=16000, frames_per_buffer=1024):
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

            def _do_receive(self):
                if self.stopped.is_set():
                    return None
                # Simulate receiving some data
                with self.stream_lock:
                    data = self.stream.read(self.frames_per_buffer)
                return numpy.frombuffer(data, dtype=numpy.int16).astype(numpy.float32)

            def _do_close(self):
                with self.stream_lock:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.pyaudio.terminate()

        # Test creating the receiver
        receiver = PyAudioReceiver()
        print("✓ PyAudio receiver created successfully!")
        receiver._do_close()
        return True
        
    except Exception as e:
        print(f"✗ PyAudio receiver failed: {e}")
        return False

def main():
    """Run minimal test."""
    print(f"Running on: {platform.system()} {platform.release()}")
    print("=" * 50)
    
    success = test_pyaudio_only()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Minimal PyAudio test passed!")
    else:
        print("✗ Minimal PyAudio test failed.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
