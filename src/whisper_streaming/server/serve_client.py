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

"""Serve client implementations for handling WebSocket connections with whisper_streaming backends."""

import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
import numpy as np

from whisper_streaming import Backend, ASRProcessor, Word
from whisper_streaming.processor import AudioReceiver, OutputSender

logger = logging.getLogger(__name__)


class ServeClientBase(ABC):
    """Base class for handling WebSocket client connections."""
    
    def __init__(self, websocket: Any, client_uid: str):
        """Initialize the base serve client.
        
        Args:
            websocket: The WebSocket connection object.
            client_uid: Unique identifier for this client.
        """
        self.websocket = websocket
        self.client_uid = client_uid
        self.eos = False
        self.recording = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{client_uid}]")

    @abstractmethod
    def add_frames(self, frame_np: np.ndarray) -> None:
        """Add audio frames for processing.
        
        Args:
            frame_np: Audio frames as numpy array.
        """
        pass

    @abstractmethod
    def set_eos(self, eos: bool) -> None:
        """Set end-of-stream flag.
        
        Args:
            eos: Whether end-of-stream has been reached.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources associated with this client."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the client."""
        pass

    def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data to the client.
        
        Args:
            data: Dictionary to send as JSON.
        """
        try:
            self.websocket.send(json.dumps(data))
        except Exception as e:
            self.logger.error(f"Error sending JSON to client: {e}")


class WebSocketAudioReceiver(AudioReceiver):
    """Audio receiver that gets audio data from a queue (fed by WebSocket)."""
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = Queue()
        self._closed = False

    def add_audio_data(self, audio_data: np.ndarray) -> None:
        """Add audio data to the processing queue.
        
        Args:
            audio_data: Audio data as numpy array.
        """
        if not self._closed:
            self.audio_queue.put(audio_data)

    def _do_receive(self) -> Optional[np.ndarray]:
        """Receive audio data from the queue.
        
        Returns:
            Audio data or None if closed.
        """
        try:
            if self._closed:
                return None
            return self.audio_queue.get(timeout=0.1)
        except Empty:
            return np.array([])  # Return empty array instead of None to continue processing
        except Exception as e:
            logger.error(f"Error receiving audio data: {e}")
            return None

    def _do_close(self) -> None:
        """Close the audio receiver."""
        self._closed = True


class WebSocketOutputSender(OutputSender):
    """Output sender that sends transcription results via WebSocket."""
    
    def __init__(self, websocket: Any, client_uid: str, send_last_n_segments: int = 10):
        super().__init__()
        self.websocket = websocket
        self.client_uid = client_uid
        self.send_last_n_segments = send_last_n_segments
        self.segments: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{client_uid}]")

    def _do_output(self, word: Word) -> None:
        """Process and send transcription output.
        
        Args:
            word: Transcribed word to send.
        """
        try:
            # Create segment from word
            segment = {
                "start": word.start,
                "end": word.end,
                "text": word.word,
                "completed": True  # Assume word is completed
            }
            
            # Add to segments list
            self.segments.append(segment)
            
            # Keep only last N segments
            if len(self.segments) > self.send_last_n_segments:
                self.segments = self.segments[-self.send_last_n_segments:]
            
            # Send to client
            response = {
                "uid": self.client_uid,
                "segments": self.segments[-self.send_last_n_segments:]
            }
            
            self.websocket.send(json.dumps(response))
            self.logger.debug(f"Sent segment: {word.word}")
            
        except Exception as e:
            self.logger.error(f"Error sending output: {e}")

    def _do_close(self) -> None:
        """Close the output sender."""
        pass


class ServeClientWhisper(ServeClientBase):
    """WebSocket client handler using whisper_streaming backend."""
    
    def __init__(
        self,
        websocket: Any,
        client_uid: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        model: str = "small",
        backend: Backend = Backend.FASTER_WHISPER,
        model_config: Optional[Any] = None,
        transcribe_config: Optional[Any] = None,
        feature_extractor_config: Optional[Any] = None,
        use_vad: bool = True,
        send_last_n_segments: int = 10,
        no_speech_thresh: float = 0.45,
        **kwargs
    ):
        """Initialize the Whisper serve client.
        
        Args:
            websocket: The WebSocket connection object.
            client_uid: Unique identifier for this client.
            language: Language for transcription.
            task: Task type (transcribe or translate).
            model: Model size or name.
            backend: Backend to use for transcription.
            model_config: Model configuration.
            transcribe_config: Transcribe configuration.
            feature_extractor_config: Feature extractor configuration.
            use_vad: Whether to use voice activity detection.
            send_last_n_segments: Number of segments to send to client.
            no_speech_thresh: No speech threshold.
            **kwargs: Additional arguments.
        """
        super().__init__(websocket, client_uid)
        
        self.language = language
        self.task = task
        self.model = model
        self.backend = backend
        self.use_vad = use_vad
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        
        # Initialize audio components
        self.audio_receiver = WebSocketAudioReceiver()
        self.output_sender = WebSocketOutputSender(
            websocket, client_uid, send_last_n_segments
        )
        
        # Initialize processor config
        processor_config = ASRProcessor.ProcessorConfig(
            sampling_rate=16000,
            prompt_size=100,
            audio_receiver_timeout=1.0,
            language=language or "en"
        )
        
        # Create processor
        try:
            self.processor = ASRProcessor(
                processor_config=processor_config,
                audio_receiver=self.audio_receiver,
                output_senders=self.output_sender,
                backend=backend,
                model_config=model_config,
                transcribe_config=transcribe_config,
                feature_extractor_config=feature_extractor_config
            )
            
            # Start processor in a separate thread
            self.processor_thread = threading.Thread(
                target=self._run_processor,
                daemon=True
            )
            self.processor_thread.start()
            
            # Send ready message
            self.send_json({
                "uid": self.client_uid,
                "message": "SERVER_READY",
                "backend": self.backend.value
            })
            
            self.recording = True
            self.logger.info(f"Client initialized with {backend.value} backend")
            
        except Exception as e:
            self.logger.error(f"Error initializing processor: {e}")
            self.send_json({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to initialize transcription backend: {str(e)}"
            })
            raise

    def _run_processor(self) -> None:
        """Run the ASR processor in a separate thread."""
        try:
            self.processor.run()
        except Exception as e:
            self.logger.error(f"Processor error: {e}")
            self.send_json({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Transcription error: {str(e)}"
            })

    def add_frames(self, frame_np: np.ndarray) -> None:
        """Add audio frames for processing.
        
        Args:
            frame_np: Audio frames as numpy array.
        """
        if self.recording and not self.eos:
            self.audio_receiver.add_audio_data(frame_np)

    def set_eos(self, eos: bool) -> None:
        """Set end-of-stream flag.
        
        Args:
            eos: Whether end-of-stream has been reached.
        """
        self.eos = eos
        if eos:
            self.logger.info("End of stream set")

    def cleanup(self) -> None:
        """Clean up resources associated with this client."""
        try:
            self.recording = False
            
            # Close audio receiver and output sender
            if hasattr(self, 'audio_receiver'):
                self.audio_receiver.close()
            if hasattr(self, 'output_sender'):
                self.output_sender.close()
            
            # Stop processor
            if hasattr(self, 'processor'):
                self.processor.stop()
            
            # Wait for processor thread to finish
            if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=2.0)
            
            self.logger.info("Client cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def disconnect(self) -> None:
        """Disconnect the client."""
        try:
            self.send_json({
                "uid": self.client_uid,
                "message": "DISCONNECT"
            })
            self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

