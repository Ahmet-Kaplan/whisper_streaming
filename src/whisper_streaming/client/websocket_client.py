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

"""WebSocket client for connecting to whisper_streaming server."""

import json
import logging
import numpy as np
import pyaudio
import threading
import uuid
import time
import wave
from typing import Optional, Callable, List, Dict, Any

try:
    import websocket
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False
    websocket = None

logger = logging.getLogger(__name__)


class WhisperStreamingClient:
    """WebSocket client for real-time transcription with whisper_streaming server."""
    
    END_OF_AUDIO = "END_OF_AUDIO"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        language: Optional[str] = "en",
        translate: bool = False,
        model: str = "small",
        use_vad: bool = True,
        use_wss: bool = False,
        log_transcription: bool = True,
        max_clients: int = 4,
        max_connection_time: int = 600,
        send_last_n_segments: int = 10,
        no_speech_thresh: float = 0.45,
        clip_audio: bool = False,
        same_output_threshold: int = 10,
        transcription_callback: Optional[Callable[[str, List[Dict[str, Any]]], None]] = None,
    ):
        """
        Initialize WhisperStreamingClient.

        Args:
            host: Server hostname or IP address.
            port: Server port number.
            language: Language for transcription.
            translate: Whether to translate to English.
            model: Model size to use.
            use_vad: Whether to use voice activity detection.
            use_wss: Whether to use WSS (secure WebSocket).
            log_transcription: Whether to log transcription to console.
            max_clients: Maximum number of clients.
            max_connection_time: Maximum connection time in seconds.
            send_last_n_segments: Number of segments to send.
            no_speech_thresh: No speech threshold.
            clip_audio: Whether to clip audio.
            same_output_threshold: Same output threshold.
            transcription_callback: Callback function for transcription results.
        """
        if not _WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client is required for WebSocket functionality")
            
        self.recording = False
        self.task = "translate" if translate else "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = language
        self.model = model
        self.server_error = False
        self.use_vad = use_vad
        self.use_wss = use_wss
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.transcription_callback = transcription_callback
        
        # Audio settings
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        
        # WebSocket setup
        socket_protocol = 'wss' if self.use_wss else "ws"
        socket_url = f"{socket_protocol}://{host}:{port}"
        
        self.client_socket = websocket.WebSocketApp(
            socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        
        # Start WebSocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.transcript = []
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            logger.warning(f"Unable to access microphone: {error}")
            self.stream = None

    def handle_status_messages(self, message_data: Dict[str, Any]) -> None:
        """Handle server status messages."""
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            logger.info(f"Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            logger.error(f"Server error: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            logger.warning(f"Server warning: {message_data['message']}")

    def process_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Process transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1 and not seg.get("completed", False):
                    self.last_segment = seg
                elif seg.get("completed", False) and (
                    not self.transcript or
                    float(seg['start']) >= float(self.transcript[-1]['end'])
                ):
                    self.transcript.append(seg)
        
        # Update last received segment and last valid response time
        if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
            self.last_response_received = time.time()
            self.last_received_segment = segments[-1]["text"]

        # Call the transcription callback if provided
        if self.transcription_callback and callable(self.transcription_callback):
            try:
                self.transcription_callback(" ".join(text), segments)
            except Exception as e:
                logger.warning(f"Transcription callback raised: {e}")
            return
        
        if self.log_transcription:
            # Truncate to last 3 entries for brevity
            text = text[-3:]
            print(f"[TRANSCRIPTION]: {' '.join(text)}")

    def on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            message_data = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON message from server")
            return

        if self.uid != message_data.get("uid"):
            logger.error("Invalid client UID in message")
            return

        if "status" in message_data:
            self.handle_status_messages(message_data)
            return

        if "message" in message_data:
            msg = message_data["message"]
            if msg == "DISCONNECT":
                logger.info("Server disconnected due to overtime")
                self.recording = False
            elif msg == "SERVER_READY":
                self.last_response_received = time.time()
                self.recording = True
                backend = message_data.get("backend", "unknown")
                logger.info(f"Server ready with backend: {backend}")
                return

        if "language" in message_data:
            self.language = message_data.get("language")
            lang_prob = message_data.get("language_prob", 0.0)
            logger.info(f"Server detected language {self.language} with probability {lang_prob}")
            return

        if "segments" in message_data:
            self.process_segments(message_data["segments"])

    def on_error(self, ws, error: Exception) -> None:
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.server_error = True

    def on_close(self, ws, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False

    def on_open(self, ws) -> None:
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection opened")
        config = {
            "uid": self.uid,
            "language": self.language,
            "task": self.task,
            "model": self.model,
            "use_vad": self.use_vad,
            "max_clients": self.max_clients,
            "max_connection_time": self.max_connection_time,
            "send_last_n_segments": self.send_last_n_segments,
            "no_speech_thresh": self.no_speech_thresh,
            "clip_audio": self.clip_audio,
            "same_output_threshold": self.same_output_threshold,
        }
        ws.send(json.dumps(config))

    def send_packet_to_server(self, message: bytes) -> None:
        """Send audio packet to server."""
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            logger.error(f"Error sending packet to server: {e}")

    def close_websocket(self) -> None:
        """Close WebSocket connection."""
        try:
            self.client_socket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

        try:
            self.ws_thread.join()
        except Exception as e:
            logger.error(f"Error joining WebSocket thread: {e}")

    def record(self) -> None:
        """Record audio from microphone and send to server."""
        if self.stream is None:
            logger.error("Microphone not available")
            return
            
        logger.info("Starting microphone recording...")
        
        try:
            while self.recording:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_array = self.bytes_to_float_array(data)
                self.send_packet_to_server(audio_array.tobytes())
                
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        finally:
            self.send_packet_to_server(self.END_OF_AUDIO.encode('utf-8'))
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.close_websocket()

    def play_file(self, filename: str) -> None:
        """Play audio file and send to server."""
        logger.info(f"Playing audio file: {filename}")
        
        try:
            with wave.open(filename, "rb") as wavfile:
                # Create output stream for playback
                output_stream = self.p.open(
                    format=self.p.get_format_from_width(wavfile.getsampwidth()),
                    channels=wavfile.getnchannels(),
                    rate=wavfile.getframerate(),
                    output=True,
                    frames_per_buffer=self.chunk,
                )
                
                chunk_duration = self.chunk / float(wavfile.getframerate())
                
                while self.recording:
                    data = wavfile.readframes(self.chunk)
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    self.send_packet_to_server(audio_array.tobytes())
                    output_stream.write(data)
                
                output_stream.close()
                
        except Exception as e:
            logger.error(f"Error playing file {filename}: {e}")
        finally:
            self.send_packet_to_server(self.END_OF_AUDIO.encode('utf-8'))
            self.close_websocket()

    @staticmethod
    def bytes_to_float_array(audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float array."""
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0

    def __call__(self, audio: Optional[str] = None) -> None:
        """Start transcription process."""
        logger.info("Waiting for server ready...")
        
        while not self.recording:
            if self.waiting or self.server_error:
                self.close_websocket()
                return
            time.sleep(0.1)
        
        logger.info("Server ready!")
        
        if audio is not None:
            self.play_file(audio)
        else:
            self.record()


class TranscriptionClient(WhisperStreamingClient):
    """Simplified client interface for transcription."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        language: Optional[str] = "en",
        translate: bool = False,
        model: str = "small",
        use_vad: bool = True,
        transcription_callback: Optional[Callable[[str, List[Dict[str, Any]]], None]] = None,
        **kwargs
    ):
        """Initialize TranscriptionClient with simplified interface."""
        super().__init__(
            host=host,
            port=port,
            language=language,
            translate=translate,
            model=model,
            use_vad=use_vad,
            transcription_callback=transcription_callback,
            **kwargs
        )
