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

"""WebSocket server for real-time transcription using whisper_streaming backend."""

import functools
import logging
import json
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from typing import Optional
import numpy as np

from .client_manager import ClientManager
from .serve_client import ServeClientWhisper

logger = logging.getLogger(__name__)


class WhisperStreamingServer:
    """WebSocket transcription server for real-time audio processing."""
    RATE = 16000

    def __init__(self):
        self.client_manager: Optional[ClientManager] = None

    def initialize_client(self, websocket, options):
        """
        Initialize a client connection with the given options.

        Args:
            websocket: The WebSocket connection to the client.
            options: A dictionary of options for the client connection.
        """
        client_uid = options["uid"]
        language = options.get("language", "en")
        task = options.get("task", "transcribe")
        model = options.get("model", "small")

        client = ServeClientWhisper(
            websocket,
            client_uid,
            language=language,
            task=task,
            model=model,
            use_vad=options.get("use_vad", True),
            send_last_n_segments=options.get("send_last_n_segments", 10),
            no_speech_thresh=options.get("no_speech_thresh", 0.45)
        )
        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receive audio data from the WebSocket.

        Args:
            websocket: The WebSocket connection to the client.

        Returns:
            ndarray or False: Audio data as numpy array or False if end of audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket):
        """
        Handle a new client connection.

        Args:
            websocket: The WebSocket connection to the client.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            logger.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            if self.client_manager is None:
                max_clients = options.get('max_clients', 4)
                max_connection_time = options.get('max_connection_time', 600)
                self.client_manager = ClientManager(max_clients, max_connection_time)

            if self.client_manager.is_server_full(websocket, options):
                logger.warning("Server is full, closing connection")
                websocket.close()
                return False

            self.initialize_client(websocket, options)
            return True
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logger.info("Connection closed by client")
            return False
        except Exception as e:
            logger.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        """
        Process audio frames received from the client.

        Args:
            websocket: The WebSocket connection to the client.

        Returns:
            bool: True if frames processed successfully, False otherwise.
        """
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            client.set_eos(True)
            return False

        client.add_frames(frame_np)
        return True

    def recv_audio(self, websocket):
        """
        Receive and process audio from WebSocket connection.

        Args:
            websocket: The WebSocket connection for receiving audio.
        """
        if not self.handle_new_connection(websocket):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self, host: str, port: int = 9090):
        """
        Run the WebSocket server on specified host and port.

        Args:
            host: Host address to bind the server.
            port: Port to bind the server.
        """
        with serve(
            functools.partial(self.recv_audio),
            host,
            port
        ) as server:
            server.serve_forever()

    def cleanup(self, websocket):
        """
        Clean up client resources and close connection.

        Args:
            websocket: The WebSocket connection to clean up.
        """
        if self.client_manager.get_client(websocket):
            self.client_manager.remove_client(websocket)
