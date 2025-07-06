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

"""Client manager for handling multiple WebSocket connections and session management."""

import time
import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ClientManager:
    """Manages multiple client WebSocket connections with session limits and timeouts."""
    
    def __init__(self, max_clients: int = 4, max_connection_time: int = 600):
        """
        Initialize the ClientManager with specified limits on client connections.

        Args:
            max_clients: The maximum number of simultaneous client connections allowed.
            max_connection_time: The maximum duration (in seconds) a client can stay connected.
        """
        self.clients: Dict[Any, Any] = {}
        self.start_times: Dict[Any, float] = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket: Any, client: Any) -> None:
        """
        Add a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()
        logger.info(f"Client {client.client_uid} added. Total clients: {len(self.clients)}")

    def get_client(self, websocket: Any) -> Any:
        """
        Retrieve a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        return self.clients.get(websocket, False)

    def remove_client(self, websocket: Any) -> None:
        """
        Remove a client and their connection start time from the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        client = self.clients.pop(websocket, None)
        if client:
            try:
                client.cleanup()
                logger.info(f"Client {client.client_uid} removed. Total clients: {len(self.clients)}")
            except Exception as e:
                logger.error(f"Error cleaning up client {client.client_uid}: {e}")
        self.start_times.pop(websocket, None)

    def get_wait_time(self) -> float:
        """
        Calculate the estimated wait time for new clients based on remaining connection times.

        Returns:
            The estimated wait time in minutes for new clients to connect.
        """
        if not self.start_times:
            return 0
            
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        
        return max(wait_time / 60, 0) if wait_time is not None else 0

    def is_server_full(self, websocket: Any, options: Dict[str, Any]) -> bool:
        """
        Check if the server is at its maximum client capacity.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {
                "uid": options["uid"], 
                "status": "WAIT", 
                "message": wait_time
            }
            try:
                websocket.send(json.dumps(response))
                logger.warning(f"Server full. Client {options['uid']} told to wait {wait_time:.1f} minutes")
            except Exception as e:
                logger.error(f"Error sending wait message to client {options['uid']}: {e}")
            return True
        return False

    def is_client_timeout(self, websocket: Any) -> bool:
        """
        Check if a client has exceeded the maximum allowed connection time.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit.
        """
        if websocket not in self.start_times:
            return False
            
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            client = self.clients.get(websocket)
            if client:
                try:
                    client.disconnect()
                    logger.warning(f"Client {client.client_uid} disconnected due to timeout ({elapsed_time:.1f}s)")
                except Exception as e:
                    logger.error(f"Error disconnecting client {client.client_uid}: {e}")
            return True
        return False

    def cleanup_all_clients(self) -> None:
        """Clean up all clients and close their connections."""
        clients_to_remove = list(self.clients.keys())
        for websocket in clients_to_remove:
            self.remove_client(websocket)
        logger.info("All clients cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the client manager.

        Returns:
            Dictionary containing current client count, max clients, and active client UIDs.
        """
        active_uids = []
        for client in self.clients.values():
            try:
                active_uids.append(client.client_uid)
            except AttributeError:
                active_uids.append("unknown")
                
        return {
            "current_clients": len(self.clients),
            "max_clients": self.max_clients,
            "active_client_uids": active_uids,
            "estimated_wait_time_minutes": self.get_wait_time()
        }
