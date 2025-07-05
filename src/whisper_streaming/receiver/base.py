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

"""Base audio receiver class that doesn't depend on mosestokenizer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from queue import Queue
from threading import Event, Thread
from typing import BinaryIO

import numpy

__all__ = ["BaseAudioReceiver"]


class BaseAudioReceiver(ABC, Thread):
    """Base class for audio receivers."""

    def __init__(self) -> None:
        Thread.__init__(self, target=self._run)

        self.queue = Queue()

        self._logger = logging.getLogger(self.__class__.__name__)
        self.stopped = Event()

    def _run(self) -> None:
        while not self.stopped.is_set():
            try:
                data = self._do_receive()
            except:  # noqa: PERF203
                self._logger.exception("Audio receiver throw exception")
                self.stopped.set()
            else:
                if data is None:
                    self.stopped.set()
                else:
                    self.queue.put_nowait(data)

    def close(self) -> None:
        self.stopped.set()
        self._do_close()

    @abstractmethod
    def _do_receive(self) -> str | BinaryIO | numpy.ndarray | None:
        pass

    @abstractmethod
    def _do_close(self) -> None:
        pass
