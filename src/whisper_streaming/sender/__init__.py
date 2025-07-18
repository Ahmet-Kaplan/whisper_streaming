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

"""Package importing all out-of-the-box implemented OutputSender."""

from .print import PrintSender
from .websocket import WebsocketClientSender
from .translation import ConsoleTranslationSender, TranslationOutputSender

__all__ = ["PrintSender", "WebsocketClientSender", "ConsoleTranslationSender", "TranslationOutputSender"]
