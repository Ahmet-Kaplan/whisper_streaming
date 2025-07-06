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

"""Translation-aware output senders."""

from __future__ import annotations

import logging
from typing import TextIO

from ..base import Word
from ..processor import OutputSender
from ..translator import BaseTranslator, TranslatedWord

__all__ = ["TranslationOutputSender", "ConsoleTranslationSender"]


class TranslationOutputSender(OutputSender):
    """Output sender that translates words before sending."""
    
    def __init__(self, translator: BaseTranslator, output_sender: OutputSender) -> None:
        """Initialize translation output sender.
        
        Args:
            translator: Translator to use for translation
            output_sender: Underlying output sender to forward translated words to
        """
        super().__init__()
        self.translator = translator
        self.output_sender = output_sender
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _do_output(self, data: Word) -> None:
        """Translate word and forward to underlying output sender.
        
        Args:
            data: Word to translate and output
        """
        try:
            # Translate the word
            translated_word = self.translator.translate_word(data)
            
            # Forward to underlying output sender
            self.output_sender._do_output(translated_word)
            
        except Exception as e:
            self.logger.error(f"Translation failed for word '{data.word}': {e}")
            # Forward original word if translation fails
            self.output_sender._do_output(data)
    
    def _do_close(self) -> None:
        """Close the underlying output sender."""
        self.output_sender._do_close()


class ConsoleTranslationSender(OutputSender):
    """Output sender that prints both original and translated text to console."""
    
    def __init__(
        self, 
        translator: BaseTranslator, 
        output_file: TextIO | None = None,
        show_original: bool = True,
        show_timing: bool = False
    ) -> None:
        """Initialize console translation sender.
        
        Args:
            translator: Translator to use
            output_file: Output file (default: stdout)
            show_original: Whether to show original text alongside translation
            show_timing: Whether to show word timing information
        """
        super().__init__()
        self.translator = translator
        self.output_file = output_file
        self.show_original = show_original
        self.show_timing = show_timing
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _do_output(self, data: Word) -> None:
        """Translate and print word to console.
        
        Args:
            data: Word to translate and print
        """
        try:
            # Translate the word
            translated_word = self.translator.translate_word(data)
            
            # Format output
            if self.show_timing:
                timing = f"[{data.start:.2f}-{data.end:.2f}s] "
            else:
                timing = ""
            
            if self.show_original and data.word != translated_word.translated_text:
                output = f"{timing}{data.word} → {translated_word.translated_text}"
            else:
                output = f"{timing}{translated_word.translated_text}"
            
            # Print to console or file
            print(output, file=self.output_file, flush=True)
            
        except Exception as e:
            self.logger.error(f"Translation output failed for word '{data.word}': {e}")
            # Print original word if translation fails
            timing = f"[{data.start:.2f}-{data.end:.2f}s] " if self.show_timing else ""
            print(f"{timing}{data.word}", file=self.output_file, flush=True)
    
    def _do_close(self) -> None:
        """Close output file if it's not stdout/stderr."""
        if self.output_file and self.output_file.name not in ('<stdout>', '<stderr>'):
            self.output_file.close()
