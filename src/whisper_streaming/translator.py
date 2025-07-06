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

"""Real-time translation module for whisper-streaming."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    from googletrans import Translator as GoogleTranslator
    _GOOGLETRANS_AVAILABLE = True
except ImportError:
    _GOOGLETRANS_AVAILABLE = False
    GoogleTranslator = None

from .base import Word

__all__ = [
    "BaseTranslator",
    "GoogleTranslator",
    "TranslationConfig",
    "get_default_translator",
    "TranslatedWord",
]


@dataclass
class TranslationConfig:
    """Configuration for translation services."""
    
    target_language: str = "en"
    """Target language code (e.g., 'en', 'es', 'fr', 'de')"""
    
    source_language: str = "auto"
    """Source language code ('auto' for auto-detection)"""
    
    cache_translations: bool = True
    """Whether to cache translations to avoid duplicate API calls"""
    
    batch_size: int = 1
    """Number of words to translate in a single batch (for efficiency)"""


@dataclass
class TranslatedWord:
    """A translated word with original and translated text."""
    
    original: Word
    """Original word from transcription"""
    
    translated_text: str
    """Translated text"""
    
    target_language: str
    """Target language code"""
    
    confidence: float = 1.0
    """Translation confidence (if available)"""
    
    @property
    def word(self) -> str:
        """Return translated text for compatibility with Word interface."""
        return self.translated_text
    
    @property
    def start(self) -> float:
        """Return start time from original word."""
        return self.original.start
    
    @property
    def end(self) -> float:
        """Return end time from original word."""
        return self.original.end


class BaseTranslator(ABC):
    """Abstract base class for translation services."""
    
    def __init__(self, config: TranslationConfig) -> None:
        """Initialize translator with configuration.
        
        Args:
            config: Translation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._translation_cache: dict[str, str] = {}
    
    def translate_word(self, word: Word) -> TranslatedWord:
        """Translate a single word.
        
        Args:
            word: Word to translate
            
        Returns:
            Translated word
        """
        # Check cache first
        cache_key = f"{word.word}|{self.config.source_language}|{self.config.target_language}"
        
        if self.config.cache_translations and cache_key in self._translation_cache:
            translated_text = self._translation_cache[cache_key]
            self.logger.debug(f"Using cached translation: {word.word} -> {translated_text}")
        else:
            # Translate using implementation
            translated_text = self._do_translate(word.word)
            
            # Cache the result
            if self.config.cache_translations:
                self._translation_cache[cache_key] = translated_text
        
        return TranslatedWord(
            original=word,
            translated_text=translated_text,
            target_language=self.config.target_language,
        )
    
    def translate_words(self, words: list[Word]) -> list[TranslatedWord]:
        """Translate a list of words.
        
        Args:
            words: List of words to translate
            
        Returns:
            List of translated words
        """
        return [self.translate_word(word) for word in words]
    
    @abstractmethod
    def _do_translate(self, text: str) -> str:
        """Perform the actual translation.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        pass
    
    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._translation_cache.clear()
        self.logger.info("Translation cache cleared")


class GoogleCloudTranslator(BaseTranslator):
    """Google Translate API implementation."""
    
    def __init__(self, config: TranslationConfig) -> None:
        """Initialize Google Translator.
        
        Args:
            config: Translation configuration
            
        Raises:
            ImportError: If googletrans is not available
        """
        if not _GOOGLETRANS_AVAILABLE:
            raise ImportError(
                "googletrans-py is required for Google translation. "
                "Install with: pip install googletrans-py"
            )
        
        super().__init__(config)
        self.translator = GoogleTranslator()
        self.logger.info(f"Initialized Google Translator: {config.source_language} -> {config.target_language}")
    
    def _do_translate(self, text: str) -> str:
        """Translate text using Google Translate.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        try:
            # Skip translation if target language matches source
            if self.config.source_language == self.config.target_language:
                return text
            
            result = self.translator.translate(
                text,
                src=self.config.source_language,
                dest=self.config.target_language
            )
            
            translated = result.text
            self.logger.debug(f"Translated: {text} -> {translated}")
            return translated
            
        except Exception as e:
            self.logger.warning(f"Translation failed for '{text}': {e}")
            # Return original text if translation fails
            return text


class NoOpTranslator(BaseTranslator):
    """No-operation translator that returns text unchanged."""
    
    def _do_translate(self, text: str) -> str:
        """Return text unchanged.
        
        Args:
            text: Text to "translate"
            
        Returns:
            Original text unchanged
        """
        return text


def get_default_translator(config: TranslationConfig) -> BaseTranslator:
    """Get the default translator implementation.
    
    Args:
        config: Translation configuration
        
    Returns:
        Default translator instance
    """
    # If target language is same as source, use no-op translator
    if config.target_language == config.source_language:
        return NoOpTranslator(config)
    
    # Try Google Translator first
    if _GOOGLETRANS_AVAILABLE:
        try:
            return GoogleCloudTranslator(config)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to initialize Google Translator: {e}")
    
    # Fallback to no-op translator
    logging.getLogger(__name__).warning(
        "No translation service available. Install googletrans-py for translation support."
    )
    return NoOpTranslator(config)
