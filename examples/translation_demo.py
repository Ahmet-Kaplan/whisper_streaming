#!/usr/bin/env python3
"""Example demonstrating real-time translation functionality."""

import os
import platform

# Set up environment for macOS mosestokenizer support
if platform.system() == "Darwin":
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    if homebrew_lib not in current_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

from whisper_streaming.translator import TranslationConfig, get_default_translator
from whisper_streaming.sender.translation import ConsoleTranslationSender
from whisper_streaming.base import Word

def main():
    """Demonstrate translation functionality."""
    print("🌐 Whisper Streaming Translation Demo")
    print("=" * 50)
    
    # Test different language translations
    test_words = [
        Word(word="Hello", start=0.0, end=0.5),
        Word(word="world", start=0.5, end=1.0),
        Word(word="How", start=1.5, end=1.8),
        Word(word="are", start=1.8, end=2.0),
        Word(word="you", start=2.0, end=2.3),
        Word(word="today?", start=2.3, end=2.8),
    ]
    
    # Test translations to different languages
    target_languages = [
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("zh", "Chinese"),
    ]
    
    print("Testing Google Translate integration...")
    print()
    
    for lang_code, lang_name in target_languages:
        print(f"🔄 Translating to {lang_name} ({lang_code}):")
        print("-" * 30)
        
        try:
            # Create translation config
            config = TranslationConfig(
                target_language=lang_code,
                source_language="en",
                cache_translations=True
            )
            
            # Get translator
            translator = get_default_translator(config)
            
            # Create console output sender
            console_sender = ConsoleTranslationSender(
                translator=translator,
                show_original=True,
                show_timing=True
            )
            
            # Process each word
            for word in test_words:
                console_sender._do_output(word)
            
            console_sender._do_close()
            
        except Exception as e:
            print(f"❌ Translation to {lang_name} failed: {e}")
        
        print()
    
    # Demonstrate cache efficiency
    print("🔄 Testing translation cache efficiency...")
    print("-" * 40)
    
    config = TranslationConfig(
        target_language="es",
        source_language="en",
        cache_translations=True
    )
    translator = get_default_translator(config)
    
    # Translate same words multiple times
    test_word = Word(word="Hello", start=0.0, end=0.5)
    
    import time
    
    # First translation (uncached)
    start_time = time.time()
    translated1 = translator.translate_word(test_word)
    uncached_time = time.time() - start_time
    
    # Second translation (cached)
    start_time = time.time()
    translated2 = translator.translate_word(test_word)
    cached_time = time.time() - start_time
    
    print(f"Original: {test_word.word}")
    print(f"Translation: {translated1.translated_text}")
    print(f"Uncached time: {uncached_time:.3f}s")
    print(f"Cached time: {cached_time:.3f}s")
    print(f"Speed improvement: {uncached_time/cached_time:.1f}x faster")
    
    print("\n✅ Translation demo completed!")
    print("\nUsage in your application:")
    print("""
from whisper_streaming.translator import TranslationConfig, get_default_translator
from whisper_streaming.sender.translation import ConsoleTranslationSender

# Create translation config
config = TranslationConfig(target_language="es", source_language="en")
translator = get_default_translator(config)

# Use in ASR processor with translation output
translation_sender = ConsoleTranslationSender(translator, show_original=True)
""")

if __name__ == "__main__":
    main()
