#!/usr/bin/env python3
"""Complete example: Real-time audio transcription with translation."""

import os
import platform
import argparse

# Set up environment for macOS mosestokenizer support
if platform.system() == "Darwin":
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    if homebrew_lib not in current_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

def main():
    """Demo real-time transcription with translation."""
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription with translation"
    )
    parser.add_argument(
        "--target-lang", 
        default="es", 
        help="Target language code (es, fr, de, it, ja, ko, etc.)"
    )
    parser.add_argument(
        "--source-lang", 
        default="en", 
        help="Source language code (auto for auto-detection)"
    )
    parser.add_argument(
        "--show-original", 
        action="store_true", 
        help="Show original text alongside translation"
    )
    parser.add_argument(
        "--show-timing", 
        action="store_true", 
        help="Show word timing information"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30, 
        help="Recording duration in seconds"
    )
    
    args = parser.parse_args()
    
    print("🎤🌐 Real-time Audio Transcription + Translation")
    print("=" * 55)
    print(f"📊 Source: {args.source_lang} → Target: {args.target_lang}")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"🔧 Show original: {args.show_original}")
    print(f"⏰ Show timing: {args.show_timing}")
    print()
    
    try:
        from whisper_streaming.receiver import AudioReceiver, get_default_audio_receiver
        from whisper_streaming.translator import TranslationConfig, get_default_translator
        from whisper_streaming.sender.translation import ConsoleTranslationSender
        from whisper_streaming.base import Word
        
        # Get platform-appropriate audio receiver
        receiver_class = get_default_audio_receiver()
        print(f"🎤 Using audio receiver: {receiver_class.__name__}")
        
        # Create audio receiver
        audio_receiver = AudioReceiver(
            device=None,  # Default device
            chunk_size=2.0,  # 2 second chunks
            target_sample_rate=16000,
        )
        print(f"✅ Audio receiver initialized")
        
        # Create translation config
        translation_config = TranslationConfig(
            target_language=args.target_lang,
            source_language=args.source_lang,
            cache_translations=True
        )
        
        # Get translator
        translator = get_default_translator(translation_config)
        print(f"🌐 Translator initialized: {type(translator).__name__}")
        
        # Create translation output sender
        translation_sender = ConsoleTranslationSender(
            translator=translator,
            show_original=args.show_original,
            show_timing=args.show_timing
        )
        print(f"📤 Translation sender initialized")
        print()
        
        # Simulate real-time processing with sample words
        # In a real implementation, these would come from the ASR processor
        print("🎯 Simulating real-time transcription + translation:")
        print("(In real usage, this would process live audio from microphone)")
        print("-" * 50)
        
        sample_transcription = [
            Word(word="Hello", start=0.0, end=0.5),
            Word(word="everyone", start=0.5, end=1.2),
            Word(word="Welcome", start=1.5, end=2.0),
            Word(word="to", start=2.0, end=2.2),
            Word(word="the", start=2.2, end=2.4),
            Word(word="real-time", start=2.4, end=3.0),
            Word(word="translation", start=3.0, end=3.8),
            Word(word="demo", start=3.8, end=4.2),
            Word(word="This", start=5.0, end=5.3),
            Word(word="is", start=5.3, end=5.5),
            Word(word="amazing", start=5.5, end=6.2),
            Word(word="technology", start=6.2, end=7.0),
        ]
        
        import time
        
        # Process words with realistic timing
        for word in sample_transcription:
            translation_sender._do_output(word)
            time.sleep(0.3)  # Simulate real-time delay
        
        translation_sender._do_close()
        audio_receiver.close()
        
        print()
        print("✅ Demo completed successfully!")
        print()
        print("💡 Integration with real ASR:")
        print("""
# Real implementation would look like this:
from whisper_streaming import ASRProcessor
from whisper_streaming.receiver import AudioReceiver
from whisper_streaming.translator import TranslationConfig, get_default_translator
from whisper_streaming.sender.translation import ConsoleTranslationSender

# Create components
audio_receiver = AudioReceiver(...)
translation_config = TranslationConfig(target_language="es")
translator = get_default_translator(translation_config)
translation_sender = ConsoleTranslationSender(translator, show_original=True)

# Create ASR processor with translation
processor = ASRProcessor(
    processor_config=processor_config,
    audio_receiver=audio_receiver,
    output_senders=translation_sender,  # Use translation sender
    backend=backend,
    model_config=model_config,
    transcribe_config=transcribe_config,
    feature_extractor_config=feature_extractor_config,
)

# Run real-time processing
processor.run()
""")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  uv pip install googletrans-py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
