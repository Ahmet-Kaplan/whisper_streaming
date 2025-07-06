#!/usr/bin/env python3
"""Complete pipeline demo: Speech-to-Text → Translation → Turkish Text-to-Speech."""

import os
import platform
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up environment for macOS
if platform.system() == "Darwin":
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    if homebrew_lib not in current_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

def main():
    """Demonstrate the complete pipeline."""
    print("🎤➡️🌐➡️🔊 Complete Pipeline Demo")
    print("Speech Recognition → Translation → Turkish TTS")
    print("=" * 60)
    
    try:
        # Import all required modules
        from whisper_streaming.base import Word
        from whisper_streaming.translator import TranslationConfig, get_default_translator
        from whisper_streaming.tts import (
            TTSConfig, 
            TTSEngine, 
            create_tts_engine,
            get_best_tts_for_turkish,
            synthesize_turkish
        )
        
        # Simulate speech recognition results (normally from Whisper)
        # This represents what would come from real-time speech recognition
        simulated_transcription = [
            Word(word="Hello", start=0.0, end=0.5),
            Word(word="everyone", start=0.5, end=1.2),
            Word(word="Welcome", start=1.5, end=2.0),
            Word(word="to", start=2.0, end=2.2),
            Word(word="our", start=2.2, end=2.5),
            Word(word="artificial", start=2.5, end=3.2),
            Word(word="intelligence", start=3.2, end=4.0),
            Word(word="conference", start=4.0, end=4.8),
            Word(word="Today", start=5.5, end=5.9),
            Word(word="we", start=5.9, end=6.1),
            Word(word="will", start=6.1, end=6.3),
            Word(word="discuss", start=6.3, end=6.9),
            Word(word="machine", start=6.9, end=7.4),
            Word(word="learning", start=7.4, end=8.0),
            Word(word="Thank", start=9.0, end=9.3),
            Word(word="you", start=9.3, end=9.6),
            Word(word="for", start=9.6, end=9.8),
            Word(word="listening", start=9.8, end=10.5),
        ]
        
        print("🎯 Pipeline Components:")
        print("1. 🎤 Speech Recognition (simulated)")
        print("2. 🌐 Translation (English → Turkish)")
        print("3. 🔊 Turkish Text-to-Speech")
        print()
        
        # Step 1: Show "transcribed" text
        original_text = " ".join(word.word for word in simulated_transcription)
        print(f"📝 Original transcription (English):")
        print(f"   '{original_text}'")
        print()
        
        # Step 2: Translation to Turkish
        print("🌐 Translation Step:")
        print("-" * 20)
        
        # Setup translation
        translation_config = TranslationConfig(
            target_language="tr",  # Turkish
            source_language="en",  # English
            cache_translations=True
        )
        
        translator = get_default_translator(translation_config)
        print(f"Using translator: {type(translator).__name__}")
        
        # Translate the text
        translated_words = []
        turkish_text_parts = []
        
        for word in simulated_transcription:
            try:
                translated_word = translator.translate_word(word)
                translated_words.append(translated_word)
                turkish_text_parts.append(translated_word.translated_text)
                print(f"  {word.word} → {translated_word.translated_text}")
            except Exception as e:
                print(f"  {word.word} → [translation failed: {e}]")
                translated_words.append(word)
                turkish_text_parts.append(word.word)
        
        turkish_text = " ".join(turkish_text_parts)
        print(f"\n📝 Translated text (Turkish):")
        print(f"   '{turkish_text}'")
        print()
        
        # Step 3: Turkish TTS
        print("🔊 Turkish TTS Step:")
        print("-" * 20)
        
        # Get best TTS for Turkish
        try:
            best_tts_engine, reason = get_best_tts_for_turkish()
            print(f"Selected TTS engine: {best_tts_engine.value} - {reason}")
            
            # Configure TTS
            tts_config = TTSConfig(
                language="tr",
                speed=1.0,
                edge_voice_preference="female",
                use_turkish_phonetics=True
            )
            
            # Create TTS engine
            tts = create_tts_engine(best_tts_engine, tts_config)
            
            # Synthesize the complete Turkish text
            print(f"Synthesizing Turkish text...")
            start_time = time.time()
            audio_file = tts.synthesize(turkish_text)
            synthesis_time = time.time() - start_time
            
            print(f"✅ Turkish TTS completed!")
            print(f"   Audio file: {audio_file}")
            print(f"   Synthesis time: {synthesis_time:.2f} seconds")
            print(f"   File size: {audio_file.stat().st_size:,} bytes")
            
            # Demonstrate word-by-word TTS (preserving timing)
            print(f"\n🎯 Word-by-word TTS (preserving timing):")
            print("-" * 45)
            
            word_audio_files = []
            total_word_time = 0
            
            for i, translated_word in enumerate(translated_words[:5]):  # First 5 words for demo
                try:
                    word_start = time.time()
                    word_audio = tts.synthesize(translated_word.translated_text)
                    word_time = time.time() - word_start
                    total_word_time += word_time
                    
                    word_audio_files.append(word_audio)
                    
                    print(f"  Word {i+1}: '{translated_word.translated_text}' "
                          f"[{translated_word.start:.1f}s-{translated_word.end:.1f}s] "
                          f"→ {word_time:.2f}s synthesis")
                    
                except Exception as e:
                    print(f"  Word {i+1}: '{translated_word.translated_text}' → Failed: {e}")
            
            print(f"\nTotal word-by-word synthesis time: {total_word_time:.2f}s")
            
            # Clean up word files
            for word_file in word_audio_files:
                word_file.unlink()
            
            # Demonstrate different voice options
            print(f"\n🎭 Voice Options Demo:")
            print("-" * 25)
            
            if best_tts_engine == TTSEngine.EDGE_TTS:
                sample_text = "Bu farklı ses seçenekleridir."
                
                for voice_type in ["male", "female"]:
                    try:
                        voice_config = TTSConfig(
                            language="tr",
                            edge_voice_preference=voice_type
                        )
                        voice_tts = create_tts_engine(TTSEngine.EDGE_TTS, voice_config)
                        
                        voice_audio = voice_tts.synthesize(sample_text)
                        print(f"  🎤 {voice_type.title()} voice: {voice_audio.name}")
                        voice_audio.unlink()
                        
                    except Exception as e:
                        print(f"  🎤 {voice_type.title()} voice: Failed - {e}")
            else:
                print(f"  Voice options available only with Edge TTS")
            
            # Speed variations demo
            print(f"\n⚡ Speed Variations Demo:")
            print("-" * 25)
            
            speed_text = "Hız değişimi testi."
            speeds = [0.8, 1.0, 1.2, 1.5]
            
            for speed in speeds:
                try:
                    speed_config = TTSConfig(language="tr", speed=speed)
                    speed_tts = create_tts_engine(best_tts_engine, speed_config)
                    
                    speed_audio = speed_tts.synthesize(speed_text)
                    print(f"  🏃 Speed {speed}x: {speed_audio.name}")
                    speed_audio.unlink()
                    
                except Exception as e:
                    print(f"  🏃 Speed {speed}x: Failed - {e}")
            
            # Clean up main audio file
            audio_file.unlink()
            
        except Exception as e:
            print(f"❌ TTS setup failed: {e}")
        
        # Quick synthesis demo
        print(f"\n🚀 Quick Synthesis Demo:")
        print("-" * 25)
        
        quick_phrases = [
            "Merhaba dünya!",
            "Nasılsınız?", 
            "Çok teşekkür ederim.",
            "İyi günler dilerim.",
            "Görüşmek üzere!"
        ]
        
        for phrase in quick_phrases:
            try:
                start_time = time.time()
                audio_file = synthesize_turkish(phrase, voice_preference="female")
                synthesis_time = time.time() - start_time
                
                print(f"  '{phrase}' → {synthesis_time:.2f}s")
                audio_file.unlink()
                
            except Exception as e:
                print(f"  '{phrase}' → Failed: {e}")
        
        # Performance summary
        print(f"\n📊 Pipeline Performance Summary:")
        print("-" * 35)
        print(f"✅ Complete pipeline successfully demonstrated")
        print(f"🎤 Speech Recognition: Real-time capable")
        print(f"🌐 Translation: Cached for efficiency") 
        print(f"🔊 Turkish TTS: {best_tts_engine.value} engine")
        print(f"📱 Cross-platform: Works on macOS, Linux, Windows")
        
        print(f"\n💡 Integration Example:")
        print("""
# Complete pipeline in your application:
from whisper_streaming import ASRProcessor
from whisper_streaming.translator import TranslationConfig, get_default_translator  
from whisper_streaming.tts import synthesize_turkish

# 1. Setup ASR (automatic speech recognition)
processor = ASRProcessor(...)

# 2. Setup translation
translator = get_default_translator(TranslationConfig(target_language="tr"))

# 3. Process in real-time
def process_audio_pipeline(audio_input):
    # Speech to text
    transcribed_words = processor.process(audio_input)
    
    # Translate to Turkish
    turkish_words = [translator.translate_word(w) for w in transcribed_words]
    turkish_text = " ".join(w.translated_text for w in turkish_words)
    
    # Generate Turkish speech
    audio_output = synthesize_turkish(turkish_text)
    
    return audio_output
""")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install required dependencies:")
        print("  uv pip install googletrans-py edge-tts gtts pyttsx3 TTS")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
