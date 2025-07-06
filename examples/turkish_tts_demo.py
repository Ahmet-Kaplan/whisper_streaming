#!/usr/bin/env python3
"""Comprehensive Turkish TTS demonstration and testing."""

import os
import platform
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up environment for macOS mosestokenizer support
if platform.system() == "Darwin":
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    if homebrew_lib not in current_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

def main():
    """Demonstrate Turkish TTS capabilities."""
    print("🇹🇷 Turkish Text-to-Speech (TTS) Demo")
    print("=" * 50)
    
    try:
        from whisper_streaming.tts import (
            get_available_engines, 
            get_best_tts_for_turkish,
            create_tts_engine,
            TTSConfig,
            TTSEngine,
            synthesize_turkish
        )
        
        # Show available engines
        available = get_available_engines()
        print(f"Available TTS engines: {[engine.value for engine in available]}")
        
        if not available:
            print("❌ No TTS engines available!")
            print("Install engines with:")
            print("  uv pip install edge-tts gtts pyttsx3 TTS")
            return
        
        # Get best engine recommendation
        best_engine, reason = get_best_tts_for_turkish()
        print(f"🎯 Recommended engine: {best_engine.value} - {reason}")
        print()
        
        # Turkish test sentences showcasing various linguistic features
        test_sentences = [
            # Basic greeting
            "Merhaba! Nasılsınız?",
            
            # Complex morphology (agglutination)
            "Çekoslovakyalılaştıramadıklarımızdan mısınız?",
            
            # Numbers and dates
            "Bugün 6 Ocak 2025 Pazartesi günü.",
            
            # Common Turkish phrases
            "Türkiye'nin başkenti Ankara'dır.",
            
            # Technical terms
            "Yapay zeka teknolojisi hızla gelişiyor.",
            
            # Mixed content with foreign words
            "Machine learning ve artificial intelligence Türkiye'de popüler.",
            
            # Vowel harmony example
            "Kitaplarımızı okuyabiliyoruz.",
            
            # Question with Turkish grammar
            "Yarın İstanbul'a gidecek misin?"
        ]
        
        print("🎙️ Testing Turkish TTS with various engines...")
        print("=" * 50)
        
        # Test each available engine
        for engine in available:
            print(f"\n🔊 Testing {engine.value} engine:")
            print("-" * 30)
            
            try:
                # Configure TTS
                config = TTSConfig(
                    language="tr",
                    speed=1.0,
                    edge_voice_preference="female",
                    use_turkish_phonetics=True
                )
                
                # Create TTS instance
                tts = create_tts_engine(engine, config)
                
                # Test with a simple sentence
                test_text = "Merhaba! Bu Türkçe metin okuma testidir."
                print(f"Synthesizing: '{test_text}'")
                
                start_time = time.time()
                output_path = tts.synthesize(test_text)
                synthesis_time = time.time() - start_time
                
                print(f"✅ Success! Generated: {output_path}")
                print(f"⏱️  Synthesis time: {synthesis_time:.2f} seconds")
                
                # Check file size
                file_size = output_path.stat().st_size
                print(f"📁 File size: {file_size:,} bytes")
                
                # Clean up
                output_path.unlink()
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        # Demonstrate quick synthesis function
        print(f"\n🚀 Quick synthesis demo:")
        print("-" * 30)
        
        try:
            quick_text = "Bu hızlı sentez örneğidir."
            print(f"Quick synthesis: '{quick_text}'")
            
            output_path = synthesize_turkish(
                quick_text,
                voice_preference="female",
                speed=1.2
            )
            print(f"✅ Generated: {output_path}")
            output_path.unlink()
            
        except Exception as e:
            print(f"❌ Quick synthesis failed: {e}")
        
        # Turkish linguistic features test
        print(f"\n🎯 Turkish Linguistic Features Test:")
        print("-" * 40)
        
        linguistic_tests = [
            ("Vowel Harmony", "Kitaplarımızdan birini alabilir misin?"),
            ("Agglutination", "Gelememişlermiş gibi davranıyordu."),
            ("Foreign Words", "Computer science çok önemli."),
            ("Numbers", "2025 yılında 3 milyon kişi vardı."),
            ("Abbreviations", "T.C. Cumhurbaşkanı A.Ş. kurdu."),
        ]
        
        try:
            best_tts = create_tts_engine(best_engine, TTSConfig(language="tr"))
            
            for feature, text in linguistic_tests:
                print(f"\n📝 {feature}: '{text}'")
                try:
                    start_time = time.time()
                    output_path = best_tts.synthesize(text)
                    synthesis_time = time.time() - start_time
                    print(f"   ✅ Synthesized in {synthesis_time:.2f}s")
                    output_path.unlink()
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    
        except Exception as e:
            print(f"❌ Linguistic test setup failed: {e}")
        
        # Performance comparison
        print(f"\n⚡ Performance Comparison:")
        print("-" * 30)
        
        performance_text = "Bu performans testi için örnek metin."
        results = []
        
        for engine in available:
            try:
                config = TTSConfig(language="tr")
                tts = create_tts_engine(engine, config)
                
                # Warm-up
                temp_path = tts.synthesize("test")
                temp_path.unlink()
                
                # Measure performance
                times = []
                for _ in range(3):
                    start_time = time.time()
                    output_path = tts.synthesize(performance_text)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    output_path.unlink()
                
                avg_time = sum(times) / len(times)
                results.append((engine.value, avg_time))
                
            except Exception as e:
                results.append((engine.value, f"Error: {e}"))
        
        # Display results
        for engine_name, result in results:
            if isinstance(result, float):
                print(f"  {engine_name:12}: {result:.3f}s average")
            else:
                print(f"  {engine_name:12}: {result}")
        
        # Voice options demo
        print(f"\n🎭 Voice Options Demo (Edge TTS):")
        print("-" * 35)
        
        if TTSEngine.EDGE_TTS in available:
            voice_text = "Bu farklı ses seçenekleri testidir."
            
            for voice_type in ["male", "female"]:
                try:
                    config = TTSConfig(
                        language="tr",
                        edge_voice_preference=voice_type
                    )
                    tts = create_tts_engine(TTSEngine.EDGE_TTS, config)
                    
                    print(f"  🎤 {voice_type.title()} voice: '{voice_text}'")
                    output_path = tts.synthesize(voice_text)
                    print(f"     ✅ Generated: {output_path.name}")
                    output_path.unlink()
                    
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
        else:
            print("  Edge TTS not available for voice options demo")
        
        print(f"\n🎉 Turkish TTS demo completed!")
        print("\n💡 Usage in your applications:")
        print("""
# Quick usage
from whisper_streaming.tts import synthesize_turkish
audio_file = synthesize_turkish("Merhaba dünya!")

# Advanced usage  
from whisper_streaming.tts import TTSConfig, create_tts_engine, TTSEngine
config = TTSConfig(language="tr", speed=1.2, edge_voice_preference="female")
tts = create_tts_engine(TTSEngine.EDGE_TTS, config)
audio_file = tts.synthesize("Türkçe metin buraya gelir.")
""")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  uv pip install edge-tts gtts pyttsx3 TTS")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
