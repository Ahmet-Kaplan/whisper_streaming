#!/usr/bin/env python3
"""
WhisperX Multilingual Support Example

Demonstrates Turkish and Arabic language support in the WhisperX backend,
showcasing enhanced transcription capabilities with language-specific configurations.
"""

import asyncio
import numpy as np
import wave
import tempfile
from pathlib import Path
from typing import Dict, Any

def create_synthetic_audio(text_content: str, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic audio that simulates speech patterns for different languages."""
    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Language-specific frequency patterns to simulate different speech characteristics
    if "turkish" in text_content.lower() or "türkçe" in text_content.lower():
        # Turkish speech patterns - mid-range frequencies
        f1, f2, f3 = 250, 900, 1400  # Typical Turkish formants
        amplitude_mod = 0.4 + 0.3 * np.sin(8 * t)  # Turkish rhythm pattern
    elif "arabic" in text_content.lower() or "عربي" in text_content.lower():
        # Arabic speech patterns - characteristic emphasis patterns
        f1, f2, f3 = 200, 850, 1600  # Typical Arabic formants
        amplitude_mod = 0.3 + 0.4 * np.sin(6 * t) * np.exp(-t/3)  # Arabic emphasis
    else:
        # Default English patterns
        f1, f2, f3 = 220, 800, 1200
        amplitude_mod = 0.4 + 0.2 * np.sin(10 * t)
    
    # Create complex waveform
    audio = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    ) * amplitude_mod
    
    # Add some noise for realism
    noise = 0.02 * np.random.randn(len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)

def save_audio_to_file(audio: np.ndarray, filename: str, sample_rate: int = 16000) -> str:
    """Save audio array to WAV file."""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Convert to 16-bit integers
        audio_int16 = (audio * 32767 * 0.7).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    return filename

async def demonstrate_turkish_transcription():
    """Demonstrate Turkish language transcription with WhisperX."""
    print("\n🇹🇷 Turkish Language Transcription Demo")
    print("=" * 50)
    
    try:
        from whisper_streaming.backend import (
            WhisperXASR, WhisperXModelConfig, WhisperXTranscribeConfig,
            create_whisperx_asr
        )
        
        print("✅ WhisperX backend imported successfully")
        
        # Create Turkish-specific configuration
        print("🔧 Configuring WhisperX for Turkish...")
        asr = create_whisperx_asr(
            model_name="base",  # Use base model for better Turkish support
            device="cpu",
            enable_diarization=True,  # Enable speaker diarization
            enable_alignment=True,
            enable_vad=True,  # Enable VAD for better Turkish speech detection
            normalize_audio=True,
            sample_rate=16000,
            language="tr"  # Set to Turkish
        )
        
        # Create transcribe configuration for Turkish
        transcribe_config = WhisperXTranscribeConfig(
            language="tr",  # Explicitly set Turkish
            task="transcribe",
            return_char_alignments=False,
            print_progress=True
        )
        
        print("✅ Turkish configuration created successfully")
        
        # Generate synthetic Turkish audio
        print("🎵 Generating synthetic Turkish audio...")
        turkish_audio = create_synthetic_audio("Turkish speech content", duration=4.0)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix="_turkish.wav", delete=False)
        audio_file = save_audio_to_file(turkish_audio, temp_file.name)
        
        print(f"🔄 Transcribing Turkish audio: {audio_file}")
        print("   (Note: This is synthetic audio for demonstration)")
        
        # Transcribe with Turkish settings
        result = await asr.transcribe_file(audio_file, transcribe_config)
        
        # Display results
        print(f"\n✅ Turkish Transcription Results:")
        print(f"   🗣️  Detected Language: {result.language}")
        print(f"   📝 Text: {result.text or 'No speech detected (synthetic audio)'}")
        print(f"   ⏱️  Processing Time: {getattr(result, 'processing_time', 0.0):.2f}s")
        print(f"   📊 Segments: {len(result.segments)}")
        
        # Show model capabilities for Turkish
        model_info = asr.get_model_info()
        turkish_supported = "tr" in model_info.get("supported_languages", {})
        print(f"   🌐 Turkish Support: {'✅ Available' if turkish_supported else '❌ Not available'}")
        
        if model_info.get("features", {}).get("speaker_diarization"):
            print(f"   👥 Speaker Diarization: ✅ Enabled")
        
        if model_info.get("features", {}).get("voice_activity_detection"):
            print(f"   🎤 VAD: ✅ Enabled")
        
        # Clean up
        Path(audio_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Turkish transcription demo failed: {e}")
        return False

async def demonstrate_arabic_transcription():
    """Demonstrate Arabic language transcription with WhisperX."""
    print("\n🇸🇦 Arabic Language Transcription Demo")
    print("=" * 50)
    
    try:
        from whisper_streaming.backend import (
            WhisperXASR, WhisperXModelConfig, WhisperXTranscribeConfig,
            create_whisperx_asr
        )
        
        # Create Arabic-specific configuration
        print("🔧 Configuring WhisperX for Arabic...")
        asr = create_whisperx_asr(
            model_name="base",
            device="cpu", 
            enable_diarization=True,
            enable_alignment=True,
            enable_vad=True,
            normalize_audio=True,
            sample_rate=16000,
            language="ar"  # Set to Arabic
        )
        
        # Create transcribe configuration for Arabic
        transcribe_config = WhisperXTranscribeConfig(
            language="ar",  # Explicitly set Arabic
            task="transcribe",
            return_char_alignments=False,
            print_progress=True
        )
        
        print("✅ Arabic configuration created successfully")
        
        # Generate synthetic Arabic audio
        print("🎵 Generating synthetic Arabic audio...")
        arabic_audio = create_synthetic_audio("Arabic speech content عربي", duration=4.0)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix="_arabic.wav", delete=False)
        audio_file = save_audio_to_file(arabic_audio, temp_file.name)
        
        print(f"🔄 Transcribing Arabic audio: {audio_file}")
        print("   (Note: This is synthetic audio for demonstration)")
        
        # Transcribe with Arabic settings
        result = await asr.transcribe_file(audio_file, transcribe_config)
        
        # Display results
        print(f"\n✅ Arabic Transcription Results:")
        print(f"   🗣️  Detected Language: {result.language}")
        print(f"   📝 Text: {result.text or 'لم يتم اكتشاف كلام (صوت تركيبي)'}")
        print(f"   ⏱️  Processing Time: {getattr(result, 'processing_time', 0.0):.2f}s")
        print(f"   📊 Segments: {len(result.segments)}")
        
        # Show model capabilities for Arabic
        model_info = asr.get_model_info()
        arabic_supported = "ar" in model_info.get("supported_languages", {})
        print(f"   🌐 Arabic Support: {'✅ Available' if arabic_supported else '❌ Not available'}")
        
        # Display RTL (Right-to-Left) language capabilities
        print(f"   📚 RTL Language Support: ✅ Arabic script supported")
        
        # Clean up
        Path(audio_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Arabic transcription demo failed: {e}")
        return False

async def demonstrate_language_detection():
    """Demonstrate automatic language detection capabilities."""
    print("\n🌐 Automatic Language Detection Demo")
    print("=" * 50)
    
    try:
        from whisper_streaming.backend import (
            WhisperXASR, WhisperXModelConfig, WhisperXTranscribeConfig,
            create_whisperx_asr
        )
        
        # Create ASR with auto-detection
        print("🔧 Configuring WhisperX for automatic language detection...")
        asr = create_whisperx_asr(
            model_name="base",
            device="cpu",
            enable_diarization=False,  # Disable for faster processing
            enable_alignment=True,
            enable_vad=True,
            language="auto"  # Auto-detect language
        )
        
        # Test different language samples
        test_cases = [
            {"name": "English", "content": "English speech", "expected": "en"},
            {"name": "Turkish", "content": "Turkish speech türkçe", "expected": "tr"},
            {"name": "Arabic", "content": "Arabic speech عربي", "expected": "ar"},
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']} detection...")
            
            # Create transcribe config with auto-detection
            transcribe_config = WhisperXTranscribeConfig(
                language=None,  # Auto-detect
                task="transcribe"
            )
            
            # Generate synthetic audio for the language
            audio = create_synthetic_audio(test_case["content"], duration=3.0)
            
            # Save to file
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{test_case['name'].lower()}.wav", delete=False)
            audio_file = save_audio_to_file(audio, temp_file.name)
            
            try:
                # Transcribe with auto-detection
                result = await asr.transcribe_file(audio_file, transcribe_config)
                
                detected_lang = result.language
                print(f"   🎯 Expected: {test_case['expected']}, Detected: {detected_lang}")
                print(f"   📝 Text: {result.text or 'No speech detected (synthetic audio)'}")
                
            except Exception as e:
                print(f"   ❌ Detection failed: {e}")
            finally:
                # Clean up
                Path(audio_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Language detection demo failed: {e}")
        return False

async def show_supported_languages():
    """Display all supported languages in WhisperX."""
    print("\n📚 WhisperX Supported Languages")
    print("=" * 50)
    
    try:
        from whisper_streaming.backend import WhisperXASR, WhisperXModelConfig
        
        # Create a basic ASR instance to get language info
        config = WhisperXModelConfig(model_name="base", device="cpu")
        asr = WhisperXASR(config, sample_rate=16000, language="en")
        
        supported_languages = asr.get_supported_languages()
        
        print(f"📊 Total Supported Languages: {len(supported_languages)}")
        print("\n🌍 Language List:")
        
        # Group languages by region for better display
        middle_eastern = ["ar", "fa", "ur", "he", "tr"]
        european = ["en", "de", "fr", "es", "it", "pt", "ru", "nl", "pl", "sv", "da", "no", "fi"]
        asian = ["zh", "ja", "ko", "hi", "th", "vi", "ms", "id"]
        
        print("\n  🏛️  Middle Eastern & Turkish:")
        for code in middle_eastern:
            if code in supported_languages:
                print(f"    {code}: {supported_languages[code]}")
        
        print("\n  🏰 European:")
        for code in european:
            if code in supported_languages:
                print(f"    {code}: {supported_languages[code]}")
        
        print("\n  🏯 Asian:")
        for code in asian:
            if code in supported_languages:
                print(f"    {code}: {supported_languages[code]}")
        
        print("\n  🌍 Other Languages:")
        other_langs = [code for code in supported_languages.keys() 
                      if code not in middle_eastern + european + asian]
        for code in sorted(other_langs):
            print(f"    {code}: {supported_languages[code]}")
        
        # Highlight Turkish and Arabic
        print(f"\n🎯 Featured Languages:")
        print(f"   🇹🇷 Turkish (tr): {supported_languages.get('tr', 'Not found')}")
        print(f"   🇸🇦 Arabic (ar): {supported_languages.get('ar', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to show supported languages: {e}")
        return False

async def main():
    """Run all multilingual demonstration examples."""
    print("🌐 WhisperX Multilingual Support Demonstration")
    print("=" * 60)
    print("Showcasing Turkish and Arabic language support with enhanced features")
    
    # Check if WhisperX is available
    try:
        from whisper_streaming.backend import WhisperXASR
        print("✅ WhisperX backend available")
    except ImportError:
        print("❌ WhisperX backend not available. Please install whisper_streaming package.")
        return 1
    
    results = []
    
    # Run all demonstrations
    demos = [
        ("Supported Languages", show_supported_languages),
        ("Turkish Transcription", demonstrate_turkish_transcription),
        ("Arabic Transcription", demonstrate_arabic_transcription),
        ("Language Detection", demonstrate_language_detection),
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🚀 Running {demo_name} demo...")
            result = await demo_func()
            results.append((demo_name, result))
        except Exception as e:
            print(f"💥 {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n📊 Demo Results:")
    print("=" * 60)
    
    passed = 0
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} {demo_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(results)} demos completed successfully")
    
    if passed == len(results):
        print("🎉 All multilingual demos completed successfully!")
        print("\n💡 Next Steps:")
        print("   • Test with real Turkish and Arabic audio files")
        print("   • Experiment with speaker diarization in multilingual contexts")
        print("   • Try mixed-language transcription scenarios")
        print("   • Explore translation capabilities (transcribe → translate)")
        return 0
    else:
        print("⚠️  Some demos failed. This may be due to missing dependencies.")
        print("   Install WhisperX: pip install whisperx")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
