#!/usr/bin/env python3
"""
Quick Voice Cloning Script

Record 15 seconds, transcribe, and immediately test voice cloning with F5-TTS.
"""

import logging
import sys
import time
import tempfile
from pathlib import Path

import pyaudio
import wave

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from whisper_streaming.tts import TTSEngine, TTSConfig, create_tts_engine
    from f5_tts.api import F5TTS
except ImportError as e:
    logger.error(f"Required modules not found: {e}")
    logger.error("Install with: pip install f5-tts")
    sys.exit(1)


def record_voice(duration=15.0, sample_rate=24000):
    """Record audio from microphone."""
    
    print(f"🎙️  Recording {duration} seconds of your voice...")
    print("Speak clearly and naturally. This will be your voice template.")
    print("\nRecording starts in:")
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("🔴 RECORDING NOW!")
    
    # Audio settings
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    # Initialize audio
    audio = pyaudio.PyAudio()
    
    stream = audio.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )
    
    frames = []
    total_chunks = int(sample_rate / chunk * duration)
    
    # Record audio
    for i in range(total_chunks):
        data = stream.read(chunk)
        frames.append(data)
        
        # Progress
        if i % (total_chunks // 5) == 0:
            progress = (i / total_chunks) * 100
            print(f"  Recording... {progress:.0f}%")
    
    print("✅ Recording complete!")
    
    # Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = Path(temp_file.name)
    temp_file.close()
    
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print(f"📁 Audio saved: {output_path}")
    return output_path


def transcribe_audio(audio_path):
    """Transcribe audio using F5-TTS."""
    
    print("🤖 Transcribing your recording...")
    
    try:
        f5_model = F5TTS()
        transcription = f5_model.transcribe(str(audio_path))
        
        if transcription:
            print(f"✅ Transcription: '{transcription}'")
            return transcription
        else:
            print("⚠️  Auto-transcription failed")
            return None
            
    except Exception as e:
        print(f"⚠️  Transcription error: {e}")
        return None


def setup_voice_cloning(ref_audio, ref_text, language="en"):
    """Create voice cloning configuration."""
    
    config = TTSConfig(
        language=language,
        f5_model="F5TTS_v1_Base",
        f5_ref_audio=str(ref_audio),
        f5_ref_text=ref_text,
        f5_device="auto",
        f5_seed=42,
        speed=1.0
    )
    
    print("⚙️  Voice cloning configured!")
    return config


def test_voice_clone(config, text):
    """Generate speech with cloned voice."""
    
    print(f"🔄 Generating: '{text}'")
    
    try:
        # Create TTS engine
        tts_engine = create_tts_engine(TTSEngine.F5_TTS, config)
        
        # Generate audio
        temp_file = tempfile.NamedTemporaryFile(suffix="_cloned.wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
        
        result_path = tts_engine.synthesize(text, output_path)
        
        print(f"✅ Generated: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None


def play_audio_macos(audio_path):
    """Play audio on macOS."""
    try:
        import subprocess
        subprocess.run(["afplay", str(audio_path)], check=True)
        return True
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")
        return False


def main():
    """Main voice cloning workflow."""
    
    print("🎯 Quick Voice Cloning with F5-TTS")
    print("=" * 35)
    print("This will:")
    print("1. Record 15 seconds of your voice")
    print("2. Transcribe the recording")
    print("3. Set up voice cloning")
    print("4. Test with sample text")
    print()
    
    try:
        # Step 1: Record voice
        input("Press Enter to start recording...")
        ref_audio = record_voice(duration=15.0)
        
        # Step 2: Transcribe
        ref_text = transcribe_audio(ref_audio)
        
        if not ref_text:
            print("\n📝 Please type what you said:")
            ref_text = input("Your text: ").strip()
            
            if not ref_text:
                print("❌ Reference text is required!")
                return 1
        
        print(f"\n✅ Reference text: '{ref_text}'")
        
        # Step 3: Language selection
        print("\n🌍 Select language:")
        print("1. English")
        print("2. Turkish") 
        print("3. Other")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "2":
            language = "tr"
            test_texts = [
                "Merhaba, bu benim klonlanmış sesim.",
                "Türkçe konuşma sentezi çok başarılı.",
                "F5-TTS gerçekten harika çalışıyor."
            ]
        elif choice == "3":
            language = input("Language code (e.g., 'es', 'fr'): ").strip() or "en"
            test_texts = ["This is my cloned voice speaking."]
        else:
            language = "en"
            test_texts = [
                "Hello, this is my cloned voice speaking.",
                "The voice cloning quality is amazing.",
                "F5-TTS works incredibly well."
            ]
        
        # Step 4: Setup voice cloning
        config = setup_voice_cloning(ref_audio, ref_text, language)
        
        # Step 5: Test with sample texts
        print(f"\n🎯 Testing voice cloning in {language}...")
        
        for i, test_text in enumerate(test_texts, 1):
            print(f"\n--- Test {i} ---")
            result_path = test_voice_clone(config, test_text)
            
            if result_path:
                play_choice = input("Play this audio? (y/n): ").lower()
                if play_choice.startswith('y'):
                    play_audio_macos(result_path)
        
        # Step 6: Interactive testing
        print("\n🎮 Interactive Mode")
        print("Enter your own text to test (or 'quit' to exit):")
        
        while True:
            user_text = input("\nText: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q', '']:
                break
            
            result_path = test_voice_clone(config, user_text)
            if result_path:
                play_choice = input("Play? (y/n): ").lower()
                if play_choice.startswith('y'):
                    play_audio_macos(result_path)
        
        print("\n🎉 Voice cloning session complete!")
        print(f"Reference audio: {ref_audio}")
        print(f"Reference text: '{ref_text}'")
        print("You can use these files for future voice cloning.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Session failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
