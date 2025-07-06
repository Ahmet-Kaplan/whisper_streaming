#!/usr/bin/env python3
"""
Voice Cloning Recorder for F5-TTS

Records 15 seconds of audio, transcribes it, and uses it for voice cloning
with F5-TTS in the whisper_streaming project.
"""

import logging
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pyaudio
import wave
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from whisper_streaming.tts import (
        TTSEngine, 
        TTSConfig, 
        create_tts_engine,
        F5TTS
    )
    from f5_tts.api import F5TTS as F5TTSModel
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure whisper_streaming and f5-tts are installed")
    sys.exit(1)


class VoiceCloningRecorder:
    """Records audio and sets up voice cloning with F5-TTS."""
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1, chunk_size: int = 1024):
        """Initialize the voice cloning recorder.
        
        Args:
            sample_rate: Audio sample rate (F5-TTS prefers 24kHz)
            channels: Number of audio channels (mono = 1)
            chunk_size: Audio buffer chunk size
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        logger.info(f"Audio recorder initialized: {sample_rate}Hz, {channels} channel(s)")
    
    def record_audio(self, duration: float = 15.0, output_path: Optional[Path] = None) -> Path:
        """Record audio for the specified duration.
        
        Args:
            duration: Recording duration in seconds
            output_path: Optional output file path
            
        Returns:
            Path to the recorded audio file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        
        logger.info(f"Starting audio recording for {duration} seconds...")
        logger.info("Speak clearly and naturally. Recording will start in 3 seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("🔴 RECORDING NOW - Speak clearly!")
        
        # Open audio stream
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        total_frames = int(self.sample_rate / self.chunk_size * duration)
        
        try:
            for i in range(total_frames):
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # Progress indicator
                progress = (i + 1) / total_frames
                if i % (total_frames // 10) == 0:  # Show progress every 10%
                    print(f"Recording... {progress*100:.0f}%")
        
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        
        finally:
            stream.stop_stream()
            stream.close()
        
        print("✅ Recording completed!")
        
        # Save the recording
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"Audio saved to: {output_path}")
        
        # Display audio info
        file_size = output_path.stat().st_size
        logger.info(f"File size: {file_size / 1024:.1f} KB")
        logger.info(f"Duration: {duration:.1f} seconds")
        
        return output_path
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe the recorded audio using F5-TTS's built-in transcription.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        logger.info("Transcribing audio...")
        
        try:
            # Use F5-TTS's transcription capability
            f5_model = F5TTSModel()
            transcription = f5_model.transcribe(str(audio_path))
            
            logger.info(f"Transcription: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            logger.info("Please provide the text manually")
            return ""
    
    def create_voice_clone_config(
        self, 
        ref_audio_path: Path, 
        ref_text: str,
        language: str = "en"
    ) -> TTSConfig:
        """Create a TTS configuration for voice cloning.
        
        Args:
            ref_audio_path: Path to reference audio
            ref_text: Reference text
            language: Target language
            
        Returns:
            Configured TTSConfig for voice cloning
        """
        config = TTSConfig(
            language=language,
            f5_model="F5TTS_v1_Base",
            f5_ref_audio=str(ref_audio_path),
            f5_ref_text=ref_text,
            f5_device="auto",  # Use GPU if available
            f5_seed=42,
            speed=1.0,
            sample_rate=self.sample_rate
        )
        
        logger.info("Voice cloning configuration created")
        return config
    
    def test_voice_clone(
        self, 
        config: TTSConfig, 
        test_text: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """Test the voice cloning with a sample text.
        
        Args:
            config: Voice cloning configuration
            test_text: Text to synthesize with cloned voice
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        logger.info(f"Testing voice clone with text: '{test_text}'")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix="_cloned.wav", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        
        try:
            # Create F5-TTS engine with voice cloning config
            tts_engine = create_tts_engine(TTSEngine.F5_TTS, config)
            
            # Generate speech with cloned voice
            result_path = tts_engine.synthesize(test_text, output_path)
            
            logger.info(f"Voice cloned audio generated: {result_path}")
            file_size = result_path.stat().st_size
            logger.info(f"Generated file size: {file_size / 1024:.1f} KB")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


def interactive_voice_cloning():
    """Interactive voice cloning session."""
    
    print("🎙️  F5-TTS Voice Cloning Recorder")
    print("=" * 40)
    
    recorder = VoiceCloningRecorder()
    
    try:
        # Step 1: Record reference audio
        print("\n📹 Step 1: Record Reference Audio")
        print("You will record 15 seconds of your voice.")
        print("Speak clearly and naturally - this will be your voice template.")
        
        input("Press Enter when ready to record...")
        
        ref_audio_path = recorder.record_audio(duration=15.0)
        
        # Step 2: Transcription
        print("\n📝 Step 2: Get Reference Text")
        print("We need the exact text that you spoke in the recording.")
        
        # Try automatic transcription first
        auto_transcription = recorder.transcribe_audio(ref_audio_path)
        
        if auto_transcription:
            print(f"\n🤖 Auto-transcription: '{auto_transcription}'")
            use_auto = input("Use this transcription? (y/n): ").lower().startswith('y')
            ref_text = auto_transcription if use_auto else ""
        else:
            ref_text = ""
        
        # Manual input if needed
        if not ref_text:
            print("\n✍️  Please type the exact text you spoke:")
            ref_text = input("Reference text: ").strip()
        
        if not ref_text:
            print("❌ Reference text is required for voice cloning")
            return
        
        # Step 3: Language selection
        print("\n🌍 Step 3: Select Language")
        print("Available languages:")
        print("1. English (en)")
        print("2. Turkish (tr)")
        print("3. Other (specify)")
        
        lang_choice = input("Select language (1-3): ").strip()
        if lang_choice == "2":
            language = "tr"
        elif lang_choice == "3":
            language = input("Enter language code (e.g., 'es', 'fr', 'de'): ").strip()
        else:
            language = "en"
        
        # Step 4: Create voice cloning configuration
        print(f"\n⚙️  Step 4: Setup Voice Cloning (Language: {language})")
        config = recorder.create_voice_clone_config(ref_audio_path, ref_text, language)
        
        print("✅ Voice cloning setup complete!")
        print(f"Reference audio: {ref_audio_path}")
        print(f"Reference text: '{ref_text}'")
        
        # Step 5: Test voice cloning
        while True:
            print("\n🎯 Step 5: Test Voice Cloning")
            print("Enter text to synthesize with your cloned voice (or 'quit' to exit):")
            
            test_text = input("Text to synthesize: ").strip()
            
            if test_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not test_text:
                continue
            
            try:
                print("\n🔄 Generating speech with your cloned voice...")
                cloned_audio_path = recorder.test_voice_clone(config, test_text)
                
                print(f"✅ Success! Cloned voice audio saved to: {cloned_audio_path}")
                
                # Option to play the audio
                play_audio = input("Play the generated audio? (y/n): ").lower().startswith('y')
                if play_audio:
                    try:
                        import subprocess
                        import platform
                        
                        if platform.system() == "Darwin":  # macOS
                            subprocess.run(["afplay", str(cloned_audio_path)])
                        elif platform.system() == "Linux":
                            subprocess.run(["aplay", str(cloned_audio_path)])
                        elif platform.system() == "Windows":
                            subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{cloned_audio_path}').PlaySync()"])
                    except Exception as e:
                        logger.warning(f"Could not play audio: {e}")
                
            except Exception as e:
                logger.error(f"Voice cloning test failed: {e}")
                continue
        
        print("\n🎉 Voice cloning session completed!")
        print("You can now use the voice cloning configuration in your applications.")
        
        # Save configuration for reuse
        save_config = input("\nSave configuration for reuse? (y/n): ").lower().startswith('y')
        if save_config:
            config_name = input("Enter configuration name: ").strip() or "my_voice"
            config_file = Path(f"voice_config_{config_name}.txt")
            
            with open(config_file, 'w') as f:
                f.write(f"ref_audio_path={ref_audio_path}\n")
                f.write(f"ref_text={ref_text}\n")
                f.write(f"language={language}\n")
            
            print(f"✅ Configuration saved to: {config_file}")
    
    except KeyboardInterrupt:
        print("\n\n❌ Session interrupted by user")
    except Exception as e:
        logger.error(f"Session failed: {e}")
    finally:
        recorder.cleanup()


def main():
    """Main function."""
    
    print("F5-TTS Voice Cloning Recorder")
    print("============================")
    print("This tool helps you record your voice and set up voice cloning with F5-TTS")
    print()
    
    # Check audio system
    try:
        audio = pyaudio.PyAudio()
        input_devices = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
        audio.terminate()
        
        if not input_devices:
            print("❌ No audio input devices found!")
            return 1
        
        print(f"✅ Found {len(input_devices)} audio input device(s)")
        
    except Exception as e:
        logger.error(f"Audio system check failed: {e}")
        return 1
    
    # Start interactive session
    try:
        interactive_voice_cloning()
        return 0
    except Exception as e:
        logger.error(f"Voice cloning session failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
