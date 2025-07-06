#!/usr/bin/env python3
"""
WhisperX Backend Example

Demonstrates how to use the WhisperX backend with the whisper_streaming framework
for enhanced transcription with word-level timestamps and speaker diarization.
"""

import asyncio
import logging
import tempfile
import wave
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_audio(duration: float = 3.0, sample_rate: int = 16000) -> str:
    """Create a sample audio file for testing."""
    # Generate a simple sine wave with speech-like characteristics
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Multiple frequencies to simulate speech formants
    f1, f2, f3 = 200, 800, 1200  # Typical formant frequencies
    audio = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add envelope to make it more speech-like
    envelope = np.exp(-t/2) * (1 + 0.5 * np.sin(10 * t))
    audio = audio * envelope
    
    # Normalize and convert to 16-bit PCM
    audio = audio / np.max(np.abs(audio))
    audio_data = (audio * 32767 * 0.5).astype(np.int16)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return temp_file.name

async def example_basic_whisperx():
    """Example 1: Basic WhisperX backend usage."""
    print("🎯 Example 1: Basic WhisperX Backend")
    print("=" * 50)
    
    try:
        from whisper_streaming import Backend
        from whisper_streaming.backend import (
            WhisperXASR, WhisperXModelConfig, WhisperXTranscribeConfig, 
            WhisperXFeatureExtractorConfig, create_whisperx_asr
        )
        
        print("✅ WhisperX backend is available")
        
        # Create sample audio
        print("🎵 Creating sample audio...")
        audio_file = create_sample_audio()
        
        try:
            # Method 1: Using the convenience function
            print("🔧 Creating WhisperX ASR using convenience function...")
            asr = create_whisperx_asr(
                model_name="tiny",  # Use smallest model for demo
                device="cpu",       # Force CPU for compatibility
                enable_diarization=False,  # Disable for basic example
                enable_alignment=True,     # Enable word-level timestamps
                sample_rate=16000,
                language="en"
            )
            
            print(f"📊 Model info: {asr.get_model_info()}")
            
            # Test transcription
            print("🔄 Testing transcription...")
            segments, language = asr.transcribe(audio_file, "")
            
            print(f"✅ Transcription completed!")
            print(f"   Detected language: {language}")
            print(f"   Segments: {len(segments)}")
            
            for i, segment in enumerate(segments):
                print(f"   Segment {i+1}: [{segment.start:.2f}s - {segment.end:.2f}s]")
                if hasattr(segment, 'words') and segment.words:
                    print(f"   Words: {len(segment.words)}")
                    for word in segment.words[:3]:  # Show first 3 words
                        print(f"     • \"{word.word}\": {word.start:.2f}s - {word.end:.2f}s")
            
        except Exception as e:
            print(f"⚠️  Transcription test skipped: {e}")
            print("   (This is expected with synthetic audio)")
        
        # Clean up
        Path(audio_file).unlink()
        
    except ImportError as e:
        print(f"❌ WhisperX backend not available: {e}")
        print("   Install with: pip install whisperx")

async def example_whisperx_processor():
    """Example 2: Using WhisperX with ASRProcessor."""
    print("\n🎯 Example 2: WhisperX with ASRProcessor")
    print("=" * 50)
    
    try:
        from whisper_streaming import Backend, ASRProcessor
        from whisper_streaming.backend import (
            WhisperXModelConfig, WhisperXTranscribeConfig, WhisperXFeatureExtractorConfig
        )
        from whisper_streaming.receiver import FileReceiver
        from whisper_streaming.sender import PrintSender
        
        print("✅ Setting up ASRProcessor with WhisperX backend...")
        
        # Create sample audio
        audio_file = create_sample_audio()
        
        try:
            # Configure WhisperX
            model_config = WhisperXModelConfig(
                model_name="tiny",
                device="cpu",
                enable_diarization=False,
                enable_alignment=True
            )
            
            transcribe_config = WhisperXTranscribeConfig(
                language="en"
            )
            
            feature_config = WhisperXFeatureExtractorConfig()
            
            # Configure processor
            processor_config = ASRProcessor.ProcessorConfig(
                sampling_rate=16000,
                prompt_size=100,
                audio_receiver_timeout=1.0,
                language="en"
            )
            
            # Setup components
            audio_receiver = FileReceiver(
                path=audio_file,
                chunk_size=1.0,  # 1 second chunks
                target_sample_rate=16000
            )
            
            output_sender = PrintSender()
            
            # Create processor with WhisperX backend
            processor = ASRProcessor(
                processor_config=processor_config,
                audio_receiver=audio_receiver,
                output_senders=output_sender,
                backend=Backend.WHISPERX,
                model_config=model_config,
                transcribe_config=transcribe_config,
                feature_extractor_config=feature_config
            )
            
            print("🔄 Running ASRProcessor with WhisperX...")
            print("   (Processing will begin shortly)")
            
            # Note: processor.run() is synchronous and will process the audio
            # For this example, we'll just show the setup
            print("✅ ASRProcessor configured successfully with WhisperX backend")
            
        except Exception as e:
            print(f"⚠️  ASRProcessor test skipped: {e}")
        
        # Clean up
        Path(audio_file).unlink()
        
    except ImportError as e:
        print(f"❌ WhisperX components not available: {e}")

async def example_enhanced_features():
    """Example 3: Demonstrate enhanced WhisperX features."""
    print("\n🎯 Example 3: Enhanced WhisperX Features")
    print("=" * 50)
    
    try:
        from whisper_streaming.backend import WhisperXModelConfig, create_whisperx_asr
        
        print("🔧 Configuration Examples:")
        
        # Basic configuration
        print("\n📝 Basic Configuration:")
        basic_config = WhisperXModelConfig(
            model_name="base",
            enable_alignment=True,
            enable_diarization=False
        )
        print(f"   Model: {basic_config.model_name}")
        print(f"   Alignment: {basic_config.enable_alignment}")
        print(f"   Diarization: {basic_config.enable_diarization}")
        
        # Advanced configuration
        print("\n🚀 Advanced Configuration:")
        advanced_config = WhisperXModelConfig(
            model_name="large-v2",
            device="cuda",  # GPU if available
            compute_type="float16",
            batch_size=32,
            enable_diarization=True,
            enable_alignment=True,
            min_speakers=2,
            max_speakers=5,
            huggingface_token="your_hf_token_here"
        )
        print(f"   Model: {advanced_config.model_name}")
        print(f"   Device: {advanced_config.device}")
        print(f"   Compute Type: {advanced_config.compute_type}")
        print(f"   Batch Size: {advanced_config.batch_size}")
        print(f"   Speaker Range: {advanced_config.min_speakers}-{advanced_config.max_speakers}")
        
        print("\n✨ Enhanced Features Available:")
        print("   • Word-level timestamps (phoneme precision)")
        print("   • Speaker diarization (who spoke when)")
        print("   • Force alignment for better timing")
        print("   • Batch processing optimization")
        print("   • GPU acceleration support")
        print("   • Multiple model sizes")
        
    except ImportError as e:
        print(f"❌ WhisperX configuration not available: {e}")

async def example_comparison():
    """Example 4: Compare WhisperX with Faster-Whisper."""
    print("\n🔍 Example 4: Backend Comparison")
    print("=" * 50)
    
    print("📊 Feature Comparison:")
    print("""
    ┌─────────────────────┬─────────────────┬─────────────────┐
    │ Feature             │ Faster-Whisper  │ WhisperX        │
    ├─────────────────────┼─────────────────┼─────────────────┤
    │ Transcription       │ ✅ Excellent    │ ✅ Excellent    │
    │ Word Timestamps     │ ⚠️ Segment-level │ ✅ Word-level   │
    │ Speaker ID          │ ❌ No           │ ✅ Yes          │
    │ Force Alignment     │ ❌ No           │ ✅ Yes          │
    │ Streaming Support   │ ✅ Yes          │ ⚠️ Batch-first  │
    │ Memory Usage        │ 💾 Lower        │ 💾 Higher      │
    │ Processing Speed    │ 🔥 Fast         │ ⚡ Very Fast   │
    └─────────────────────┴─────────────────┴─────────────────┘
    """)
    
    print("🎯 Use Case Recommendations:")
    print("\n   Faster-Whisper Backend:")
    print("   • Real-time streaming applications")
    print("   • Live conversations")
    print("   • Memory-constrained environments")
    print("   • Simple transcription needs")
    
    print("\n   WhisperX Backend:")
    print("   • High-accuracy batch processing")
    print("   • Meeting transcriptions")
    print("   • Multi-speaker scenarios")
    print("   • Research applications")
    print("   • Subtitle generation")

async def main():
    """Run all examples."""
    print("🧪 WhisperX Backend Integration Examples")
    print("=" * 60)
    print("Demonstrating WhisperX backend integration with whisper_streaming")
    
    await example_basic_whisperx()
    await example_whisperx_processor()
    await example_enhanced_features()
    await example_comparison()
    
    print("\n🎉 Examples completed!")
    print("\n💡 Next Steps:")
    print("   1. Install WhisperX: pip install whisperx")
    print("   2. Try the examples with real audio files")
    print("   3. Explore speaker diarization features")
    print("   4. Integrate into your applications")
    print("   5. Compare performance with Faster-Whisper")

if __name__ == "__main__":
    asyncio.run(main())
