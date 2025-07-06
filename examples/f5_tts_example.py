#!/usr/bin/env python3
"""
F5-TTS Integration Example for Whisper Streaming

This example demonstrates how to use F5-TTS as a high-quality TTS engine
in the whisper_streaming project for Turkish text synthesis.

F5-TTS offers state-of-the-art quality with voice cloning capabilities.
"""

import logging
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from whisper_streaming.tts import (
        TTSEngine, 
        TTSConfig, 
        create_tts_engine,
        get_available_engines,
        get_best_tts_for_turkish,
        synthesize_turkish
    )
except ImportError as e:
    logger.error(f"Failed to import whisper_streaming TTS module: {e}")
    exit(1)


def demonstrate_f5_tts():
    """Demonstrate F5-TTS functionality."""
    
    logger.info("=== F5-TTS Integration Demo ===")
    
    # Check available engines
    available = get_available_engines()
    logger.info(f"Available TTS engines: {[engine.value for engine in available]}")
    
    # Check if F5-TTS is available
    if TTSEngine.F5_TTS not in available:
        logger.warning("F5-TTS is not available. Please ensure f5-tts is installed.")
        logger.info("You can install it with: pip install f5-tts")
        return
    
    # Get best TTS recommendation
    best_engine, reason = get_best_tts_for_turkish(prefer_offline=True)
    logger.info(f"Best TTS for Turkish: {best_engine.value} - {reason}")
    
    # Test Turkish text
    turkish_text = """
    Merhaba! Bu, F5-TTS kullanarak Türkçe konuşma sentezi örneğidir. 
    F5-TTS son teknoloji ses klonlama yetenekleri sunar ve yüksek kaliteli 
    doğal ses üretimi sağlar.
    """
    
    logger.info(f"Text to synthesize: {turkish_text.strip()}")
    
    # Example 1: Basic F5-TTS usage with default configuration
    logger.info("\n--- Example 1: Basic F5-TTS Usage ---")
    try:
        config = TTSConfig(
            language="tr",
            speed=1.0,
            f5_model="F5TTS_v1_Base",
            f5_device="auto",  # Will use GPU if available, otherwise CPU
            f5_seed=42
        )
        
        f5_tts = create_tts_engine(TTSEngine.F5_TTS, config)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
        
        result_path = f5_tts.synthesize(turkish_text, output_path)
        logger.info(f"F5-TTS synthesis completed: {result_path}")
        logger.info(f"File size: {result_path.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"F5-TTS synthesis failed: {e}")
    
    # Example 2: Using the convenience function
    logger.info("\n--- Example 2: Convenience Function ---")
    try:
        result_path = synthesize_turkish(
            text=turkish_text,
            engine=TTSEngine.F5_TTS,
            speed=1.2  # Slightly faster speech
        )
        logger.info(f"Quick synthesis completed: {result_path}")
        logger.info(f"File size: {result_path.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"Quick synthesis failed: {e}")
    
    # Example 3: Auto-selection (should prefer F5-TTS if available)
    logger.info("\n--- Example 3: Auto Engine Selection ---")
    try:
        result_path = synthesize_turkish(
            text="Bu otomatik motor seçimi kullanılarak oluşturulan bir örnektir.",
            engine=TTSEngine.AUTO
        )
        logger.info(f"Auto-selected TTS synthesis completed: {result_path}")
        
    except Exception as e:
        logger.error(f"Auto synthesis failed: {e}")


def demonstrate_voice_cloning():
    """Demonstrate F5-TTS voice cloning capabilities."""
    
    logger.info("\n=== F5-TTS Voice Cloning Demo ===")
    
    # Check if F5-TTS is available
    available = get_available_engines()
    if TTSEngine.F5_TTS not in available:
        logger.warning("F5-TTS not available for voice cloning demo")
        return
    
    logger.info("Note: Voice cloning requires reference audio and text.")
    logger.info("This example shows configuration for voice cloning.")
    logger.info("To use voice cloning, provide:")
    logger.info("- f5_ref_audio: Path to reference audio file (3-10 seconds)")
    logger.info("- f5_ref_text: Exact text spoken in the reference audio")
    
    # Example configuration for voice cloning
    cloning_config = TTSConfig(
        language="tr",
        f5_model="F5TTS_v1_Base",
        f5_ref_audio=None,  # Set to your reference audio path
        f5_ref_text=None,   # Set to your reference text
        f5_device="auto",
        f5_seed=42,
        speed=1.0
    )
    
    logger.info("Voice cloning configuration created (reference files not provided)")
    logger.info("To enable voice cloning:")
    logger.info("1. Record 3-10 seconds of clear Turkish speech")
    logger.info("2. Set f5_ref_audio to the audio file path")
    logger.info("3. Set f5_ref_text to the exact spoken text")


def main():
    """Main demonstration function."""
    
    print("F5-TTS Integration Example for Whisper Streaming")
    print("=" * 50)
    
    try:
        demonstrate_f5_tts()
        demonstrate_voice_cloning()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("F5-TTS is now integrated into whisper_streaming")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
