#!/usr/bin/env python3
"""Test script for Piper TTS integration."""

import logging
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from whisper_streaming.tts import (
    TTSEngine,
    TTSConfig,
    PiperTTS,
    get_available_engines,
    get_best_tts_for_turkish,
    synthesize_turkish,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_available_engines():
    """Test which TTS engines are available."""
    logger.info("Checking available TTS engines...")
    available = get_available_engines()
    logger.info(f"Available engines: {[e.value for e in available]}")
    
    if TTSEngine.PIPER_TTS in available:
        logger.info("✅ Piper TTS is available!")
        return True
    else:
        logger.warning("❌ Piper TTS is not available. Please install: pip install piper-tts")
        return False


def test_best_tts_selection():
    """Test the best TTS engine selection for Turkish."""
    logger.info("\nTesting TTS engine selection...")
    
    try:
        # Test offline preference
        engine, reason = get_best_tts_for_turkish(prefer_offline=True)
        logger.info(f"Best offline TTS: {engine.value} - {reason}")
        
        # Test online preference
        engine, reason = get_best_tts_for_turkish(prefer_offline=False)
        logger.info(f"Best online TTS: {engine.value} - {reason}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to get best TTS engine: {e}")
        return False


def test_piper_tts_creation():
    """Test creating a Piper TTS instance."""
    logger.info("\nTesting Piper TTS creation...")
    
    try:
        config = TTSConfig(
            language="tr",
            piper_model="dfki-medium",  # Use shorthand model name
            speed=1.0
        )
        
        piper_tts = PiperTTS(config)
        logger.info("✅ Piper TTS instance created successfully!")
        
        # Test if it's available
        if piper_tts.is_available():
            logger.info("✅ Piper TTS is available and ready to use!")
            return piper_tts
        else:
            logger.warning("❌ Piper TTS instance created but not available")
            return None
            
    except ImportError as e:
        logger.warning(f"❌ Piper TTS not available: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create Piper TTS: {e}")
        return None


def test_piper_synthesis(piper_tts):
    """Test Piper TTS synthesis."""
    logger.info("\nTesting Piper TTS synthesis...")
    
    test_text = "Merhaba! Bu bir Piper TTS testi. Türkçe metin sentezi çalışıyor mu?"
    
    try:
        # Test synthesis
        output_path = piper_tts.synthesize(test_text)
        logger.info(f"✅ Synthesis completed! Audio saved to: {output_path}")
        
        # Check if file exists and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✅ Audio file is valid (size: {output_path.stat().st_size} bytes)")
            return True
        else:
            logger.error("❌ Audio file is empty or doesn't exist")
            return False
            
    except Exception as e:
        logger.error(f"❌ Synthesis failed: {e}")
        return False


def test_convenience_function():
    """Test the convenience synthesis function."""
    logger.info("\nTesting convenience synthesis function...")
    
    test_text = "Bu bir kısa test cümlesidir."
    
    try:
        # Try with auto engine selection
        output_path = synthesize_turkish(
            text=test_text,
            engine=TTSEngine.AUTO,
            voice_preference="female",
            speed=1.1
        )
        
        logger.info(f"✅ Convenience function worked! Audio saved to: {output_path}")
        
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✅ Audio file is valid (size: {output_path.stat().st_size} bytes)")
            return True
        else:
            logger.error("❌ Audio file is empty or doesn't exist")
            return False
            
    except Exception as e:
        logger.error(f"❌ Convenience function failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=== Piper TTS Integration Test ===\n")
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Check available engines
    if test_available_engines():
        success_count += 1
    
    # Test 2: Test engine selection
    if test_best_tts_selection():
        success_count += 1
    
    # Test 3: Create Piper TTS instance
    piper_tts = test_piper_tts_creation()
    if piper_tts is not None:
        success_count += 1
        
        # Test 4: Test synthesis (only if Piper TTS is available)
        if test_piper_synthesis(piper_tts):
            success_count += 1
    else:
        logger.info("Skipping synthesis test - Piper TTS not available")
        total_tests = 3  # Adjust total test count
    
    # Test 5: Test convenience function (bonus test)
    logger.info("\n=== Bonus Test ===")
    if test_convenience_function():
        logger.info("✅ Bonus test passed!")
    
    # Summary
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        logger.info("🎉 All tests passed! Piper TTS integration is working correctly!")
        return 0
    else:
        logger.warning(f"⚠️  {total_tests - success_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
