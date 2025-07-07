#!/usr/bin/env python3
"""
Test script for Kyutai TTS integration in whisper_streaming.

This script tests the newly integrated KyutaiTTS engine to ensure it works
correctly with the existing TTS architecture.
"""

import sys
import logging
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from whisper_streaming.tts import (
    TTSConfig,
    TTSEngine,
    KyutaiTTS,
    get_available_engines,
    get_best_tts_for_turkish,
    create_tts_engine,
    synthesize_turkish
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_availability():
    """Test if Kyutai TTS is available."""
    logger.info("=== Testing Kyutai TTS Availability ===")
    
    available_engines = get_available_engines()
    logger.info(f"Available TTS engines: {[e.value for e in available_engines]}")
    
    kyutai_available = TTSEngine.KYUTAI_TTS in available_engines
    logger.info(f"Kyutai TTS available: {kyutai_available}")
    
    if kyutai_available:
        logger.info("✅ Kyutai TTS is available for testing")
        return True
    else:
        logger.warning("❌ Kyutai TTS is not available. Dependencies might be missing.")
        return False

def test_engine_selection():
    """Test engine selection with Kyutai TTS."""
    logger.info("=== Testing Engine Selection ===")
    
    try:
        best_engine, reason = get_best_tts_for_turkish(prefer_offline=True)
        logger.info(f"Best offline TTS: {best_engine.value} - {reason}")
        
        best_engine, reason = get_best_tts_for_turkish(prefer_offline=False)
        logger.info(f"Best quality TTS: {best_engine.value} - {reason}")
        
        return True
    except Exception as e:
        logger.error(f"Engine selection failed: {e}")
        return False

def test_kyutai_config():
    """Test Kyutai TTS configuration."""
    logger.info("=== Testing Kyutai TTS Configuration ===")
    
    try:
        # Test default config
        config = TTSConfig()
        logger.info(f"Default Kyutai model repo: {config.kyutai_model_repo}")
        logger.info(f"Default Kyutai voice: {config.kyutai_voice}")
        logger.info(f"Default Kyutai device: {config.kyutai_device}")
        
        # Test custom config
        custom_config = TTSConfig(
            kyutai_model_repo="kyutai/tts-1.6b-en_fr",
            kyutai_voice="expresso/ex03-ex01_happy_001_channel1_334s.wav",
            kyutai_device="cpu",  # Use CPU for testing
            kyutai_streaming=False,  # Disable streaming for simpler testing
            kyutai_temp=0.8,
            kyutai_cfg_coef=1.5
        )
        logger.info(f"Custom config created successfully")
        
        return True, custom_config
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False, None

def test_kyutai_initialization(config):
    """Test Kyutai TTS initialization."""
    logger.info("=== Testing Kyutai TTS Initialization ===")
    
    try:
        # Test direct initialization
        kyutai_tts = KyutaiTTS(config)
        logger.info("✅ KyutaiTTS initialized successfully")
        
        # Test through factory function
        tts_engine = create_tts_engine(TTSEngine.KYUTAI_TTS, config)
        logger.info("✅ KyutaiTTS created via factory function")
        
        return True, kyutai_tts
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False, None

def test_kyutai_synthesis(kyutai_tts):
    """Test Kyutai TTS synthesis."""
    logger.info("=== Testing Kyutai TTS Synthesis ===")
    
    # Test texts
    test_texts = [
        "Hello, this is a test of Kyutai TTS synthesis.",
        "Bu bir Türkçe test metnidir.",  # Turkish text
        "This is a longer text to test the synthesis quality and performance of the Kyutai TTS system."
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        try:
            logger.info(f"Synthesizing text {i+1}: '{text[:50]}...'")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = Path(tmp_file.name)
            
            # Synthesize
            result_path = kyutai_tts.synthesize(text, output_path)
            
            # Check if file was created
            if result_path.exists() and result_path.stat().st_size > 0:
                logger.info(f"✅ Synthesis successful: {result_path} ({result_path.stat().st_size} bytes)")
                results.append((text, result_path, True))
            else:
                logger.warning(f"❌ Synthesis failed: Empty or missing file")
                results.append((text, result_path, False))
                
        except Exception as e:
            logger.error(f"❌ Synthesis failed for text {i+1}: {e}")
            results.append((text, None, False))
    
    return results

def test_convenience_function():
    """Test the convenience synthesis function."""
    logger.info("=== Testing Convenience Function ===")
    
    try:
        text = "Testing the convenience synthesis function."
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
        
        # This should auto-select the best engine (hopefully Kyutai if available)
        result_path = synthesize_turkish(text, output_path, engine=TTSEngine.AUTO)
        
        if result_path.exists() and result_path.stat().st_size > 0:
            logger.info(f"✅ Convenience function works: {result_path}")
            return True
        else:
            logger.warning("❌ Convenience function failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Convenience function test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Kyutai TTS Integration Tests")
    
    # Test 1: Availability
    if not test_availability():
        logger.error("Kyutai TTS not available. Install with: pip install moshi")
        return False
    
    # Test 2: Engine selection
    if not test_engine_selection():
        return False
    
    # Test 3: Configuration
    config_success, config = test_kyutai_config()
    if not config_success:
        return False
    
    # Test 4: Initialization
    init_success, kyutai_tts = test_kyutai_initialization(config)
    if not init_success:
        logger.error("Kyutai TTS initialization failed. Check dependencies and GPU availability.")
        return False
    
    # Test 5: Synthesis
    synthesis_results = test_kyutai_synthesis(kyutai_tts)
    successful_syntheses = sum(1 for _, _, success in synthesis_results if success)
    logger.info(f"Synthesis tests: {successful_syntheses}/{len(synthesis_results)} successful")
    
    # Test 6: Convenience function
    convenience_success = test_convenience_function()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"✅ Availability: Passed")
    logger.info(f"✅ Engine Selection: Passed") 
    logger.info(f"✅ Configuration: Passed")
    logger.info(f"✅ Initialization: Passed")
    logger.info(f"{'✅' if successful_syntheses > 0 else '❌'} Synthesis: {successful_syntheses}/{len(synthesis_results)} passed")
    logger.info(f"{'✅' if convenience_success else '❌'} Convenience Function: {'Passed' if convenience_success else 'Failed'}")
    
    overall_success = successful_syntheses > 0 and convenience_success
    
    if overall_success:
        logger.info("🎉 All critical tests passed! Kyutai TTS integration is working.")
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
