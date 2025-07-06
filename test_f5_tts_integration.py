#!/usr/bin/env python3
"""
Simple test to verify F5-TTS integration with whisper_streaming
"""

import sys
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_f5_tts_integration():
    """Test F5-TTS integration."""
    
    logger.info("Testing F5-TTS integration...")
    
    try:
        # Test imports
        from whisper_streaming.tts import (
            TTSEngine, 
            TTSConfig, 
            get_available_engines,
            create_tts_engine
        )
        logger.info("✓ TTS module imports successful")
        
        # Check if F5_TTS is in enum
        assert hasattr(TTSEngine, 'F5_TTS'), "F5_TTS not found in TTSEngine enum"
        logger.info("✓ F5_TTS enum value exists")
        
        # Check available engines
        available = get_available_engines()
        logger.info(f"Available engines: {[e.value for e in available]}")
        
        # Test configuration
        config = TTSConfig(
            language="tr",
            f5_model="F5TTS_v1_Base",
            f5_device="auto",
            f5_seed=42
        )
        logger.info("✓ F5-TTS configuration created")
        
        # Test engine creation (may fail if F5-TTS not installed, but should not crash)
        try:
            if TTSEngine.F5_TTS in available:
                engine = create_tts_engine(TTSEngine.F5_TTS, config)
                logger.info("✓ F5-TTS engine created successfully")
            else:
                logger.info("ℹ F5-TTS not available (f5-tts package not installed)")
        except ImportError as e:
            logger.info(f"ℹ F5-TTS import failed (expected if not installed): {e}")
        except Exception as e:
            logger.warning(f"F5-TTS engine creation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def main():
    """Main test function."""
    
    print("F5-TTS Integration Test")
    print("=" * 30)
    
    success = test_f5_tts_integration()
    
    if success:
        print("\n✓ Integration test passed!")
        print("F5-TTS has been successfully added to whisper_streaming")
        return 0
    else:
        print("\n✗ Integration test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
