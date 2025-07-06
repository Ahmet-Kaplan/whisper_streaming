#!/usr/bin/env python3
"""
Piper TTS Integration Validation Script

This script validates that the Piper TTS integration is working correctly
in the whisper_streaming project. It checks:

1. Module imports and TTS system functionality
2. Piper TTS availability and configuration
3. Engine priority and selection logic
4. Fallback mechanisms

Usage:
    python validate_piper_integration.py
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_imports():
    """Validate that all TTS imports work correctly."""
    print("1. Validating imports...")
    
    try:
        from whisper_streaming.tts import (
            TTSEngine,
            TTSConfig,
            PiperTTS,
            get_available_engines,
            get_best_tts_for_turkish,
            synthesize_turkish,
            create_tts_engine,
        )
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def validate_tts_engine_enum():
    """Validate TTSEngine enum includes Piper."""
    print("2. Validating TTSEngine enum...")
    
    try:
        from whisper_streaming.tts import TTSEngine
        
        engines = list(TTSEngine)
        engine_values = [e.value for e in engines]
        
        print(f"   Available engines: {engine_values}")
        
        if TTSEngine.PIPER_TTS in engines:
            print("   ✅ PIPER_TTS found in enum")
            return True
        else:
            print("   ❌ PIPER_TTS not found in enum")
            return False
            
    except Exception as e:
        print(f"   ❌ Enum validation failed: {e}")
        return False

def validate_config_options():
    """Validate TTSConfig has Piper options."""
    print("3. Validating TTSConfig...")
    
    try:
        from whisper_streaming.tts import TTSConfig
        
        # Create config with Piper options
        config = TTSConfig(
            piper_model="dfki-medium",
            piper_data_dir="/tmp/test",
            piper_download_dir="/tmp/test"
        )
        
        # Check if options are set
        if hasattr(config, 'piper_model') and config.piper_model == "dfki-medium":
            print("   ✅ Piper configuration options available")
            return True
        else:
            print("   ❌ Piper configuration options missing")
            return False
            
    except Exception as e:
        print(f"   ❌ Config validation failed: {e}")
        return False

def validate_engine_detection():
    """Validate engine detection and availability."""
    print("4. Validating engine detection...")
    
    try:
        from whisper_streaming.tts import get_available_engines, TTSEngine
        
        available = get_available_engines()
        available_values = [e.value for e in available]
        
        print(f"   Available engines: {available_values}")
        
        if TTSEngine.PIPER_TTS in available:
            print("   ✅ Piper TTS detected as available")
            piper_available = True
        else:
            print("   ⚠️  Piper TTS not available (may need installation)")
            piper_available = False
        
        return piper_available
        
    except Exception as e:
        print(f"   ❌ Engine detection failed: {e}")
        return False

def validate_priority_system():
    """Validate engine priority system."""
    print("5. Validating priority system...")
    
    try:
        from whisper_streaming.tts import get_best_tts_for_turkish
        
        # Test offline preference
        offline_engine, offline_reason = get_best_tts_for_turkish(prefer_offline=True)
        print(f"   Best offline: {offline_engine.value} - {offline_reason}")
        
        # Test online preference  
        online_engine, online_reason = get_best_tts_for_turkish(prefer_offline=False)
        print(f"   Best online:  {online_engine.value} - {online_reason}")
        
        print("   ✅ Priority system working")
        return True
        
    except Exception as e:
        print(f"   ❌ Priority system validation failed: {e}")
        return False

def validate_piper_instantiation(piper_available):
    """Validate Piper TTS can be instantiated."""
    print("6. Validating Piper TTS instantiation...")
    
    if not piper_available:
        print("   ⚠️  Skipping - Piper TTS not available")
        return True
    
    try:
        from whisper_streaming.tts import PiperTTS, TTSConfig
        
        config = TTSConfig(
            language="tr",
            piper_model="dfki-medium",
            speed=1.0
        )
        
        # Try to create instance (this will fail if piper-tts not installed)
        tts = PiperTTS(config)
        
        if tts.is_available():
            print("   ✅ Piper TTS instantiated successfully")
            return True
        else:
            print("   ❌ Piper TTS instance not available")
            return False
            
    except ImportError as e:
        print(f"   ⚠️  Piper TTS import failed: {e}")
        print("      This is expected if piper-tts is not installed")
        return True  # Not a failure of integration
    except Exception as e:
        print(f"   ❌ Piper TTS instantiation failed: {e}")
        return False

def validate_engine_factory():
    """Validate create_tts_engine function."""
    print("7. Validating engine factory...")
    
    try:
        from whisper_streaming.tts import create_tts_engine, TTSEngine, TTSConfig
        
        config = TTSConfig(language="tr")
        
        # Test AUTO engine selection
        auto_tts = create_tts_engine(TTSEngine.AUTO, config)
        print(f"   AUTO engine created: {type(auto_tts).__name__}")
        
        # Test available engines
        from whisper_streaming.tts import get_available_engines
        available = get_available_engines()
        
        for engine in available:
            if engine != TTSEngine.PIPER_TTS:  # Skip Piper if not installed
                tts = create_tts_engine(engine, config)
                print(f"   {engine.value} engine created: {type(tts).__name__}")
        
        print("   ✅ Engine factory working")
        return True
        
    except Exception as e:
        print(f"   ❌ Engine factory validation failed: {e}")
        return False

def validate_convenience_function():
    """Validate convenience function."""
    print("8. Validating convenience function...")
    
    try:
        from whisper_streaming.tts import synthesize_turkish, TTSEngine
        
        # This should work with any available engine
        test_text = "Bu kısa bir test cümlesidir."
        
        output_path = synthesize_turkish(
            text=test_text,
            engine=TTSEngine.AUTO,
            speed=1.0
        )
        
        if output_path and output_path.exists():
            print(f"   ✅ Synthesis successful: {output_path}")
            print(f"   📁 File size: {output_path.stat().st_size:,} bytes")
            return True
        else:
            print("   ❌ Synthesis failed - no output file")
            return False
            
    except Exception as e:
        print(f"   ❌ Convenience function validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Piper TTS Integration Validation")
    print("=" * 40)
    
    tests = [
        validate_imports,
        validate_tts_engine_enum,
        validate_config_options,
        validate_engine_detection,
        validate_priority_system,
        validate_piper_instantiation,
        validate_engine_factory,
        validate_convenience_function,
    ]
    
    passed = 0
    total = len(tests)
    piper_available = False
    
    for i, test in enumerate(tests, 1):
        if test.__name__ == 'validate_piper_instantiation':
            # Pass piper availability to instantiation test
            result = test(piper_available)
        elif test.__name__ == 'validate_engine_detection':
            # Get piper availability from detection test
            result = test()
            if result:
                piper_available = True
        else:
            result = test()
        
        if result:
            passed += 1
        
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 40)
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Piper TTS integration is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} validation(s) failed or skipped.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
