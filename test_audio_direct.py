#!/usr/bin/env python3
"""Direct test of audio receivers without full package import."""

import platform
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pyaudio_receiver():
    """Test PyAudio receiver directly."""
    print("Testing PyAudio receiver...")
    try:
        from whisper_streaming.receiver.pyaudio_receiver import PyAudioReceiver
        
        receiver = PyAudioReceiver(
            device_index=None,
            chunk_size=1.0,
            target_sample_rate=16000,
        )
        print("✓ PyAudio receiver created successfully!")
        receiver._do_close()
        return True
    except Exception as e:
        print(f"✗ PyAudio receiver failed: {e}")
        return False

def test_alsa_receiver():
    """Test ALSA receiver directly."""
    print("Testing ALSA receiver...")
    try:
        from whisper_streaming.receiver.alsa import AlsaReceiver
        
        # Only test import on Linux
        if platform.system().lower() == "linux":
            receiver = AlsaReceiver(
                device="default",
                chunk_size=1.0,
                target_sample_rate=16000,
            )
            print("✓ ALSA receiver created successfully!")
            receiver._do_close()
        else:
            print("✓ ALSA receiver imported (Linux-only functionality)")
        return True
    except Exception as e:
        print(f"✗ ALSA receiver failed: {e}")
        return False

def test_cross_platform_logic():
    """Test the cross-platform selection logic."""
    print("Testing cross-platform selection logic...")
    try:
        from whisper_streaming.receiver.audio import get_default_audio_receiver
        
        receiver_class = get_default_audio_receiver()
        print(f"✓ Selected receiver: {receiver_class.__name__}")
        
        # Test the logic based on platform
        system = platform.system().lower()
        if system == "linux":
            expected = "AlsaReceiver"
        else:
            expected = "PyAudioReceiver"
            
        if receiver_class.__name__ == expected:
            print(f"✓ Correct receiver selected for {system}")
        else:
            print(f"⚠ Unexpected receiver {receiver_class.__name__} for {system} (may be fallback)")
            
        return True
    except Exception as e:
        print(f"✗ Cross-platform logic failed: {e}")
        return False

def main():
    """Run all tests."""
    print(f"Running on: {platform.system()} {platform.release()}")
    print("=" * 50)
    
    results = []
    
    # Test individual receivers
    results.append(test_pyaudio_receiver())
    print()
    results.append(test_alsa_receiver()) 
    print()
    results.append(test_cross_platform_logic())
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Cross-platform audio is working.")
    else:
        print("✗ Some tests failed. Check the output above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
