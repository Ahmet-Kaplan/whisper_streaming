#!/usr/bin/env python3
"""Simple test for cross-platform audio receiver."""

import platform
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_audio_receiver():
    """Test the cross-platform audio receiver."""
    print(f"Running on: {platform.system()} {platform.release()}")
    
    try:
        from whisper_streaming.receiver.audio import get_default_audio_receiver, AudioReceiver
        
        # Get the default receiver class for this platform
        receiver_class = get_default_audio_receiver()
        print(f"Default audio receiver: {receiver_class.__name__}")
        
        # Test creating an instance
        receiver = AudioReceiver(
            device=None,  # Use default device
            chunk_size=1.0,  # 1 second chunks
            target_sample_rate=16000,  # 16 kHz
        )
        print(f"Created receiver: {type(receiver).__name__}")
        print("✓ Audio receiver initialized successfully!")
        
        # Test platform-specific imports
        print("\nTesting platform-specific imports:")
        
        try:
            from whisper_streaming.receiver.alsa import AlsaReceiver
            print("✓ ALSA receiver available")
        except ImportError:
            print("✗ ALSA receiver not available (expected on non-Linux)")
            
        try:
            from whisper_streaming.receiver.pyaudio_receiver import PyAudioReceiver  
            print("✓ PyAudio receiver available")
        except ImportError:
            print("✗ PyAudio receiver not available")
        
        # Clean up
        receiver.close()
        
    except Exception as e:
        print(f"✗ Failed to test audio receiver: {e}")
        print("Make sure you have the required audio libraries installed:")
        if platform.system().lower() == "linux":
            print("  - On Linux: sudo apt-get install libasound2-dev")
        print("  - PyAudio: pip install pyaudio (cross-platform)")
        return False
        
    return True

if __name__ == "__main__":
    success = test_audio_receiver()
    sys.exit(0 if success else 1)
