#!/usr/bin/env python3
"""Example demonstrating cross-platform audio receiver usage."""

import os
import platform

# Set up environment for macOS mosestokenizer support
if platform.system() == "Darwin":
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    if homebrew_lib not in current_path:
        os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

from whisper_streaming.receiver import AudioReceiver, get_default_audio_receiver

def main():
    """Demonstrate cross-platform audio receiver."""
    print(f"Running on: {platform.system()} {platform.release()}")
    
    # Get the default receiver class for this platform
    receiver_class = get_default_audio_receiver()
    print(f"Default audio receiver: {receiver_class.__name__}")
    
    # Create an audio receiver instance
    # The AudioReceiver factory will automatically choose the right implementation
    try:
        receiver = AudioReceiver(
            device=None,  # Use default device
            chunk_size=1.0,  # 1 second chunks
            target_sample_rate=16000,  # 16 kHz
        )
        print(f"Created receiver: {type(receiver).__name__}")
        print("✓ Audio receiver initialized successfully!")
        
        # Clean up
        receiver.close()
        
    except Exception as e:
        print(f"✗ Failed to initialize audio receiver: {e}")
        print("Make sure you have the required audio libraries installed:")
        if platform.system().lower() == "linux":
            print("  - On Linux: sudo apt-get install libasound2-dev")
        print("  - PyAudio: pip install pyaudio (cross-platform)")

if __name__ == "__main__":
    main()
