# Cross-Platform Audio Compatibility

This project now supports both Linux and macOS (and Windows) by automatically selecting the appropriate audio library based on the platform.

## Supported Platforms

### Linux
- **Primary**: ALSA (Advanced Linux Sound Architecture) via `pyalsaaudio`
- **Fallback**: PyAudio via `pyaudio`
- **Installation**: `sudo apt-get install libasound2-dev` (for ALSA headers)

### macOS
- **Primary**: PyAudio with PortAudio via `pyaudio`
- **Installation**: `brew install portaudio` (automatically handled)

### Windows
- **Primary**: PyAudio with PortAudio via `pyaudio`

## Usage

### Simple Usage (Recommended)
```python
from whisper_streaming.receiver import AudioReceiver

# Automatically selects the best receiver for your platform
receiver = AudioReceiver(
    device=None,  # Use default device
    chunk_size=1.0,  # 1 second chunks
    target_sample_rate=16000,  # 16 kHz
)
```

### Platform-Specific Usage
```python
from whisper_streaming.receiver import get_default_audio_receiver

# Get the default receiver class for the current platform
ReceiverClass = get_default_audio_receiver()
receiver = ReceiverClass(...)
```

### Direct Platform-Specific Import
```python
# Linux (ALSA)
from whisper_streaming.receiver import AlsaReceiver
receiver = AlsaReceiver(device="default", ...)

# macOS/Windows (PyAudio)
from whisper_streaming.receiver import PyAudioReceiver
receiver = PyAudioReceiver(device_index=None, ...)
```

## Device Selection

### ALSA (Linux)
- **String device names**: `"default"`, `"hw:0"`, `"plughw:1,0"`
- **Integer conversion**: `device=0` becomes `"hw:0"`

### PyAudio (macOS/Windows)
- **Integer device index**: `0`, `1`, `2`, etc. (use `None` for default)
- **String fallback**: String device names will show a warning and use default device

## Dependencies

The project automatically installs the correct dependencies based on your platform:

```txt
# requirements/library/requirements.txt
librosa
numpy
pyaudio
pyalsaaudio; sys_platform == "linux"  # Only on Linux
websockets
sacremoses  # Pure Python tokenizer (cross-platform)
```

## Testing

Run the example script to test cross-platform compatibility:

```bash
python examples/cross_platform_audio.py
```

This will:
1. Detect your platform
2. Show which audio receiver is being used
3. Test initialization of the audio receiver
4. Provide installation instructions if something fails

## Troubleshooting

### Linux Issues
```bash
# Install ALSA development headers
sudo apt-get install libasound2-dev

# Install PyAudio as fallback
pip install pyaudio
```

### macOS Issues
```bash
# Install PortAudio
brew install portaudio

# Reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio

# For tokenization (sentence trimming), use sacremoses (easier on macOS)
pip install sacremoses

# Alternative: Install mosestokenizer with proper linking (advanced)
# brew install gettext cmake
# LDFLAGS="-L/opt/homebrew/opt/gettext/lib" CPPFLAGS="-I/opt/homebrew/opt/gettext/include" pip install --no-cache-dir --force-reinstall mosestokenizer
# echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
```

### Windows Issues
```bash
# PyAudio should work out of the box, but if not:
pip install pyaudio
```

## Migration from ALSA-only

If you were previously using `AlsaReceiver` directly:

```python
# Old (Linux-only)
from whisper_streaming.receiver import AlsaReceiver
receiver = AlsaReceiver(device="default", ...)

# New (Cross-platform)
from whisper_streaming.receiver import AudioReceiver
receiver = AudioReceiver(device="default", ...)  # Works on all platforms
```

The new `AudioReceiver` class handles device parameter conversion automatically.
