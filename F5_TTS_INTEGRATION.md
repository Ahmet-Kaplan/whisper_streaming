# F5-TTS Integration with Whisper Streaming

## Overview

F5-TTS has been successfully integrated into the whisper_streaming project as a high-quality text-to-speech engine. F5-TTS offers state-of-the-art speech synthesis with voice cloning capabilities.

## Features

- **High-Quality Synthesis**: F5-TTS provides natural-sounding speech with excellent quality
- **Voice Cloning**: Ability to clone voices using reference audio (3-10 seconds)
- **Multilingual Support**: Supports multiple languages including Turkish
- **GPU Acceleration**: Automatic GPU detection and usage when available
- **Configurable**: Extensive configuration options for fine-tuning

## Installation

F5-TTS is automatically installed when you install the whisper_streaming project:

```bash
# F5-TTS is included in requirements.txt
pip install f5-tts
```

## Usage

### Basic Usage

```python
from whisper_streaming.tts import (
    TTSEngine, 
    TTSConfig, 
    create_tts_engine,
    synthesize_turkish
)

# Quick synthesis using convenience function
audio_path = synthesize_turkish(
    text="Merhaba! Bu F5-TTS kullanarak oluşturulan ses.",
    engine=TTSEngine.F5_TTS,
    speed=1.0
)
```

### Advanced Configuration

```python
config = TTSConfig(
    language="tr",
    f5_model="F5TTS_v1_Base",
    f5_device="auto",  # Will use GPU if available
    f5_seed=42,
    speed=1.0,
    # Optional: Voice cloning parameters
    f5_ref_audio="path/to/reference.wav",
    f5_ref_text="Reference text matching the audio"
)

tts_engine = create_tts_engine(TTSEngine.F5_TTS, config)
output_path = tts_engine.synthesize("Text to synthesize")
```

### Voice Cloning

For voice cloning, provide reference audio and text:

```python
config = TTSConfig(
    language="tr",
    f5_model="F5TTS_v1_Base",
    f5_ref_audio="reference_voice.wav",  # 3-10 seconds of clear speech
    f5_ref_text="Exact text spoken in reference audio",
    f5_device="auto",
    f5_seed=42
)

tts_engine = create_tts_engine(TTSEngine.F5_TTS, config)
cloned_voice_path = tts_engine.synthesize("New text in cloned voice")
```

## Configuration Options

### TTSConfig F5-TTS Parameters

- **f5_model**: Model name (default: "F5TTS_v1_Base")
  - Available models: F5TTS_v1_Base, F5TTS_Base, E2TTS_Base, etc.
- **f5_ref_audio**: Path to reference audio file for voice cloning
- **f5_ref_text**: Text corresponding to reference audio
- **f5_device**: Device for inference ("auto", "cpu", "cuda")
- **f5_seed**: Random seed for reproducible generation

### Available Models

- **F5TTS_v1_Base**: Latest version with best quality (default)
- **F5TTS_Base**: Standard F5-TTS model
- **E2TTS_Base**: E2-TTS model variant

## Priority in Engine Selection

F5-TTS is configured as the highest priority engine for both offline and online scenarios:

1. **Offline Priority**: F5_TTS → Piper → System → Edge → Google
2. **Quality Priority**: F5_TTS → Edge → Piper → Google → System

## Examples

See the following example files:

- `examples/f5_tts_example.py`: Comprehensive F5-TTS demonstration
- `test_f5_tts_integration.py`: Integration test and verification

Run the example:

```bash
cd /path/to/whisper_streaming
python examples/f5_tts_example.py
```

## Requirements

- Python 3.9+
- PyTorch
- f5-tts package
- Optional: CUDA for GPU acceleration

## Best Practices

### For Turkish TTS

1. **Reference Audio**: Use clear Turkish speech (3-10 seconds)
2. **Text Preprocessing**: The integration includes Turkish-specific text preprocessing
3. **Voice Selection**: Provide Turkish reference audio for best results

### Performance Tips

1. **GPU Usage**: Set `f5_device="cuda"` for faster generation
2. **Model Selection**: Use F5TTS_v1_Base for best quality
3. **Reference Quality**: Use high-quality, noise-free reference audio

## Troubleshooting

### Common Issues

1. **Missing Reference**: F5-TTS requires reference audio and text
   ```python
   # Solution: Always provide reference
   config.f5_ref_audio = "path/to/reference.wav"
   config.f5_ref_text = "Reference text"
   ```

2. **Model Not Found**: Ensure correct model name
   ```python
   # Use valid model names
   config.f5_model = "F5TTS_v1_Base"  # Correct
   # config.f5_model = "F5-TTS"  # Incorrect
   ```

3. **Memory Issues**: Use CPU if GPU memory is insufficient
   ```python
   config.f5_device = "cpu"
   ```

## Integration Details

F5-TTS has been integrated into whisper_streaming with:

1. **Engine Registration**: Added to `TTSEngine` enum as `F5_TTS`
2. **Configuration**: Extended `TTSConfig` with F5-TTS specific options
3. **Implementation**: `F5TTSEngine` class implementing the `BaseTTS` interface
4. **Priority**: Highest priority in engine selection for quality
5. **Dependencies**: Added to requirements.txt

## Performance

- **Quality**: Excellent natural speech synthesis
- **Speed**: Fast generation with GPU acceleration
- **Memory**: Moderate GPU/CPU memory usage
- **Languages**: Supports multiple languages including Turkish

## Contributing

To extend F5-TTS integration:

1. Modify `TTSConfig` for new parameters
2. Update `F5TTSEngine` for new features
3. Add tests in `test_f5_tts_integration.py`
4. Update examples in `examples/f5_tts_example.py`

## References

- [F5-TTS GitHub Repository](https://github.com/SWivid/F5-TTS)
- [F5-TTS Paper](https://arxiv.org/abs/2410.06885)
- [Whisper Streaming Project](https://github.com/nkaaf/ufal-whisper_streaming)
