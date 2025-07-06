# Piper-Plus TTS Integration

This document describes the integration of Piper-Plus TTS (Text-to-Speech) engine into the whisper_streaming project, providing high-quality Turkish language synthesis.

## Overview

Piper-Plus TTS is an enhanced version of Piper TTS, a fast, local neural text-to-speech system optimized for efficiency and quality. It provides excellent Turkish language support with multiple voice models and is designed for offline operation. Piper-Plus includes Japanese language optimizations and improved build automation.

### Key Features

- ✅ **High-quality Turkish synthesis** - Neural TTS optimized for Turkish
- ✅ **Offline operation** - No internet connection required after model download
- ✅ **Fast synthesis** - Optimized for speed and low latency
- ✅ **Multiple Turkish models** - Different quality/speed tradeoffs
- ✅ **Automatic model management** - Downloads and caches models automatically
- ✅ **Turkish text preprocessing** - Handles abbreviations, numbers, and phonetics
- ✅ **Seamless integration** - Works with existing TTS interface

## Installation

### Basic Installation

```bash
pip install piper-tts-plus
```

### Virtual Environment Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install piper-tts-plus
```

### Project Dependencies

Piper-Plus TTS is already included in the project requirements. Install all dependencies:

```bash
pip install -r requirements/library/requirements.txt
```

### Alternative Installation (if package conflicts occur)

If you encounter dependency conflicts with `piper-phonemize`, you can install without dependencies:

```bash
pip install --no-deps piper-tts-plus
pip install "onnxruntime>=1.11.0,<2"
```

**Note**: Without `piper-phonemize`, text preprocessing will use a fallback implementation. This may result in some phoneme warnings but synthesis will still work.

## Available Turkish Models

Piper TTS supports several Turkish voice models with different characteristics:

| Model | Full Name | Quality | Size | Speed | Description |
|-------|-----------|---------|------|-------|-------------|
| `dfki-medium` | `tr_TR-dfki-medium` | High | Medium | Medium | Best overall quality |
| `dfki-low` | `tr_TR-dfki-low` | Good | Small | Fast | Faster, smaller model |
| `fgl-medium` | `tr_TR-fgl-medium` | High | Medium | Medium | Alternative voice |

Models are automatically downloaded on first use and cached locally in `~/.local/share/piper/`.

## Usage

### Basic Usage

```python
from whisper_streaming.tts import TTSEngine, TTSConfig, PiperTTS

# Create configuration
config = TTSConfig(
    language=\"tr\",
    piper_model=\"dfki-medium\",
    speed=1.0
)

# Create TTS engine
tts = PiperTTS(config)

# Synthesize text
output_path = tts.synthesize(\"Merhaba! Bu Piper-Plus TTS ile oluşturulmuş bir ses örneğidir.\")
print(f\"Audio saved to: {output_path}\")
```

### Using Convenience Function

```python
from whisper_streaming.tts import synthesize_turkish, TTSEngine

# Quick synthesis
output_path = synthesize_turkish(
    text=\"Türkçe metin sentezi\",
    engine=TTSEngine.PIPER_TTS,
    speed=1.2
)
```

### Auto Engine Selection

```python
from whisper_streaming.tts import synthesize_turkish, TTSEngine

# Let the system choose the best available engine
output_path = synthesize_turkish(
    text=\"Otomatik motor seçimi\",
    engine=TTSEngine.AUTO  # Will prefer Piper if available
)
```

### Advanced Configuration

```python
from whisper_streaming.tts import TTSConfig, PiperTTS

config = TTSConfig(
    language=\"tr\",
    piper_model=\"tr_TR-dfki-medium\",  # Full model name
    speed=1.1,
    use_turkish_phonetics=True,
    handle_foreign_words=True,
    piper_data_dir=\"/custom/path/to/models\",
    piper_download_dir=\"/custom/download/path\",
    sample_rate=22050,
    output_format=\"wav\"
)

tts = PiperTTS(config)
output_path = tts.synthesize(\"Gelişmiş yapılandırma örneği\")
```

## Configuration Options

### Piper-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `piper_model` | `str` | `\"tr_TR-dfki-medium\"` | Turkish model to use |
| `piper_data_dir` | `str` | `None` | Directory to store models |
| `piper_download_dir` | `str` | `None` | Directory for downloads |

### General TTS Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | `str` | `\"tr\"` | Language code |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `use_turkish_phonetics` | `bool` | `True` | Enable Turkish preprocessing |
| `handle_foreign_words` | `bool` | `True` | Handle foreign words |
| `sample_rate` | `int` | `22050` | Audio sample rate |
| `output_format` | `str` | `\"wav\"` | Output audio format |

## Engine Priority

When using `TTSEngine.AUTO`, the system selects engines in the following order:

### Offline Preference (`prefer_offline=True`)
1. **Piper TTS** - Excellent offline quality and fast
2. Coqui TTS - Good offline quality (heavier)
3. System TTS - Basic offline option
4. Edge TTS - High quality (requires internet)
5. Google TTS - Good quality (requires internet)

### Online Preference (`prefer_offline=False`)
1. **Edge TTS** - Best overall quality for Turkish
2. **Piper TTS** - Excellent offline quality and fast
3. Google TTS - Good quality and reliability
4. Coqui TTS - Good offline quality
5. System TTS - Basic but available

## Text Preprocessing

Piper TTS includes Turkish-specific text preprocessing:

### Abbreviation Expansion
- `T.C.` → `Türkiye Cumhuriyeti`
- `vs.` → `vesaire`
- `vb.` → `ve benzeri`
- `A.Ş.` → `Anonim Şirketi`

### Number Processing
- `1.` → `birinci`
- `2.` → `ikinci`
- `3.` → `üçüncü`

### Example
```python
# Input text with abbreviations and numbers
text = \"T.C. Cumhurbaşkanlığı 1. sırada yer alıyor vs.\"

# Preprocessed for better synthesis
# \"Türkiye Cumhuriyeti Cumhurbaşkanlığı birinci sırada yer alıyor vesaire\"
```

## Error Handling

The integration includes comprehensive error handling:

```python
from whisper_streaming.tts import PiperTTS, TTSConfig

try:
    config = TTSConfig(piper_model=\"dfki-medium\")
    tts = PiperTTS(config)
    
    if not tts.is_available():
        print(\"Piper TTS is not available\")
        # Fallback to other engines
    
    output_path = tts.synthesize(\"Test metni\")
    
except ImportError:
    print(\"Piper TTS not installed. Install with: pip install piper-tts\")
except Exception as e:
    print(f\"Synthesis failed: {e}\")
```

## Performance Characteristics

### Speed Comparison (approximate)
- **Piper TTS**: Very fast (0.1-0.2x real-time)
- Edge TTS: Fast (network dependent)
- Google TTS: Medium (network dependent)
- Coqui TTS: Slower (0.2-0.5x real-time)
- System TTS: Variable

### Quality Comparison
- **Edge TTS**: Excellent (best overall)
- **Piper TTS**: Very good (best offline)
- Google TTS: Good
- Coqui TTS: Good
- System TTS: Basic

### Resource Usage
- **Piper TTS**: Low CPU, moderate memory
- Edge TTS: Low (network-based)
- Google TTS: Low (network-based)
- Coqui TTS: High CPU and memory
- System TTS: Low

## Troubleshooting

### Installation Issues

**Problem**: `piper-tts` installation fails
```bash
pip install --upgrade pip
pip install piper-tts
```

**Problem**: Missing dependencies
```bash
pip install onnxruntime numpy
pip install piper-tts
```

### Model Download Issues

**Problem**: Model download fails
- Check internet connection
- Verify `piper_download_dir` permissions
- Try manual download from [Hugging Face](https://huggingface.co/rhasspy/piper-voices)

**Problem**: Insufficient disk space
- Models are ~50-100MB each
- Default location: `~/.local/share/piper/`
- Configure custom directory with `piper_data_dir`

### Synthesis Issues

**Problem**: Poor quality output
- Try different model: `tr_TR-dfki-medium` vs `tr_TR-fgl-medium`
- Check text preprocessing settings
- Verify input text encoding (UTF-8)

**Problem**: Slow synthesis
- Use `tr_TR-dfki-low` for faster synthesis
- Check system resources
- Consider smaller batch sizes

## Examples

### Complete Example Script

```python
#!/usr/bin/env python3
\"\"\"Piper TTS Example\"\"\"

from whisper_streaming.tts import (
    TTSEngine, TTSConfig, PiperTTS, 
    get_available_engines, synthesize_turkish
)

def main():
    # Check availability
    engines = get_available_engines()
    print(f\"Available engines: {[e.value for e in engines]}\")
    
    if TTSEngine.PIPER_TTS not in engines:
        print(\"Piper TTS not available. Please install: pip install piper-tts\")
        return
    
    # Test different models
    models = [\"dfki-medium\", \"dfki-low\", \"fgl-medium\"]
    
    for model in models:
        print(f\"\\nTesting {model}...\")
        
        try:
            config = TTSConfig(
                language=\"tr\",
                piper_model=model,
                speed=1.0,
                use_turkish_phonetics=True
            )
            
            tts = PiperTTS(config)
            output_path = tts.synthesize(f\"Bu {model} modeliyle oluşturulmuş bir test.\")
            
            print(f\"✅ Success: {output_path}\")
            print(f\"📁 Size: {output_path.stat().st_size:,} bytes\")
            
        except Exception as e:
            print(f\"❌ Error with {model}: {e}\")
    
    # Test convenience function
    print(\"\\nTesting convenience function...\")
    output_path = synthesize_turkish(
        text=\"Bu basit kullanım örneğidir.\",
        engine=TTSEngine.PIPER_TTS,
        speed=1.2
    )
    print(f\"✅ Convenience function: {output_path}\")

if __name__ == \"__main__\":
    main()
```

## Integration with whisper_streaming

Piper TTS integrates seamlessly with the whisper_streaming project's TTS system:

1. **Automatic Detection**: Available engines are detected at runtime
2. **Unified Interface**: Same API as other TTS engines
3. **Priority System**: Preferred for offline Turkish synthesis
4. **Configuration**: Integrated with project configuration system
5. **Error Handling**: Graceful fallback to other engines

## References

- [Piper TTS GitHub](https://github.com/rhasspy/piper)
- [Piper Voice Models](https://huggingface.co/rhasspy/piper-voices)
- [Piper Documentation](https://github.com/rhasspy/piper/blob/master/README.md)
- [Turkish Voice Samples](https://rhasspy.github.io/piper-samples/#turkish-tr_tr)

## License

Piper TTS is licensed under the MIT License. Turkish voice models may have different licenses - check the `MODEL_CARD` file for each model for specific licensing information.
