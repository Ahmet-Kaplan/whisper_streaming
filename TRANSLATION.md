# Real-time Translation Feature

The whisper-streaming project now includes real-time translation capabilities, allowing you to translate transcribed speech into different languages on-the-fly.

## Features

- 🌐 **Real-time translation** of transcribed text
- 🚀 **Automatic caching** for improved performance
- 🔄 **Multiple translation services** (Google Translate, with extensible architecture)
- 🎯 **Seamless integration** with existing ASR pipeline
- 📊 **Timing preservation** from original transcription
- 🔧 **Configurable output formats** (original + translation, translation only, etc.)

## Supported Languages

The translation feature supports all languages available through Google Translate, including:

- **European**: Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Dutch (nl), etc.
- **Asian**: Japanese (ja), Korean (ko), Chinese (zh), Hindi (hi), Arabic (ar), etc.
- **And many more**: 100+ languages supported

## Quick Start

### Basic Translation

```python
from whisper_streaming.translator import TranslationConfig, get_default_translator
from whisper_streaming.base import Word

# Create translation config
config = TranslationConfig(
    target_language="es",  # Spanish
    source_language="en",  # English (or "auto" for auto-detection)
    cache_translations=True
)

# Get translator
translator = get_default_translator(config)

# Translate a word
word = Word(word="Hello", start=0.0, end=0.5)
translated = translator.translate_word(word)

print(f"Original: {word.word}")
print(f"Translated: {translated.translated_text}")
# Output: Original: Hello, Translated: Hola
```

### Integration with ASR Pipeline

```python
from whisper_streaming import ASRProcessor
from whisper_streaming.receiver import AudioReceiver
from whisper_streaming.translator import TranslationConfig, get_default_translator
from whisper_streaming.sender.translation import ConsoleTranslationSender

# Create audio receiver
audio_receiver = AudioReceiver(device=None, chunk_size=2.0, target_sample_rate=16000)

# Create translation components
translation_config = TranslationConfig(target_language="es", source_language="en")
translator = get_default_translator(translation_config)
translation_sender = ConsoleTranslationSender(
    translator=translator,
    show_original=True,  # Show both original and translated text
    show_timing=True     # Include timing information
)

# Create ASR processor with translation
processor = ASRProcessor(
    processor_config=processor_config,
    audio_receiver=audio_receiver,
    output_senders=translation_sender,  # Use translation sender
    backend=backend,
    model_config=model_config,
    transcribe_config=transcribe_config,
    feature_extractor_config=feature_extractor_config,
)

# Run real-time processing with translation
processor.run()
```

## Configuration Options

### TranslationConfig

```python
from whisper_streaming.translator import TranslationConfig

config = TranslationConfig(
    target_language="es",        # Target language code
    source_language="en",        # Source language ("auto" for auto-detection)
    cache_translations=True,     # Enable caching for performance
    batch_size=1,               # Batch size for translation requests
)
```

### Output Senders

#### ConsoleTranslationSender

```python
from whisper_streaming.sender.translation import ConsoleTranslationSender

sender = ConsoleTranslationSender(
    translator=translator,
    show_original=True,     # Show original text alongside translation
    show_timing=False,      # Include timing information
    output_file=None        # Output file (None = stdout)
)
```

#### TranslationOutputSender (Wrapper)

```python
from whisper_streaming.sender.translation import TranslationOutputSender
from whisper_streaming.sender import PrintSender

# Wrap any existing output sender with translation
original_sender = PrintSender()
translation_sender = TranslationOutputSender(
    translator=translator,
    output_sender=original_sender
)
```

## Language Codes

Common language codes for translation:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | Spanish | es |
| French | fr | German | de |
| Italian | it | Portuguese | pt |
| Japanese | ja | Korean | ko |
| Chinese (Simplified) | zh | Arabic | ar |
| Hindi | hi | Russian | ru |
| Dutch | nl | Swedish | sv |

For a complete list, see [Google Translate supported languages](https://cloud.google.com/translate/docs/languages).

## Performance

### Caching

The translation feature includes intelligent caching to improve performance:

```python
# Cache is automatically enabled by default
config = TranslationConfig(cache_translations=True)
translator = get_default_translator(config)

# First translation: ~200ms (API call)
translated1 = translator.translate_word(word)

# Second translation: ~0.001ms (cached)
translated2 = translator.translate_word(word)  # Same word

# Clear cache if needed
translator.clear_cache()
```

### Performance Tips

1. **Enable caching**: Always use `cache_translations=True` for repeated words
2. **Batch processing**: For large volumes, consider batching (future feature)
3. **Error handling**: Translation gracefully falls back to original text on failures

## Error Handling

The translation system is designed to be resilient:

```python
# If translation fails, original text is preserved
try:
    translated = translator.translate_word(word)
    print(translated.translated_text)  # Translated text
except Exception:
    print(word.word)  # Falls back to original text
```

## Examples

### Multiple Language Demo

```bash
# Run the translation demo
python examples/translation_demo.py
```

### Real-time Translation Simulation

```bash
# Basic usage
python examples/realtime_translation.py

# With options
python examples/realtime_translation.py \
    --target-lang fr \
    --show-original \
    --show-timing
```

## Installation

The translation feature requires the Google Translate library:

```bash
# Using uv
uv pip install googletrans-py

# Using pip
pip install googletrans-py
```

## API Reference

### Classes

- **`TranslationConfig`**: Configuration for translation services
- **`BaseTranslator`**: Abstract base class for translators
- **`GoogleCloudTranslator`**: Google Translate implementation
- **`NoOpTranslator`**: Pass-through translator (no translation)
- **`TranslatedWord`**: Word with translation metadata
- **`ConsoleTranslationSender`**: Console output with translation
- **`TranslationOutputSender`**: Wrapper for existing output senders

### Functions

- **`get_default_translator(config)`**: Get appropriate translator for configuration

## Future Enhancements

- 🔄 Additional translation services (Azure, AWS, OpenAI)
- 📦 Batch translation for improved efficiency
- 🔍 Language auto-detection improvements
- 📊 Translation confidence scoring
- 🎛️ Advanced output formatting options

## Troubleshooting

### Common Issues

1. **"No translation service available"**
   - Install: `pip install googletrans-py`

2. **Translation API errors**
   - Google Translate has rate limits for free usage
   - Errors are handled gracefully (falls back to original text)

3. **Slow translation**
   - Enable caching: `cache_translations=True`
   - Consider using shorter source language codes

### Getting Help

For issues or feature requests related to translation:
1. Check the examples in `examples/`
2. Review this documentation
3. Open an issue on the project repository
