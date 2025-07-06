# Voice Cloning Guide - F5-TTS with 15-Second Recording

## Overview

This guide shows how to record 15 seconds of your voice and use it for voice cloning with F5-TTS in the whisper_streaming project. The system will capture your voice, transcribe it, and allow you to generate new speech in your voice.

## 🎙️ Quick Start

### Method 1: Quick Voice Cloning (Recommended)

The simplest way to get started:

```bash
cd /path/to/whisper_streaming
python quick_voice_clone.py
```

This script will:
1. ✅ Record 15 seconds of your voice
2. 🤖 Auto-transcribe the recording
3. ⚙️ Set up voice cloning configuration
4. 🎯 Test with sample texts
5. 🎮 Allow interactive testing

### Method 2: Advanced Voice Cloning

For more control and features:

```bash
python voice_cloning_recorder.py
```

This provides:
- Detailed recording controls
- Manual transcription override
- Configuration saving for reuse
- Advanced testing options

### Method 3: Load Saved Configuration

If you've already recorded your voice:

```bash
python load_voice_config.py
```

This allows you to:
- Load previously saved voice configurations
- Generate new speech with your saved voice
- Batch process text files

## 📋 Prerequisites

### Required Packages
- ✅ f5-tts (already installed)
- ✅ pyaudio (for microphone recording)
- ✅ whisper_streaming TTS module

### Audio Setup
- 🎤 Working microphone
- 🔊 Speakers/headphones (for playback)
- 🎵 Quiet recording environment

## 🎯 Step-by-Step Guide

### Step 1: Prepare for Recording

1. **Find a quiet space** - Minimize background noise
2. **Test your microphone** - Speak normally, not too close/far
3. **Prepare your text** - Know what you'll say (15 seconds)

### Step 2: Record Your Voice

```bash
python quick_voice_clone.py
```

**Recording Tips:**
- Speak clearly and naturally
- Use your normal speaking pace
- Include varied intonation
- Avoid long pauses
- Don't rush or whisper

**Good 15-second examples:**
```
Hello, this is my voice for cloning. I'm speaking clearly and naturally 
with good intonation. This recording will be used to create a digital 
version of my voice that can speak any text.
```

```
Merhaba, ben sesimi klonlamak için kaydediyorum. Açık ve doğal bir şekilde 
konuşuyorum. Bu kayıt, herhangi bir metni söyleyebilecek dijital bir ses 
versiyonu oluşturmak için kullanılacak.
```

### Step 3: Transcription

The system will automatically transcribe your recording. You can:
- ✅ Accept the auto-transcription
- ✏️ Edit it manually
- ❌ Reject and type it yourself

**Important:** The transcription must exactly match what you said!

### Step 4: Language Selection

Choose your language:
1. **English** - Best model support
2. **Turkish** - Good support with Turkish preprocessing
3. **Other** - Specify language code (es, fr, de, etc.)

### Step 5: Test Voice Cloning

The system will test your cloned voice with sample texts:

**English samples:**
- "Hello, this is my cloned voice speaking."
- "The voice cloning quality is amazing."
- "F5-TTS works incredibly well."

**Turkish samples:**
- "Merhaba, bu benim klonlanmış sesim."
- "Türkçe konuşma sentezi çok başarılı."
- "F5-TTS gerçekten harika çalışıyor."

### Step 6: Interactive Mode

Enter your own text to test:
```
Text: How are you doing today?
🔄 Generating: 'How are you doing today?'
✅ Generated: /tmp/tmpXXX_cloned.wav
Play? (y/n): y
```

## 🔧 Advanced Usage

### Save Configuration for Reuse

When prompted, save your voice configuration:
```
Save configuration for reuse? (y/n): y
Enter configuration name: my_voice
✅ Configuration saved to: voice_config_my_voice.txt
```

### Load Saved Voice

```bash
python load_voice_config.py
```

Select from available configurations and generate new speech.

### Batch Processing

Process multiple texts at once:
```bash
# Create a text file with one sentence per line
echo "Hello world" > texts.txt
echo "How are you?" >> texts.txt
echo "Nice to meet you" >> texts.txt

# Batch generate
python load_voice_config.py voice_config_my_voice.txt texts.txt output_folder/
```

## 🎚️ Configuration Options

### Audio Quality Settings
- **Sample Rate:** 24kHz (F5-TTS optimized)
- **Channels:** Mono (1 channel)
- **Format:** 16-bit WAV
- **Duration:** 15 seconds (recommended 3-10 seconds)

### F5-TTS Parameters
- **Model:** F5TTS_v1_Base (latest, best quality)
- **Device:** Auto (GPU if available, otherwise CPU)
- **Speed:** 1.0 (normal speech rate)
- **Seed:** 42 (for reproducible results)

## 📁 File Structure

After recording, you'll have:
```
whisper_streaming/
├── quick_voice_clone.py          # Quick recording script
├── voice_cloning_recorder.py     # Advanced recording
├── load_voice_config.py          # Load saved configurations
├── voice_config_my_voice.txt     # Saved voice configuration
└── generated_audio/              # Output folder (batch mode)
    ├── line_001.wav
    ├── line_002.wav
    └── ...
```

## 🎵 Audio Playback

The scripts automatically play generated audio on macOS using `afplay`. For other systems:

- **Linux:** Uses `aplay`
- **Windows:** Uses PowerShell SoundPlayer
- **Manual:** Open the generated .wav files in any audio player

## 🔍 Troubleshooting

### Common Issues

**1. Recording Problems**
```
❌ No audio input devices found!
```
**Solution:** Check microphone permissions and connections

**2. Transcription Issues**
```
⚠️ Auto-transcription failed
```
**Solution:** Manually type what you said exactly

**3. Voice Cloning Errors**
```
❌ F5-TTS requires reference audio and text
```
**Solution:** Ensure both audio file and text are provided

**4. Audio Playback Issues**
```
⚠️ Could not play audio
```
**Solution:** Check speaker/headphone connections, or manually open the .wav file

### Performance Tips

**For Best Quality:**
1. **Use GPU** - Much faster generation
2. **Good Microphone** - Higher quality input = better output
3. **Clear Speech** - Speak distinctly but naturally
4. **Proper Duration** - 15 seconds is optimal, minimum 3-10 seconds

**Troubleshooting Commands:**
```bash
# Check audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"

# Test F5-TTS installation
python -c "from f5_tts.api import F5TTS; print('F5-TTS OK')"

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## 🌟 Tips for Best Results

### Recording Quality
- **Environment:** Quiet room, no echo
- **Distance:** 6-12 inches from microphone
- **Volume:** Speak at normal conversational level
- **Consistency:** Maintain same tone throughout

### Text Content
- **Variety:** Include different sounds and intonations
- **Natural:** Use conversational language
- **Complete:** Avoid cutting off words at the 15-second mark
- **Clear:** Pronounce words distinctly

### Language-Specific Tips

**English:**
- Use varied sentence structures
- Include common words and sounds
- Natural rhythm and stress patterns

**Turkish:**
- Pronounce Turkish characters clearly (ç, ğ, ı, ö, ş, ü)
- Use natural Turkish intonation
- Include common Turkish phonemes

## 📊 Quality Comparison

| Method | Quality | Speed | Ease of Use | Voice Similarity |
|--------|---------|-------|-------------|------------------|
| F5-TTS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Edge TTS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Piper TTS | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Coqui TTS | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| System TTS | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

## 🚀 Next Steps

1. **Record your voice** with `quick_voice_clone.py`
2. **Save the configuration** for reuse
3. **Generate speech** for your applications
4. **Integrate** with whisper_streaming workflows
5. **Experiment** with different languages and styles

## 📚 API Integration

Use voice cloning in your Python code:

```python
from whisper_streaming.tts import TTSEngine, TTSConfig, create_tts_engine

# Load your voice configuration
config = TTSConfig(
    language="en",
    f5_model="F5TTS_v1_Base",
    f5_ref_audio="path/to/your_voice.wav",
    f5_ref_text="Your reference text here",
    f5_device="auto"
)

# Create TTS engine
tts = create_tts_engine(TTSEngine.F5_TTS, config)

# Generate speech with your voice
output = tts.synthesize("Any text you want to say in your voice")
print(f"Generated: {output}")
```

## 🎉 Conclusion

With these tools, you can easily create high-quality voice clones using just 15 seconds of recording. The F5-TTS integration provides state-of-the-art results with a simple workflow that works for multiple languages.

Start with `quick_voice_clone.py` for the best experience!
