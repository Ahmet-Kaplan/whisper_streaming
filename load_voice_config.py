#!/usr/bin/env python3
"""
Voice Configuration Loader

Load and use saved voice cloning configurations for F5-TTS.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from whisper_streaming.tts import TTSEngine, TTSConfig, create_tts_engine
except ImportError as e:
    logger.error(f"Required modules not found: {e}")
    sys.exit(1)


def load_voice_config(config_file: str) -> Dict[str, str]:
    """Load voice configuration from file."""
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    
    required_keys = ['ref_audio_path', 'ref_text', 'language']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return config


def create_tts_config_from_saved(saved_config: Dict[str, str]) -> TTSConfig:
    """Create TTSConfig from saved configuration."""
    
    config = TTSConfig(
        language=saved_config['language'],
        f5_model="F5TTS_v1_Base",
        f5_ref_audio=saved_config['ref_audio_path'],
        f5_ref_text=saved_config['ref_text'],
        f5_device="auto",
        f5_seed=42,
        speed=1.0
    )
    
    return config


def generate_speech(config: TTSConfig, text: str, output_path: Optional[Path] = None) -> Path:
    """Generate speech using the loaded voice configuration."""
    
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix="_generated.wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
    
    # Create TTS engine
    tts_engine = create_tts_engine(TTSEngine.F5_TTS, config)
    
    # Generate speech
    result_path = tts_engine.synthesize(text, output_path)
    
    return result_path


def list_available_configs() -> list:
    """List available voice configuration files."""
    
    config_files = list(Path.cwd().glob("voice_config_*.txt"))
    return config_files


def interactive_mode():
    """Interactive text-to-speech with loaded voice."""
    
    print("🎙️  Voice Configuration Loader")
    print("=" * 30)
    
    # List available configurations
    config_files = list_available_configs()
    
    if not config_files:
        print("❌ No voice configuration files found!")
        print("Run 'python voice_cloning_recorder.py' or 'python quick_voice_clone.py' first.")
        return 1
    
    print("📁 Available voice configurations:")
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file.name}")
    
    # Select configuration
    while True:
        try:
            choice = input(f"\nSelect configuration (1-{len(config_files)}): ").strip()
            config_index = int(choice) - 1
            
            if 0 <= config_index < len(config_files):
                selected_config = config_files[config_index]
                break
            else:
                print("❌ Invalid choice, try again.")
        except ValueError:
            print("❌ Please enter a number.")
    
    # Load configuration
    try:
        print(f"\n📤 Loading configuration: {selected_config.name}")
        saved_config = load_voice_config(selected_config)
        tts_config = create_tts_config_from_saved(saved_config)
        
        print("✅ Configuration loaded successfully!")
        print(f"   Language: {saved_config['language']}")
        print(f"   Reference: '{saved_config['ref_text'][:50]}...'")
        
        # Verify reference audio exists
        ref_audio_path = Path(saved_config['ref_audio_path'])
        if not ref_audio_path.exists():
            print(f"⚠️  Warning: Reference audio not found: {ref_audio_path}")
            print("   Voice cloning may not work properly.")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Interactive text-to-speech
    print(f"\n🎯 Interactive Voice Synthesis ({saved_config['language']})")
    print("Enter text to generate with your cloned voice (or 'quit' to exit):")
    
    while True:
        user_text = input("\nText: ").strip()
        
        if user_text.lower() in ['quit', 'exit', 'q', '']:
            break
        
        try:
            print("🔄 Generating speech...")
            result_path = generate_speech(tts_config, user_text)
            
            print(f"✅ Generated: {result_path}")
            
            # Option to play audio
            play_choice = input("Play audio? (y/n): ").lower()
            if play_choice.startswith('y'):
                try:
                    import subprocess
                    import platform
                    
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["afplay", str(result_path)])
                    elif platform.system() == "Linux":
                        subprocess.run(["aplay", str(result_path)])
                    elif platform.system() == "Windows":
                        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{result_path}').PlaySync()"])
                except Exception as e:
                    print(f"⚠️  Could not play audio: {e}")
            
            # Option to save with custom name
            save_choice = input("Save with custom name? (y/n): ").lower()
            if save_choice.startswith('y'):
                custom_name = input("File name (without extension): ").strip()
                if custom_name:
                    custom_path = Path(f"{custom_name}.wav")
                    import shutil
                    shutil.copy2(result_path, custom_path)
                    print(f"💾 Saved as: {custom_path}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            continue
    
    print("\n🎉 Session complete!")
    return 0


def batch_mode(config_file: str, text_file: str, output_dir: str = "generated_audio"):
    """Batch processing mode."""
    
    print(f"📊 Batch Mode: {config_file} → {text_file}")
    
    try:
        # Load configuration
        saved_config = load_voice_config(config_file)
        tts_config = create_tts_config_from_saved(saved_config)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Read text file
        text_path = Path(text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")
        
        with open(text_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"📝 Processing {len(lines)} lines...")
        
        # Generate speech for each line
        for i, text in enumerate(lines, 1):
            print(f"🔄 [{i}/{len(lines)}] Generating: '{text[:50]}...'")
            
            output_file = output_path / f"line_{i:03d}.wav"
            result_path = generate_speech(tts_config, text, output_file)
            
            print(f"✅ Saved: {result_path}")
        
        print(f"\n🎉 Batch processing complete! Files saved in: {output_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return 1


def main():
    """Main function."""
    
    if len(sys.argv) == 1:
        # Interactive mode
        return interactive_mode()
    
    elif len(sys.argv) == 3:
        # Batch mode: config_file text_file
        config_file, text_file = sys.argv[1], sys.argv[2]
        return batch_mode(config_file, text_file)
    
    elif len(sys.argv) == 4:
        # Batch mode with custom output directory
        config_file, text_file, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
        return batch_mode(config_file, text_file, output_dir)
    
    else:
        print("Usage:")
        print("  Interactive mode: python load_voice_config.py")
        print("  Batch mode:       python load_voice_config.py config.txt texts.txt [output_dir]")
        return 1


if __name__ == "__main__":
    exit(main())
