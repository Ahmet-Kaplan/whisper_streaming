#!/usr/bin/env python3
"""
Piper TTS Integration Example

This example demonstrates how to use the integrated Piper TTS engine
for Turkish text-to-speech synthesis in the whisper_streaming project.

Installation:
    pip install piper-tts

Usage:
    python examples/piper_tts_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_streaming.tts import (
    TTSEngine,
    TTSConfig,
    PiperTTS,
    create_tts_engine,
    get_available_engines,
    get_best_tts_for_turkish,
    synthesize_turkish,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_piper_features():
    """Demonstrate various Piper TTS features."""
    
    print("🎤 Piper TTS Integration Example")
    print("=" * 40)
    
    # Check available engines
    print("\\n1. Available TTS Engines:")
    available = get_available_engines()
    for engine in available:
        print(f"   ✓ {engine.value}")
    
    if TTSEngine.PIPER_TTS not in available:
        print("\\n❌ Piper TTS is not available.")
        print("   Please install it with: pip install piper-tts")
        print("\\n   Falling back to other available engines...")
        demonstrate_fallback()
        return
    
    print("\\n✅ Piper TTS is available!")
    
    # Show best engine selection
    print("\\n2. Best Engine Selection:")
    offline_engine, offline_reason = get_best_tts_for_turkish(prefer_offline=True)
    online_engine, online_reason = get_best_tts_for_turkish(prefer_offline=False)
    
    print(f"   Offline preference: {offline_engine.value} - {offline_reason}")
    print(f"   Online preference:  {online_engine.value} - {online_reason}")
    
    # Demonstrate different Piper models
    print("\\n3. Piper Turkish Models:")
    models = {
        "dfki-medium": "tr_TR-dfki-medium (Best quality)",
        "dfki-low": "tr_TR-dfki-low (Faster, smaller)",
        "fgl-medium": "tr_TR-fgl-medium (Alternative voice)",
    }
    
    for short_name, description in models.items():
        print(f"   • {short_name:12} -> {description}")
    
    # Demonstrate Piper synthesis
    print("\\n4. Piper TTS Synthesis Examples:")
    
    examples = [
        ("Basic Turkish", "Merhaba! Bu Piper TTS ile oluşturulmuş bir ses örneğidir."),
        ("Numbers & Dates", "Bugün 6 Temmuz 2025 tarihinde test yapıyoruz."),
        ("Technical Terms", "Whisper streaming projesi ile Piper TTS entegrasyonu başarılı."),
        ("Mixed Content", "Bu T.C. Cumhurbaşkanlığı vs. diğer kurumlar A.Ş. gibi kısaltmalar içeriyor."),
    ]
    
    for name, text in examples:
        print(f"\\n   🎵 {name}:")
        print(f"      Text: \\\"{text}\\\"")
        
        try:
            # Create Piper TTS with different configurations
            config = TTSConfig(
                language="tr",
                piper_model="dfki-medium",
                speed=1.0,
                use_turkish_phonetics=True,
                handle_foreign_words=True,
            )
            
            tts = PiperTTS(config)
            output_path = tts.synthesize(text)
            
            print(f"      ✅ Generated: {output_path}")
            print(f"      📁 Size: {output_path.stat().st_size:,} bytes")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Demonstrate convenience function
    print("\\n5. Convenience Function:")
    try:
        output_path = synthesize_turkish(
            text="Bu basit kullanım örneğidir.",
            engine=TTSEngine.PIPER_TTS,
            voice_preference="female",  # Not used by Piper but part of interface
            speed=1.2
        )
        print(f"   ✅ Quick synthesis: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Configuration options
    print("\\n6. Configuration Options:")
    config_options = {
        "piper_model": "tr_TR-dfki-medium",
        "piper_data_dir": "~/.local/share/piper",
        "piper_download_dir": "~/.local/share/piper",
        "speed": 1.0,
        "use_turkish_phonetics": True,
        "handle_foreign_words": True,
    }
    
    print("   Configuration options for Piper TTS:")
    for key, value in config_options.items():
        print(f"     {key:25} = {value}")


def demonstrate_fallback():
    """Demonstrate using other TTS engines when Piper is not available."""
    
    print("\\n🔄 Fallback TTS Demonstration")
    print("-" * 30)
    
    # Get best available engine
    try:
        engine, reason = get_best_tts_for_turkish(prefer_offline=True)
        print(f"   Using: {engine.value} - {reason}")
        
        # Test with convenience function
        test_text = "Piper mevcut olmadığı için alternatif TTS kullanıyoruz."
        
        output_path = synthesize_turkish(
            text=test_text,
            engine=TTSEngine.AUTO,
            speed=1.0
        )
        
        print(f"   ✅ Generated with {engine.value}: {output_path}")
        print(f"   📁 Size: {output_path.stat().st_size:,} bytes")
        
    except Exception as e:
        print(f"   ❌ Fallback failed: {e}")


def show_installation_guide():
    """Show installation guide for Piper TTS."""
    
    print("\\n📦 Piper TTS Installation Guide")
    print("=" * 35)
    
    print("\\n1. Install Piper TTS:")
    print("   pip install piper-tts")
    
    print("\\n2. Install in virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate")
    print("   pip install piper-tts")
    
    print("\\n3. Available Turkish models (auto-downloaded):")
    print("   • tr_TR-dfki-medium    - High quality, medium size")
    print("   • tr_TR-dfki-low       - Lower quality, smaller size")  
    print("   • tr_TR-fgl-medium     - Alternative voice")
    
    print("\\n4. Model storage location:")
    print("   ~/.local/share/piper/")
    
    print("\\n5. Integration features:")
    print("   ✓ Automatic model downloading")
    print("   ✓ Turkish text preprocessing")
    print("   ✓ Multiple model support")
    print("   ✓ Configurable data directories")
    print("   ✓ Offline operation")
    print("   ✓ Fast synthesis")


def main():
    """Main function."""
    
    show_installation_guide()
    
    print("\\n" + "=" * 50)
    
    try:
        demonstrate_piper_features()
        
    except KeyboardInterrupt:
        print("\\n\\n👋 Example interrupted by user.")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\\n❌ Error running example: {e}")
        return 1
    
    print("\\n" + "=" * 50)
    print("🎉 Piper TTS integration example completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
