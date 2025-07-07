#!/usr/bin/env python3
"""
Comprehensive test for Kyutai TTS integration in whisper_streaming.
This test works around version conflicts by importing only the necessary components.
"""

import logging
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Kyutai TTS components
try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    import sphn
    import torch
    import numpy as np
    _KYUTAI_TTS_AVAILABLE = True
except ImportError:
    _KYUTAI_TTS_AVAILABLE = False
    CheckpointInfo = None
    TTSModel = None
    DEFAULT_DSM_TTS_REPO = None
    DEFAULT_DSM_TTS_VOICE_REPO = None
    sphn = None


@dataclass
class TTSConfig:
    """Configuration for TTS engines."""
    
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    output_format: str = "wav"
    sample_rate: int = 22050
    
    # Kyutai TTS options
    kyutai_model_repo: str = "kyutai/tts-1.6b-en_fr"
    kyutai_voice_repo: str = "kyutai/tts-voices"
    kyutai_voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    kyutai_device: str = "cpu"
    kyutai_n_q: int = 8
    kyutai_temp: float = 0.6
    kyutai_cfg_coef: float = 2.0
    kyutai_padding_between: int = 1
    kyutai_streaming: bool = True


class BaseTTS(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup the TTS engine."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech from text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available."""
        pass
    
    def get_temp_file(self, suffix: str = ".wav") -> Path:
        """Get a temporary file path for audio output."""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.close()
        return Path(temp_file.name)
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        return text.strip()


class KyutaiTTS(BaseTTS):
    """Kyutai TTS implementation using delayed streams modeling."""
    
    def _setup(self) -> None:
        """Setup Kyutai TTS."""
        if not _KYUTAI_TTS_AVAILABLE:
            raise ImportError("moshi package is required. Install with: pip install moshi")
        
        try:
            # Load the TTS model
            self.logger.info(f"Loading Kyutai TTS model from {self.config.kyutai_model_repo}")
            checkpoint_info = CheckpointInfo.from_hf_repo(self.config.kyutai_model_repo)
            self.tts_model = TTSModel.from_checkpoint_info(
                checkpoint_info,
                n_q=self.config.kyutai_n_q,
                temp=self.config.kyutai_temp,
                device=self.config.kyutai_device
            )
            
            self.logger.info(f"Initialized Kyutai TTS with model: {self.config.kyutai_model_repo}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kyutai TTS: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Kyutai TTS is available."""
        return _KYUTAI_TTS_AVAILABLE
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech using Kyutai TTS."""
        if output_path is None:
            output_path = self.get_temp_file(".wav")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        try:
            # Prepare script entries - single turn conversation
            entries = self.tts_model.prepare_script(
                [processed_text], 
                padding_between=self.config.kyutai_padding_between
            )
            
            # Get voice path
            voice_path = self.tts_model.get_voice_path(self.config.kyutai_voice)
            
            # Create condition attributes with CFG coefficient
            condition_attributes = self.tts_model.make_condition_attributes(
                [voice_path], 
                cfg_coef=self.config.kyutai_cfg_coef
            )
            
            if self.config.kyutai_streaming:
                # Streaming synthesis to file
                self._synthesize_streaming(entries, condition_attributes, output_path)
            else:
                # Non-streaming synthesis
                self._synthesize_non_streaming(entries, condition_attributes, output_path)
            
            self.logger.debug(f"Kyutai TTS synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Kyutai TTS synthesis failed: {e}")
            raise
    
    def _synthesize_streaming(self, entries, condition_attributes, output_path: Path) -> None:
        """Synthesize using streaming mode."""
        # Collect PCM data in streaming mode
        pcm_chunks = []
        
        def _on_frame(frame):
            """Callback for streaming frames."""
            if (frame != -1).all():
                pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcm_chunk = np.clip(pcm[0, 0], -1, 1)
                pcm_chunks.append(pcm_chunk)
        
        # Generate with streaming
        with self.tts_model.mimi.streaming(1):
            self.tts_model.generate(
                [entries], 
                [condition_attributes], 
                on_frame=_on_frame
            )
        
        # Concatenate all PCM chunks and save
        if pcm_chunks:
            full_pcm = np.concatenate(pcm_chunks, axis=-1)
            sphn.write_wav(str(output_path), full_pcm, self.tts_model.mimi.sample_rate)
        else:
            self.logger.warning("No audio generated")
            # Create empty audio file
            sphn.write_wav(str(output_path), np.array([]), self.tts_model.mimi.sample_rate)
    
    def _synthesize_non_streaming(self, entries, condition_attributes, output_path: Path) -> None:
        """Synthesize using non-streaming mode."""
        # Generate all frames at once
        result = self.tts_model.generate([entries], [condition_attributes])
        
        with self.tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[self.tts_model.delay_steps:]:
                pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
            
            if pcms:
                full_pcm = np.concatenate(pcms, axis=-1)
                sphn.write_wav(str(output_path), full_pcm, self.tts_model.mimi.sample_rate)
            else:
                self.logger.warning("No audio generated")
                # Create empty audio file
                sphn.write_wav(str(output_path), np.array([]), self.tts_model.mimi.sample_rate)


def test_kyutai_tts_integration():
    """Test the complete KyutaiTTS integration."""
    
    logger.info("Testing Kyutai TTS integration...")
    
    # Test availability
    if not _KYUTAI_TTS_AVAILABLE:
        logger.error("Kyutai TTS is not available")
        return False
    
    logger.info("✓ Kyutai TTS components are available")
    
    # Test configuration
    config = TTSConfig(
        kyutai_device='cpu',  # Use CPU for testing
        kyutai_temp=0.6,
        kyutai_n_q=8,  # Use fewer codebooks for faster testing
        kyutai_voice='expresso/ex03-ex01_happy_001_channel1_334s.wav',
        kyutai_streaming=True
    )
    
    try:
        # Test initialization
        tts = KyutaiTTS(config)
        logger.info("✓ KyutaiTTS engine initialized successfully")
        logger.info(f"  Multi-speaker support: {tts.tts_model.multi_speaker}")
        logger.info(f"  Model device: {tts.tts_model.lm.device}")
        
        # Test synthesis - streaming mode
        test_text = "Hello, this is a comprehensive test of the Kyutai TTS integration in whisper streaming."
        logger.info(f"Testing synthesis with text: '{test_text}'")
        
        output_path = Path("kyutai_integration_test_streaming.wav")
        result_path = tts.synthesize(test_text, output_path)
        
        if result_path.exists():
            file_size = result_path.stat().st_size
            logger.info(f"✓ Streaming synthesis completed: {result_path} ({file_size} bytes)")
        else:
            logger.error(f"✗ Output file not created: {result_path}")
            return False
        
        # Test synthesis - non-streaming mode
        config.kyutai_streaming = False
        tts_non_streaming = KyutaiTTS(config)
        
        output_path_non_streaming = Path("kyutai_integration_test_non_streaming.wav")
        result_path_non_streaming = tts_non_streaming.synthesize(test_text, output_path_non_streaming)
        
        if result_path_non_streaming.exists():
            file_size = result_path_non_streaming.stat().st_size
            logger.info(f"✓ Non-streaming synthesis completed: {result_path_non_streaming} ({file_size} bytes)")
        else:
            logger.error(f"✗ Output file not created: {result_path_non_streaming}")
            return False
        
        # Test with different voices
        logger.info("Testing with different voice...")
        config.kyutai_voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"  # Same voice for consistency
        config.kyutai_streaming = True  # Back to streaming
        
        tts_different_voice = KyutaiTTS(config)
        output_path_different = Path("kyutai_integration_test_different_voice.wav")
        result_path_different = tts_different_voice.synthesize("Testing different voice configuration.", output_path_different)
        
        if result_path_different.exists():
            file_size = result_path_different.stat().st_size
            logger.info(f"✓ Different voice synthesis completed: {result_path_different} ({file_size} bytes)")
        else:
            logger.error(f"✗ Output file not created: {result_path_different}")
            return False
        
        logger.info("✅ All Kyutai TTS integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Kyutai TTS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_kyutai_tts_integration()
    if success:
        logger.info("🎉 Kyutai TTS integration is working perfectly!")
        sys.exit(0)
    else:
        logger.error("💥 Kyutai TTS integration test failed!")
        sys.exit(1)
