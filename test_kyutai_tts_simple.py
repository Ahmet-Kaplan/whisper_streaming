#!/usr/bin/env python3
"""
Simple test script for Kyutai TTS integration.
Tests the KyutaiTTS class functionality directly.
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_kyutai_tts():
    """Test Kyutai TTS functionality."""
    try:
        # Import Kyutai TTS components directly
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
        import sphn
        import torch
        import numpy as np
        
        logger.info("Successfully imported Kyutai TTS components")
        
        # Create TTS model
        logger.info(f"Loading model from {DEFAULT_DSM_TTS_REPO}")
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
            n_q=8,  # Fewer codebooks for faster inference
            temp=0.6,
            device='cpu',  # Use CPU to avoid CUDA issues
        )
        
        logger.info("TTS model loaded successfully")
        logger.info(f"Multi-speaker support: {tts_model.multi_speaker}")
        
        # Test text to synthesize
        test_text = "Hello, this is a test of Kyutai TTS integration."
        
        # Prepare script entries
        entries = tts_model.prepare_script([test_text], padding_between=1)
        logger.info(f"Prepared {len(entries)} entries for synthesis")
        
        # Get voice path
        voice_name = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        voice_path = tts_model.get_voice_path(voice_name)
        logger.info(f"Using voice: {voice_path}")
        
        # Create condition attributes
        condition_attributes = tts_model.make_condition_attributes([voice_path])
        
        # Generate audio
        logger.info("Starting synthesis...")
        result = tts_model.generate([entries], [condition_attributes])
        
        logger.info(f"Generated {len(result.frames)} frames")
        
        # Decode audio
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[tts_model.delay_steps:]:
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
        
        if pcms:
            full_pcm = np.concatenate(pcms, axis=-1)
            output_path = Path("kyutai_test_output.wav")
            sphn.write_wav(str(output_path), full_pcm, tts_model.mimi.sample_rate)
            logger.info(f"Audio saved to: {output_path}")
            logger.info(f"Audio duration: {len(full_pcm) / tts_model.mimi.sample_rate:.2f} seconds")
            return True
        else:
            logger.error("No audio generated")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kyutai_tts()
    if success:
        logger.info("✅ Kyutai TTS test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Kyutai TTS test failed!")
        sys.exit(1)
