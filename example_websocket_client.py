#!/usr/bin/env python3
"""
Example WebSocket Client for whisper_streaming

This example demonstrates how to connect to the whisper_streaming WebSocket server
for real-time transcription and translation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from whisper_streaming.client import TranscriptionClient, WhisperStreamingClient
except ImportError as e:
    print(f"Error importing whisper_streaming client: {e}")
    print("Make sure whisper_streaming is properly installed with WebSocket support.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def transcription_callback(text: str, segments: list) -> None:
    """
    Callback function for handling transcription results.
    
    Args:
        text: The transcribed text.
        segments: List of segment dictionaries with timing information.
    """
    print(f"\n[TRANSCRIPTION]: {text}")
    print("Segments:")
    for i, segment in enumerate(segments[-3:]):  # Show last 3 segments
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        seg_text = segment.get('text', '')
        print(f"  {i+1}. [{start:.2f}s - {end:.2f}s]: {seg_text}")
    print("-" * 50)


def main():
    """Main entry point for the example client."""
    parser = argparse.ArgumentParser(
        description="Example WebSocket client for whisper_streaming"
    )
    
    # Connection settings
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host address (default: localhost)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9090,
        help="Server port number (default: 9090)"
    )
    
    # Transcription settings
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language for transcription (default: en)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="small",
        help="Model size to use (default: small)"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate to English instead of transcribe"
    )
    
    # Input settings
    parser.add_argument(
        "--audio-file", "-f",
        type=str,
        help="Audio file to transcribe (if not provided, use microphone)"
    )
    
    # Advanced settings
    parser.add_argument(
        "--use-vad",
        action="store_true",
        default=True,
        help="Enable voice activity detection (default: enabled)"
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection"
    )
    parser.add_argument(
        "--use-callback",
        action="store_true",
        help="Use custom transcription callback for detailed output"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Handle VAD settings
    use_vad = args.use_vad and not args.no_vad
    
    # Setup transcription callback
    callback = transcription_callback if args.use_callback else None
    
    logger.info("Starting whisper_streaming WebSocket client...")
    logger.info(f"Connection: {args.host}:{args.port}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Translate: {args.translate}")
    logger.info(f"VAD enabled: {use_vad}")
    logger.info(f"Audio file: {args.audio_file or 'microphone'}")
    
    try:
        # Create client
        client = TranscriptionClient(
            host=args.host,
            port=args.port,
            language=args.language,
            model=args.model,
            translate=args.translate,
            use_vad=use_vad,
            transcription_callback=callback,
            log_transcription=not args.use_callback  # Disable default logging if using callback
        )
        
        logger.info("Client created, connecting to server...")
        
        # Start transcription
        if args.audio_file:
            if not Path(args.audio_file).exists():
                logger.error(f"Audio file not found: {args.audio_file}")
                sys.exit(1)
            logger.info(f"Transcribing audio file: {args.audio_file}")
            client(args.audio_file)
        else:
            logger.info("Starting microphone transcription...")
            logger.info("Speak into your microphone. Press Ctrl+C to stop.")
            client()
            
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
