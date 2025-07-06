#!/usr/bin/env python3
"""
WebSocket Server for whisper_streaming

Real-time transcription and translation server using WebSocket protocol.
This server integrates WhisperLive's architecture with whisper_streaming's
modular backend system.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from whisper_streaming.server import WhisperStreamingServer
    from whisper_streaming import Backend
except ImportError as e:
    print(f"Error importing whisper_streaming: {e}")
    print("Make sure whisper_streaming is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the WebSocket server."""
    parser = argparse.ArgumentParser(
        description="WebSocket server for real-time transcription using whisper_streaming"
    )
    
    # Server configuration
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host address to bind the server (default: localhost)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int, 
        default=9090,
        help="Port number to bind the server (default: 9090)"
    )
    
    # Backend configuration  
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["faster_whisper", "whisperx"],
        default="faster_whisper",
        help="Backend to use for transcription (default: faster_whisper)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="small",
        help="Model size to use (default: small)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Default language for transcription (default: en)"
    )
    
    # Server limits
    parser.add_argument(
        "--max-clients",
        type=int,
        default=4,
        help="Maximum number of concurrent clients (default: 4)"
    )
    parser.add_argument(
        "--max-connection-time",
        type=int,
        default=600,
        help="Maximum connection time per client in seconds (default: 600)"
    )
    
    # Advanced options
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # WhisperX specific options
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker diarization (WhisperX only)"
    )
    parser.add_argument(
        "--huggingface-token",
        type=str,
        help="HuggingFace token for diarization (WhisperX only)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Handle VAD settings
    use_vad = args.use_vad and not args.no_vad
    
    # Validate backend
    backend_map = {
        "faster_whisper": Backend.FASTER_WHISPER,
        "whisperx": Backend.WHISPERX
    }
    
    if args.backend not in backend_map:
        logger.error(f"Unsupported backend: {args.backend}")
        sys.exit(1)
    
    backend = backend_map[args.backend]
    
    # Check WhisperX specific requirements
    if args.backend == "whisperx":
        try:
            from whisper_streaming.backend import WhisperXASR
            logger.info("WhisperX backend available")
        except ImportError:
            logger.error("WhisperX backend not available. Install whisperx to use this backend.")
            sys.exit(1)
            
        if args.enable_diarization and not args.huggingface_token:
            logger.warning("Diarization enabled but no HuggingFace token provided. Some features may not work.")
    
    logger.info(f"Starting WhisperStreaming WebSocket server...")
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Language: {args.language}")
    logger.info(f"  Max clients: {args.max_clients}")
    logger.info(f"  Max connection time: {args.max_connection_time}s")
    logger.info(f"  VAD enabled: {use_vad}")
    
    if args.backend == "whisperx":
        logger.info(f"  Diarization enabled: {args.enable_diarization}")
    
    try:
        # Create and start server
        server = WhisperStreamingServer()
        logger.info(f"Server starting on {args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start server
        server.run(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
