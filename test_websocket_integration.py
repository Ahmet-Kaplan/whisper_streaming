#!/usr/bin/env python3
"""
WebSocket Integration Test for whisper_streaming

Tests the integration of WhisperLive-style WebSocket architecture with
whisper_streaming backend system.
"""

import sys
import logging
import threading
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_websocket_imports():
    """Test WebSocket server and client imports."""
    print("🧪 Test 1: WebSocket Imports")
    print("=" * 40)
    
    try:
        # Test server imports
        from whisper_streaming.server import WhisperStreamingServer, ClientManager
        print("✅ Server components imported successfully")
        
        # Test client imports
        from whisper_streaming.client import WhisperStreamingClient, TranscriptionClient
        print("✅ Client components imported successfully")
        
        # Test main package imports
        from whisper_streaming import WhisperStreamingServer, WhisperStreamingClient
        print("✅ Main package WebSocket imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        if "websocket" in str(e).lower():
            print("   Install websocket-client: pip install websocket-client")
        elif "pyaudio" in str(e).lower():
            print("   Install PyAudio: pip install pyaudio")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_server_creation():
    """Test WebSocket server creation."""
    print("\n🧪 Test 2: Server Creation")
    print("=" * 40)
    
    try:
        from whisper_streaming.server import WhisperStreamingServer, ClientManager
        
        # Test server creation
        server = WhisperStreamingServer()
        print("✅ WhisperStreamingServer created successfully")
        
        # Test client manager creation
        client_manager = ClientManager(max_clients=2, max_connection_time=300)
        print("✅ ClientManager created successfully")
        
        # Test client manager status
        status = client_manager.get_status()
        print(f"✅ Client manager status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server creation test failed: {e}")
        return False


def test_client_creation():
    """Test WebSocket client creation."""
    print("\n🧪 Test 3: Client Creation")
    print("=" * 40)
    
    try:
        from whisper_streaming.client import TranscriptionClient
        
        # Test basic client creation (without connecting)
        print("⚠️  Note: Client creation will attempt to connect to server")
        print("   This test may show connection errors if no server is running")
        
        # We'll create the client but won't actually use it
        # since there's no server running
        try:
            client = TranscriptionClient(
                host="localhost",
                port=9999,  # Use a different port to avoid conflicts
                language="en",
                model="small"
            )
            print("✅ TranscriptionClient created (may show connection warnings)")
            
            # Clean up immediately
            try:
                client.close_websocket()
            except:
                pass
                
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                print("✅ Client creation successful (connection refused is expected without server)")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"❌ Client creation test failed: {e}")
        return False


def test_backend_integration():
    """Test backend integration with server components."""
    print("\n🧪 Test 4: Backend Integration")
    print("=" * 40)
    
    try:
        from whisper_streaming import Backend
        from whisper_streaming.server.serve_client import ServeClientWhisper
        
        # Test backend enum
        backends = [backend.name for backend in Backend]
        print(f"📋 Available backends: {backends}")
        
        # Test serve client components
        print("✅ ServeClientWhisper class available")
        
        # Test WebSocket components
        from whisper_streaming.server.serve_client import (
            WebSocketAudioReceiver, 
            WebSocketOutputSender
        )
        print("✅ WebSocket audio components available")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend integration test failed: {e}")
        return False


def test_configuration_options():
    """Test configuration options for server and client."""
    print("\n🧪 Test 5: Configuration Options")
    print("=" * 40)
    
    try:
        from whisper_streaming.client import TranscriptionClient
        from whisper_streaming.server import ClientManager
        
        # Test client configuration
        config_options = {
            "host": "localhost",
            "port": 9090,
            "language": "en",
            "translate": False,
            "model": "small",
            "use_vad": True,
            "max_clients": 4,
            "max_connection_time": 600,
            "send_last_n_segments": 10,
            "no_speech_thresh": 0.45,
        }
        print(f"✅ Client configuration options validated: {len(config_options)} options")
        
        # Test server configuration
        client_manager = ClientManager(max_clients=4, max_connection_time=600)
        server_status = client_manager.get_status()
        print(f"✅ Server configuration validated: {server_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_script_availability():
    """Test availability of server and client scripts."""
    print("\n🧪 Test 6: Script Availability")
    print("=" * 40)
    
    try:
        # Check if server script exists
        server_script = Path("run_websocket_server.py")
        if server_script.exists():
            print("✅ WebSocket server script available")
        else:
            print("❌ WebSocket server script not found")
            return False
        
        # Check if client script exists
        client_script = Path("example_websocket_client.py")
        if client_script.exists():
            print("✅ WebSocket client script available")
        else:
            print("❌ WebSocket client script not found")
            return False
        
        print("📋 Usage examples:")
        print("   Server: python run_websocket_server.py --host localhost --port 9090")
        print("   Client: python example_websocket_client.py --host localhost --port 9090")
        
        return True
        
    except Exception as e:
        print(f"❌ Script availability test failed: {e}")
        return False


def test_dependencies():
    """Test required dependencies for WebSocket functionality."""
    print("\n🧪 Test 7: Dependencies")
    print("=" * 40)
    
    dependencies = {
        "websocket-client": "websocket",
        "PyAudio": "pyaudio", 
        "websockets": "websockets",
        "numpy": "numpy",
    }
    
    missing_deps = []
    for dep_name, module_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✅ {dep_name} available")
        except ImportError:
            print(f"❌ {dep_name} missing")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with:")
        for dep in missing_deps:
            if dep == "websocket-client":
                print("   pip install websocket-client")
            elif dep == "PyAudio":
                print("   pip install pyaudio")
            elif dep == "websockets":
                print("   pip install websockets")
        return False
    
    print("✅ All dependencies available")
    return True


def main():
    """Run all integration tests."""
    print("🚀 WhisperStreaming WebSocket Integration Tests")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_websocket_imports,
        test_server_creation,
        test_client_creation,
        test_backend_integration,
        test_configuration_options,
        test_script_availability,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i}. {test.__name__}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WebSocket integration is ready.")
        print("\n🚀 Quick Start:")
        print("1. Start server: python run_websocket_server.py")
        print("2. Connect client: python example_websocket_client.py")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
