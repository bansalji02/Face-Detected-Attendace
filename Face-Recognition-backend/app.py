# import threading
# import logging
# import time
# import argparse
# from config import (
#     MONGO_URI, MONGO_DB, DETECTION_COOLDOWN, 
#     KNOWN_FACES_DIR, API_PORT, DEBUG_MODE,
#     DEFAULT_CAMERA, DEFAULT_LOCATION, DEFAULT_CAMERA_ID
# )
# from database import Database
# from facial_recognition import FacialRecognitionSystem
# from api import FacialRecognitionAPI

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("facial_recognition.log"),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# def start_api_server(facial_recognition_system, port, debug):
#     """Start the API server in a separate thread."""
#     api = FacialRecognitionAPI(facial_recognition_system, port, debug)
#     logger.info(f"Starting API server on port {port}")
#     # Using threaded=True instead of relying on werkzeug's signal handling
#     api.run()
#     # api.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False, threaded=True)

# def parse_arguments():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Facial Recognition System")
#     parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, 
#                         help="Camera index to use (default: 0)")
#     parser.add_argument("--location", type=str, default=DEFAULT_LOCATION,
#                         help="Location description for the camera")
#     parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID,
#                         help="Unique identifier for the camera")
#     parser.add_argument("--api-only", action="store_true",
#                         help="Run only the API server without video processing")
#     parser.add_argument("--no-api", action="store_true",
#                         help="Run only video processing without the API server")
#     parser.add_argument("--port", type=int, default=API_PORT,
#                         help=f"Port for the API server (default: {API_PORT})")
#     parser.add_argument("--mode", type=str, default="cooldown",
#                         choices=["cooldown", "once_per_day", "always"],
#                         help="Deduplication mode for detections")
#     parser.add_argument("--cooldown", type=int, default=DETECTION_COOLDOWN,
#                         help=f"Cooldown period in seconds (default: {DETECTION_COOLDOWN})")
    
#     return parser.parse_args()



# # Modified FacialRecognitionSystem usage in main function
# def main():
#     """Main entry point for the application."""
#     args = parse_arguments()
    
#     try:
#         # Initialize database
#         logger.info(f"Connecting to MongoDB at {MONGO_URI}")
#         db = Database(MONGO_URI, MONGO_DB)
        
#         # Initialize facial recognition system
#         logger.info("Initializing Facial Recognition System")
#         system = FacialRecognitionSystem(
#             database=db,
#             known_faces_dir=KNOWN_FACES_DIR,
#             detection_cooldown=args.cooldown
#         )
        
#         # Start API server if not disabled
#         if not args.no_api:
#             api_thread = threading.Thread(
#                 target=start_api_server,
#                 args=(system, args.port, DEBUG_MODE),
#                 daemon=True
#             )
#             api_thread.start()
#             logger.info("API server thread started")
        
#         # Run video processing if not in API-only mode
#         if not args.api_only:
#             logger.info(f"Starting video processing from camera {args.camera}")
#             logger.info(f"Camera location: {args.location}")
#             # Use process_video or the correct method name that exists in your FacialRecognitionSystem class
#             system.process_video_stream(
#                 # camera_index=args.camera,
#                 location=args.location,
#                 camera_id=args.camera_id,
#                 # detection_mode=args.mode
#             )
#         else:
#             logger.info("Running in API-only mode, no video processing")
#             # Keep the main thread alive if only running the API
#             while True:
#                 time.sleep(1)
                
#     except KeyboardInterrupt:
#         logger.info("Application shutdown requested")
#     except Exception as e:
#         logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
#     finally:
#         logger.info("Shutting down application")
#         # Perform any necessary cleanup
#         if hasattr(system, 'stop_video_processing'):
#             system.stop_video_processing()
#         elif hasattr(system, 'stop_processing'):
#             system.stop_processing()
#         logger.info("Application shutdown complete")

# if __name__ == "__main__":
#     main()



import threading
import logging
import time
import argparse
import cv2
import sys
import os
from config import (
    MONGO_URI, MONGO_DB, DETECTION_COOLDOWN, 
    KNOWN_FACES_DIR, API_PORT, DEBUG_MODE,
    DEFAULT_CAMERA, DEFAULT_LOCATION, DEFAULT_CAMERA_ID
)
from database import Database
from facial_recognition import FacialRecognitionSystem
from api import FacialRecognitionAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facial_recognition.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def start_api_server(facial_recognition_system, port, debug):
    """Start the API server in a separate thread."""
    api = FacialRecognitionAPI(facial_recognition_system, port, debug)
    logger.info(f"Starting API server on port {port}")
    # Call run() without any parameters - adjust based on your API class implementation
    api.run()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, 
                        help="Camera index to use (default: 0)")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION,
                        help="Location description for the camera")
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID,
                        help="Unique identifier for the camera")
    parser.add_argument("--api-only", action="store_true",
                        help="Run only the API server without video processing")
    parser.add_argument("--no-api", action="store_true",
                        help="Run only video processing without the API server")
    parser.add_argument("--port", type=int, default=API_PORT,
                        help=f"Port for the API server (default: {API_PORT})")
    parser.add_argument("--mode", type=str, default="cooldown",
                        choices=["cooldown", "once_per_day", "always"],
                        help="Deduplication mode for detections")
    parser.add_argument("--cooldown", type=int, default=DETECTION_COOLDOWN,
                        help=f"Cooldown period in seconds (default: {DETECTION_COOLDOWN})")
    parser.add_argument("--rtsp-url", type=str, 
                        help="RTSP URL for IP camera (overrides --camera if provided)")
    parser.add_argument("--subtype", type=int, default=0,
                        help="RTSP subtype (0=high quality, 1=lower quality)")
    return parser.parse_args()

def setup_rtsp_stream(args):
    """Set up the RTSP stream with appropriate settings."""
    if args.rtsp_url:
        # Use the provided RTSP URL
        rtsp_url = args.rtsp_url
    else:
        # Construct RTSP URL from components
        # Default format for many CCTV systems
        rtsp_url = f"rtsp://testcam:Testcam12345@10.5.0.51:80/cam/realmonitor?channel=12&subtype={args.subtype}"
    
    logger.info(f"Using RTSP URL with subtype={args.subtype}: {rtsp_url}")
    
    # Pre-configure OpenCV for better RTSP handling
    # These settings can help with stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|buffer_size;10485760"
    
    return rtsp_url

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    try:
        # Initialize database
        logger.info(f"Connecting to MongoDB at {MONGO_URI}")
        db = Database(MONGO_URI, MONGO_DB)
        
        # Initialize facial recognition system
        logger.info("Initializing Facial Recognition System")
        system = FacialRecognitionSystem(
            database=db,
            known_faces_dir=KNOWN_FACES_DIR,
            detection_cooldown=args.cooldown
        )
        
        # Start API server if not disabled
        if not args.no_api:
            api_thread = threading.Thread(
                target=start_api_server,
                args=(system, args.port, False),  # Force debug to False to avoid reloader issues
                daemon=True
            )
            api_thread.start()
            logger.info("API server thread started")
        
        # Run video processing if not in API-only mode
        if not args.api_only:
            logger.info(f"Starting video processing from camera {args.camera}")
            logger.info(f"Camera location: {args.location}")
            
            # Get appropriate video source
            if hasattr(args, 'rtsp_url') and args.rtsp_url:
                video_source = args.rtsp_url
            else:
                # Set up RTSP stream with appropriate settings
                video_source = setup_rtsp_stream(args)
            
            # Configure OpenCV to prefer FFMPEG backend for video capture
            cv2.setUseOptimized(True)
            
            # Call process_video_stream with only the parameters it accepts
            # Based on the error, we need to remove the extra parameters
            system.process_video_stream(
                video_source=video_source,
                location=args.location,
                camera_id=args.camera_id,
                deduplication_mode=args.mode
            )
        else:
            logger.info("Running in API-only mode, no video processing")
            # Keep the main thread alive if only running the API
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    finally:
        logger.info("Shutting down application")
        # Perform any necessary cleanup
        if 'system' in locals():
            if hasattr(system, 'stop_video_processing'):
                system.stop_video_processing()
            elif hasattr(system, 'stop_processing'):
                system.stop_processing()
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    # Only attempt to handle signals in the main thread/process
    if threading.current_thread() is threading.main_thread():
        try:
            # Handle SIGTERM gracefully
            import signal
            signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
            signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
            logger.info("Signal handlers registered in main thread")
        except (ImportError, ValueError) as e:
            # Log but don't crash if signal handling isn't available
            logger.warning(f"Could not set up signal handlers: {e}")
    
    main()