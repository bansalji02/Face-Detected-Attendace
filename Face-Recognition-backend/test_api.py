from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import pymongo
import datetime
import os
import face_recognition
import numpy as np
import base64
import cv2
import shutil
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# KNOWN_FACES_DIR = "/known_faces"

# Mock database class for testing
class MockDatabase:
    def __init__(self, connection_string):
        """Initialize connection to MongoDB."""
        try:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client["face_recognition"]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_collection(self, collection_name):
        """Get a collection by name."""
        return self.db[collection_name]

# Standalone API class
class FacialRecognitionAPI:
    def __init__(self, database, port=5000, debug=False):
        """Initialize Flask API for the facial recognition system."""
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        self.port = port
        self.debug = debug
        self.database = database
        
        # Directory to store face encodings
        self.KNOWN_FACES_DIR = os.path.abspath("known_faces")
        if not os.path.exists(self.KNOWN_FACES_DIR):
            os.makedirs(self.KNOWN_FACES_DIR)
        
        # Register API routes
        self.register_routes()
        
    def register_routes(self):
        """Register all API endpoints."""
        
        @self.app.route('/api/camera/<camera_id>', methods=['GET'])
        def get_camera_records(camera_id):
            """Get all records from a specific camera within a date range."""
            try:
                # Validate camera ID
                if camera_id not in ["Cam-01", "Cam-02"]:
                    return jsonify({
                        "status": "error",
                        "message": f"Invalid camera ID: {camera_id}. Valid options are 'Cam-01' and 'Cam-02'."
                    }), 400
                
                # Get optional query parameters
                limit = request.args.get('limit', default=100, type=int)
                skip = request.args.get('skip', default=0, type=int)
                from_date = request.args.get('fromDate', default=None, type=str)
                to_date = request.args.get('toDate', default=None, type=str)
                
                # Prepare the query
                query = {"camera_id": camera_id}
                if from_date and to_date:
                    query["date_str"] = {"$gte": from_date, "$lte": to_date}
                elif from_date:
                    query["date_str"] = {"$gte": from_date}
                elif to_date:
                    query["date_str"] = {"$lte": to_date}
                    
                # Query the database for camera records
                records = self.database.get_collection(camera_id).find(
                    query
                ).sort("timestamp", -1).skip(skip).limit(limit)
                
                # Format the results
                formatted_records = []
                for record in records:
                    # Convert ObjectId to string for JSON serialization
                    if '_id' in record:
                        record['_id'] = str(record['_id'])
                        
                    formatted_records.append({
                        "person": record.get("person_name"),
                        "timestamp": record.get("timestamp"),
                        "date": record.get("date_str"),
                        "location": record.get("location"),
                        "confidence": record.get("confidence"),
                        "camera": record.get("camera_id")
                    })
                
                return jsonify({
                    "status": "success",
                    "camera": camera_id,
                    "records": formatted_records,
                    "count": len(formatted_records)
                })
                
            except Exception as e:
                logger.error(f"Error in get_camera_records: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500

        @self.app.route('/api/person/<person_name>', methods=['GET'])
        def get_person_records(person_name):
            """Get all records for a specific person from both cameras within a date range, sorted by timestamp."""
            try:
                # Get optional query parameters
                limit = request.args.get('limit', default=100, type=int)
                skip = request.args.get('skip', default=0, type=int)
                from_date = request.args.get('fromDate', default=None, type=str)
                to_date = request.args.get('toDate', default=None, type=str)
                
                # Prepare query
                query = {"person_name": person_name}
                if from_date and to_date:
                    query["date_str"] = {"$gte": from_date, "$lte": to_date}
                elif from_date:
                    query["date_str"] = {"$gte": from_date}
                elif to_date:
                    query["date_str"] = {"$lte": to_date}
                
                # Get records from both cameras
                cam01_records = list(self.database.get_collection("Cam-01").find(query))
                cam02_records = list(self.database.get_collection("Cam-02").find(query))
                
                # Combine and sort records by timestamp (newest first)
                all_records = cam01_records + cam02_records
                sorted_records = sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)
                
                # Apply pagination
                paginated_records = sorted_records[skip:skip + limit]
                
                # Format the results
                formatted_records = []
                for record in paginated_records:
                    # Convert ObjectId to string for JSON serialization
                    if '_id' in record:
                        record['_id'] = str(record['_id'])
                        
                    formatted_records.append({
                        "person": record.get("person_name"),
                        "timestamp": record.get("timestamp"),
                        "date": record.get("date_str"),
                        "location": record.get("location"),
                        "confidence": record.get("confidence"),
                        "camera": record.get("camera_id")
                    })
                
                return jsonify({
                    "status": "success",
                    "person": person_name,
                    "records": formatted_records,
                    "total_records": len(sorted_records),
                    "returned_records": len(formatted_records)
                })
                
            except Exception as e:
                logger.error(f"Error in get_person_records: {e}")
                return jsonify({
                    "status": "error",          
                    "message": str(e)
                }), 500

        @self.app.route('/api/register-face', methods=['POST'])
        def register_face():
            try:
                data = request.json
                logger.info("Received registration request")

                if 'image' not in data or 'personName' not in data or 'personId' not in data:
                    logger.error("Missing required fields")
                    return jsonify({"error": "Missing required fields"}), 400

                image_data = data['image']
                person_name = data.get('personName', '')
                person_id = data.get('personId', '')

                logger.info(f"Processing registration for {person_name} (ID: {person_id})")

                person_dir = os.path.join(self.KNOWN_FACES_DIR, f"{person_name}_{person_id}")
                if os.path.exists(person_dir):
                    shutil.rmtree(person_dir)
                os.makedirs(person_dir)

                try:
                    if not image_data.startswith('data:image'):
                        logger.error("Invalid image data format")
                        return jsonify({"error": "Invalid image data format"}), 400

                    image_bytes = base64.b64decode(image_data.split(',')[1])
                    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                    if image is None:
                        logger.error("Failed to decode image")
                        return jsonify({"error": "Failed to process image"}), 400

                    logger.info(f"Image decoded successfully. Shape: {image.shape}")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    return jsonify({"error": f"Image processing error: {str(e)}"}), 400

                debug_img_path = os.path.join(person_dir, "debug_original.jpg")
                cv2.imwrite(debug_img_path, image)

                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    logger.info(f"Detected {len(face_locations)} faces")

                    if len(face_locations) == 0:
                        logger.error("No face detected in the image")
                        return jsonify({"error": "No face detected in the image"}), 400

                    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                    logger.info("Successfully extracted face encoding")
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
                    return jsonify({"error": f"Face detection failed: {str(e)}"}), 400

                angles = ["frontal", "slight_left", "profile_left", "slight_right", "profile_right", "tilted_up", "tilted_down"]
                num_files = len([f for f in os.listdir(person_dir) if f.endswith('.npy') and f != "debug_original.jpg"])

                if num_files >= len(angles):
                    logger.error("All required angles have already been captured")
                    return jsonify({"error": "All required angles have already been captured"}), 400

                angle_label = angles[num_files]
                logger.info(f"Assigning angle label: {angle_label}")

                encoding_path = os.path.join(person_dir, f"{angle_label}.npy")
                np.save(encoding_path, face_encoding)

                image_path = os.path.join(person_dir, f"{angle_label}.jpg")
                cv2.imwrite(image_path, image)

                metadata_path = os.path.join(person_dir, "metadata.json")

                if num_files == 0:
                    metadata = {
                        "name": person_name,
                        "id": person_id,
                        "registration_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "angles_captured": [angle_label]
                    }
                else:
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        metadata["angles_captured"].append(angle_label)
                        metadata["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logger.error(f"Error reading metadata: {e}")
                        metadata = {
                            "name": person_name,
                            "id": person_id,
                            "registration_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "angles_captured": [angle_label]
                        }

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                logger.info(f"Successfully registered face with angle: {angle_label}")
                return jsonify({
                    "status": "success",
                    "angle": angle_label,
                    "remaining_angles": len(angles) - num_files - 1
                })

            except Exception as e:
                logger.error(f"Error in register_face: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
    def run(self):
        """Run the Flask API server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)

# Main entry point for direct execution
if __name__ == "__main__":
    # Use environment variables or default to localhost MongoDB
    mongo_uri = os.environ.get("MONGO_URI")
    
    # Create database connection
    try:
        db = MockDatabase(mongo_uri)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        exit(1)
    
    # Create and run API
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    api = FacialRecognitionAPI(db, port=port, debug=debug)
    logger.info(f"Starting API server on port {port}")
    api.run()



