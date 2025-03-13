from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

logger = logging.getLogger(__name__)

class FacialRecognitionAPI:
    def __init__(self, facial_recognition_system, port=5000, debug=False):
        """Initialize Flask API for the facial recognition system."""
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        self.port = port
        self.debug = debug
        self.system = facial_recognition_system
        
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
                records = self.system.database.get_collection(camera_id).find(
                    query
                ).sort("timestamp", -1).skip(skip).limit(limit)
                
                # Format the results
                formatted_records = []
                for record in records:
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
                cam01_records = list(self.system.database.get_collection("Cam-01").find(query))
                cam02_records = list(self.system.database.get_collection("Cam-02").find(query))
                
                # Combine and sort records by timestamp (newest first)
                all_records = cam01_records + cam02_records
                sorted_records = sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)
                
                # Apply pagination
                paginated_records = sorted_records[skip:skip + limit]
                
                # Format the results
                formatted_records = []
                for record in paginated_records:
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
                
    def run(self):
        """Run the Flask API server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)