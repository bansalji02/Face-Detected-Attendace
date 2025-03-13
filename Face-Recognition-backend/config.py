

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# MongoDB configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://iAmBansal:Bansalji999@himanshulearn.lcenhzn.mongodb.net/?retryWrites=true&w=majority&appName=HimanshuLearn")
MONGO_DB = os.environ.get("MONGO_DB", "face_recognition")

# Face recognition settings
DETECTION_COOLDOWN = int(os.environ.get("DETECTION_COOLDOWN", "60"))
KNOWN_FACES_DIR = os.environ.get("KNOWN_FACES_DIR", "known_faces")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")) 

# API settings
API_PORT = int(os.environ.get("API_PORT", "5000"))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "True").lower() == "true"

# Camera settings
DEFAULT_CAMERA = int(os.environ.get("DEFAULT_CAMERA", 0))
DEFAULT_LOCATION = os.environ.get("DEFAULT_LOCATION", "Office Entrance")
DEFAULT_CAMERA_ID = os.environ.get("DEFAULT_CAMERA_ID", "Cam-02")

KNOWN_FACES_DIR = "/known_faces"







# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import datetime
# # from facial_recognition_system import FacialRecognitionSystem
# from app import FacialRecognitionSystem


# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Initialize the facial recognition system
# system = FacialRecognitionSystem(
#     mongo_uri="mongodb+srv://iAmBansal:Bansalji999@himanshulearn.lcenhzn.mongodb.net/?retryWrites=true&w=majority&appName=HimanshuLearn",
#     mongo_db="face_recognition",
#     # mongo_collection="detections",
#     detection_cooldown=60,
# )

# @app.route('/api/people', methods=['GET'])
# def get_people():
#     """Get list of all known people."""
#     return jsonify({
#         "people": system.known_face_names
#     })

# @app.route('/api/presence/current', methods=['GET'])
# def get_current_presence():
#     """Get list of people currently in the office."""
#     try:
#         occupants = system.get_office_presence_data()
#         return jsonify({
#             "status": "success",
#             "occupants": occupants,
#             "count": len(occupants)
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/api/presence/person/<name>', methods=['GET'])
# def get_person_presence(name):
#     """Get presence history for a specific person."""
#     try:
#         limit = request.args.get('limit', default=10, type=int)
#         history = system.get_office_presence_data(name, limit)
#         return jsonify({
#             "status": "success",
#             "person": name,
#             "history": history
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/api/presence/summary', methods=['GET'])
# def get_presence_summary():
#     """Get summary of office presence for a specific date."""
#     try:
#         date = request.args.get('date', default=None)
#         summary = system.get_office_summary(date)
#         return jsonify({
#             "status": "success",
#             "summary": summary
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/api/presence/daily', methods=['GET'])
# def get_daily_presence():
#     """Get presence data for a specific date with details of each entry/exit."""
#     try:
#         date = request.args.get('date', default=None)
#         if date is None:
#             date = datetime.date.today().strftime("%Y-%m-%d")
            
#         # This requires a new method in OfficePresenceTracker to get all presence records for a day
#         # We'll need to implement this
#         # For now, we'll return a placeholder
#         return jsonify({
#             "status": "success",
#             "date": date,
#             "message": "This endpoint needs implementation"
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)