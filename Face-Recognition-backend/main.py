#---------------------This is the code that includes alternative to the app.py file with an option to sqlite db-------------#







# # import cv2
# import face_recognition
# import numpy as np
# import datetime
# import sqlite3
# import os
# from typing import List, Dict, Tuple, Optional
# import pymongo
# from pymongo import MongoClient



# class FacialRecognitionSystem:
#     def __init__(self, database_type: str = "sqlite", 
#                  sqlite_path: str = "face_records.db", 
#                  mongo_uri: str = "mongodb://localhost:27017/",
#                  mongo_db: str = "face_recognition",
#                  mongo_collection: str = "detections",
#                  known_faces_dir: str = "known_faces",
#                  detection_cooldown: int = 60):
#         """
#         Initialize the facial recognition system.
        
#         Args:
#             database_type: Type of database to use ("sqlite" or "mongodb")
#             sqlite_path: Path to SQLite database if using SQLite
#             mongo_uri: MongoDB connection URI if using MongoDB
#             mongo_db: MongoDB database name if using MongoDB
#             mongo_collection: MongoDB collection name if using MongoDB
#             known_faces_dir: Directory containing known face images
#             detection_cooldown: Cooldown period in seconds before storing another detection for the same person
#         """
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_faces_dir = known_faces_dir
#         self.database_type = database_type.lower()
#         self.detection_cooldown = detection_cooldown
        
#         # Track last detection time for each person to avoid duplicate entries
#         self.last_detection_times = {}
#         # Current processing date for daily reset
#         self.current_processing_date = datetime.date.today()
#         # Set to track people already detected today (for daily one-time tracking)
#         self.detected_today = set()
        
#         # Initialize database connection
#         if self.database_type == "sqlite":
#             self.sqlite_path = sqlite_path
#             self.conn = sqlite3.connect(sqlite_path)
#             self.cursor = self.conn.cursor()
#             self.create_sqlite_database()
#             self.mongo_client = None
#             self.mongo_db = None
#             self.mongo_collection = None
#         elif self.database_type == "mongodb":
#             self.mongo_uri = mongo_uri
#             self.mongo_client = MongoClient(mongo_uri)
#             self.mongo_db = self.mongo_client[mongo_db]
#             self.mongo_collection = self.mongo_db[mongo_collection]
#             self.conn = None
#             self.cursor = None
#         else:
#             raise ValueError("database_type must be 'sqlite' or 'mongodb'")
         
#         # Load known faces
#         self.load_known_faces()
    
#     def create_sqlite_database(self) -> None:
#         """Create the SQLite database table if it doesn't exist."""
#         self.cursor.execute('''
#         CREATE TABLE IF NOT EXISTS face_detections (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             person_name TEXT NOT NULL,
#             timestamp TEXT NOT NULL,
#             date TEXT NOT NULL,
#             confidence REAL,
#             location TEXT,
#             camera_id TEXT
#         )
#         ''')
#         self.conn.commit()
    
#     def load_known_faces(self) -> None:
#         """Load all known faces from the directory."""
#         if not os.path.exists(self.known_faces_dir):
#             os.makedirs(self.known_faces_dir)
#             print(f"Created directory {self.known_faces_dir} for storing known faces.")
#             return
            
#         for filename in os.listdir(self.known_faces_dir):
#             if filename.endswith(".jpg") or filename.endswith(".png"):
#                 name = os.path.splitext(filename)[0]
#                 image_path = os.path.join(self.known_faces_dir, filename)
                
#                 # Load image and get face encoding
#                 image = face_recognition.load_image_file(image_path)
#                 face_encodings = face_recognition.face_encodings(image)
                
#                 if len(face_encodings) > 0:
#                     self.known_face_encodings.append(face_encodings[0])
#                     self.known_face_names.append(name)
#                     print(f"Loaded known face: {name}")
#                 else:
#                     print(f"Warning: No face found in {filename}")
    
#     def add_new_face(self, image, name: str) -> bool:
#         """
#         Add a new known face to the system.
        
#         Args:
#             image: Image containing the face
#             name: Name of the person
            
#         Returns:
#             bool: True if face was added successfully
#         """
#         face_encodings = face_recognition.face_encodings(image)
#         if len(face_encodings) == 0:
#             return False
            
#         # Save face image
#         filename = f"{name}.jpg"
#         filepath = os.path.join(self.known_faces_dir, filename)
#         cv2.imwrite(filepath, image)
        
#         # Add to known faces
#         self.known_face_encodings.append(face_encodings[0])
#         self.known_face_names.append(name)
#         return True
    
#     def should_store_detection(self, name: str, mode: str = "cooldown") -> bool:
#         """
#         Determine if a detection should be stored based on selected deduplication mode.
        
#         Args:
#             name: Name of the detected person
#             mode: Deduplication mode:
#                   "cooldown" - Store detection after cooldown period
#                   "once_per_day" - Store only one detection per person per day
#                   "always" - Store every detection
                  
#         Returns:
#             bool: True if the detection should be stored
#         """
#         current_time = datetime.datetime.now()
#         current_date = current_time.date()
        
#         # Check if date has changed, reset daily tracking if it has
#         if current_date != self.current_processing_date:
#             self.detected_today.clear()
#             self.current_processing_date = current_date
        
#         if mode == "always":
#             return True
            
#         elif mode == "once_per_day":
#             # Only store if this person hasn't been detected today
#             if name not in self.detected_today:
#                 self.detected_today.add(name)
#                 return True
#             return False
            
#         elif mode == "cooldown":
#             # Check if this is the first detection or if cooldown period has passed
#             if name not in self.last_detection_times:
#                 self.last_detection_times[name] = current_time
#                 return True
                
#             last_time = self.last_detection_times[name]
#             time_diff = (current_time - last_time).total_seconds()
            
#             if time_diff > self.detection_cooldown:
#                 self.last_detection_times[name] = current_time
#                 return True
#             return False
            
#         # Default behavior is to store
#         return True
    
#     def store_detection(self, name: str, confidence: float, location: str = None, camera_id: str = None, 
#                    deduplication_mode: str = "cooldown") -> None:
#         """
#         Store a face detection in the database with deduplication.
        
#         Args:
#             name: Name of the detected person
#             confidence: Confidence level of the detection
#             location: Optional location information
#             camera_id: Optional camera identifier
#             deduplication_mode: Mode for deduplication ("cooldown", "once_per_day", or "always")
#         """
#         # Check if we should store this detection based on selected mode
#         if not self.should_store_detection(name, deduplication_mode):
#             return
            
#         timestamp = datetime.datetime.now()
#         timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#         date_str = timestamp.strftime("%Y-%m-%d")
        
#         if self.database_type == "sqlite":
#             self.cursor.execute(
#                 "INSERT INTO face_detections (person_name, timestamp, date, confidence, location, camera_id) "
#                 "VALUES (?, ?, ?, ?, ?, ?)",
#                 (name, timestamp_str, date_str, confidence, location, camera_id)
#             )
#             self.conn.commit()
#         elif self.database_type == "mongodb":
#             detection_record = {
#                 "person_name": name,
#                 "timestamp": timestamp,
#                 # Use datetime.datetime instead of datetime.date for MongoDB compatibility
#                 "date": timestamp,  # Store full timestamp and use date part when querying
#                 "date_str": date_str,  # Store string version for easy querying if needed
#                 "confidence": float(confidence),  # Convert numpy float to Python float
#                 "location": location,
#                 "camera_id": camera_id
#             }
#             self.mongo_collection.insert_one(detection_record)
        
#         print(f"Stored detection: {name} at {timestamp_str}, confidence: {confidence:.2f}")



#     def get_recent_detections(self, limit: int = 10) -> List[Dict]:
#         """
#         Get recent face detections from the database.
        
#         Args:
#             limit: Maximum number of records to retrieve
            
#         Returns:
#             List of detection records
#         """
#         if self.database_type == "sqlite":
#             self.cursor.execute(
#                 "SELECT person_name, timestamp, confidence, location, camera_id FROM face_detections "
#                 "ORDER BY timestamp DESC LIMIT ?",
#                 (limit,)
#             )
#             rows = self.cursor.fetchall()
#             return [
#                 {"person_name": row[0], "timestamp": row[1], "confidence": row[2], "location": row[3], "camera_id": row[4]}
#                 for row in rows
#             ]
#         elif self.database_type == "mongodb":
#             cursor = self.mongo_collection.find().sort("timestamp", -1).limit(limit)
#             return list(cursor)
    
#     def get_daily_detections(self, date_str: str = None) -> List[Dict]:
#         """
#         Get all detections for a specific date.
        
#         Args:
#             date_str: Date string in format 'YYYY-MM-DD', or None for today
            
#         Returns:
#             List of detection records
#         """
#         if date_str is None:
#             date_str = datetime.date.today().strftime("%Y-%m-%d")
            
#         if self.database_type == "sqlite":
#             self.cursor.execute(
#                 "SELECT person_name, timestamp, confidence, location, camera_id FROM face_detections "
#                 "WHERE date = ? ORDER BY timestamp",
#                 (date_str,)
#             )
#             rows = self.cursor.fetchall()
#             return [
#                 {"person_name": row[0], "timestamp": row[1], "confidence": row[2], "location": row[3], "camera_id": row[4]}
#                 for row in rows
#             ]
#         elif self.database_type == "mongodb":
#             # Parse the date string to get year, month, day
#             year, month, day = map(int, date_str.split('-'))
            
#             # Create datetime objects for the start and end of the day
#             start_date = datetime.datetime(year, month, day, 0, 0, 0)
#             end_date = datetime.datetime(year, month, day, 23, 59, 59, 999999)
            
#             # Query for all timestamps within that day
#             cursor = self.mongo_collection.find({
#                 "timestamp": {"$gte": start_date, "$lte": end_date}
#             }).sort("timestamp", 1)
            
#             return list(cursor)
    
#     def get_person_history(self, name: str, limit: int = 10) -> List[Dict]:
#         """
#         Get detection history for a specific person.
        
#         Args:
#             name: Name of the person
#             limit: Maximum number of records to retrieve
            
#         Returns:
#             List of detection records
#         """
#         if self.database_type == "sqlite":
#             self.cursor.execute(
#                 "SELECT timestamp, confidence, location, camera_id FROM face_detections "
#                 "WHERE person_name = ? ORDER BY timestamp DESC LIMIT ?",
#                 (name, limit)
#             )
#             rows = self.cursor.fetchall()
#             return [
#                 {"person_name": name, "timestamp": row[0], "confidence": row[1], "location": row[2], "camera_id": row[3]}
#                 for row in rows
#             ]
#         elif self.database_type == "mongodb":
#             cursor = self.mongo_collection.find({"person_name": name}).sort("timestamp", -1).limit(limit)
#             return list(cursor)
    
#     def process_frame(self, frame, location: str = None, camera_id: str = None, 
#                      deduplication_mode: str = "cooldown") -> Tuple[np.ndarray, List[str]]:
#         """
#         Process a single frame from the video stream.
        
#         Args:
#             frame: Video frame to process
#             location: Optional location information to store with detections
#             camera_id: Optional camera identifier
#             deduplication_mode: Mode for deduplication
            
#         Returns:
#             Tuple containing the annotated frame and list of detected names
#         """
#         # Resize frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
#         # Convert from BGR (OpenCV) to RGB (face_recognition)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
#         # Find faces in the frame
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
#         detected_names = []
        
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # Scale back up face locations
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4
            
#             # Check if the face matches any known faces
#             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#             name = "Unknown"
#             confidence = 0.0
            
#             # Use the known face with the smallest distance to the new face
#             if len(self.known_face_encodings) > 0:
#                 face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 confidence = 1 - min(face_distances)
                
#                 if matches[best_match_index] and confidence > 0.6:
#                     name = self.known_face_names[best_match_index]
            
#             # Store in database if known person (with deduplication)
#             if name != "Unknown":
#                 self.store_detection(name, confidence, location, camera_id, deduplication_mode)
#                 detected_names.append(name)
            
#             # Draw box and label on the frame
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(frame, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), 
#                         cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
#         return frame, detected_names
    
#     def process_video_stream(self, video_source=0, location: str = None, camera_id: str = None,
#                            detection_cooldown: int = None, deduplication_mode: str = "cooldown") -> None:
#         """
#         Process a video stream from camera or file.
        
#         Args:
#             video_source: Camera index or video file path
#             location: Optional location information to store with detections
#             camera_id: Optional camera identifier
#             detection_cooldown: Override the default cooldown period in seconds
#             deduplication_mode: Mode for deduplication ("cooldown", "once_per_day", or "always")
#         """
#         # Override cooldown if provided
#         if detection_cooldown is not None:
#             self.detection_cooldown = detection_cooldown
        
#         # Open video capture
#         video_capture = cv2.VideoCapture(video_source)
        
#         if not video_capture.isOpened():
#             print(f"Error: Could not open video source {video_source}")
#             return
        
#         print(f"Processing video stream. Press 'q' to quit, 'a' to add a new face, 'r' to see recent detections.")
#         print(f"Deduplication mode: {deduplication_mode}")
#         if deduplication_mode == "cooldown":
#             print(f"Detection cooldown set to {self.detection_cooldown} seconds.")
        
#         try:
#             while True:
#                 # Capture frame-by-frame
#                 ret, frame = video_capture.read()
                
#                 if not ret:
#                     print("Error: Failed to capture frame")
#                     break
                    
#                 # Process frame
#                 processed_frame, detected_names = self.process_frame(
#                     frame, location, camera_id, deduplication_mode
#                 )
                
#                 # Display the resulting frame
#                 cv2.imshow('Video', processed_frame)
                
#                 # Handle key presses
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     break
#                 elif key == ord('a'):
#                     # Capture a frame for adding a new face
#                     ret, new_face_frame = video_capture.read()
#                     if ret:
#                         name = input("Enter name for the new face: ")
#                         if self.add_new_face(new_face_frame, name):
#                             print(f"Added new face: {name}")
#                         else:
#                             print("No face detected in the frame")
#                 elif key == ord('r'):
#                     # Show recent detections
#                     recent = self.get_recent_detections()
#                     print("\nRecent detections:")
#                     for i, detection in enumerate(recent):
#                         print(f"{i+1}. {detection['person_name']} - {detection['timestamp']} - Confidence: {detection['confidence']:.2f}")
#                     print()
#                 elif key == ord('d'):
#                     # Show today's detections
#                     today = datetime.date.today().strftime("%Y-%m-%d")
#                     detections = self.get_daily_detections(today)
#                     print(f"\nDetections for today ({today}):")
#                     for i, detection in enumerate(detections):
#                         print(f"{i+1}. {detection['person_name']} - {detection['timestamp']} - Confidence: {detection['confidence']:.2f}")
#                     print()
#                 elif key == ord('m'):
#                     # Cycle through deduplication modes
#                     modes = ["cooldown", "once_per_day", "always"]
#                     current_index = modes.index(deduplication_mode)
#                     deduplication_mode = modes[(current_index + 1) % len(modes)]
#                     print(f"\nSwitched to deduplication mode: {deduplication_mode}")
#                     if deduplication_mode == "cooldown":
#                         print(f"Detection cooldown set to {self.detection_cooldown} seconds.")
#                     print()
#         finally:
#             # Release resources
#             video_capture.release()
#             cv2.destroyAllWindows()
#             if self.database_type == "sqlite" and self.conn:
#                 self.conn.close()
#             if self.database_type == "mongodb" and self.mongo_client:
#                 self.mongo_client.close()

# def main():
#     """Main function to run the facial recognition system."""
#     # Uncomment the appropriate line for your database preference
    
#     # For SQLite (default)
#     # system = FacialRecognitionSystem(
#     #     database_type="sqlite",
#     #     detection_cooldown=60  # Only store one detection per person every 60 seconds
#     # )
    
#     # For MongoDB
#     system = FacialRecognitionSystem(
#         database_type="mongodb",
#         mongo_uri=mongoURI,
#         mongo_db="face_recognition",
#         mongo_collection="detections",
#         detection_cooldown=60  # Only store one detection per person every 60 seconds
#     )
    
#     # Choose your deduplication mode:
#     # - "cooldown": Store detections with a cooldown period (default: 60 seconds)
#     # - "once_per_day": Store only one detection per person per day
#     # - "always": Store every detection (no deduplication)
#     deduplication_mode = "cooldown"
    
#     # Optional: specify a location and camera ID
#     system.process_video_stream(
#         location="Office Entrance", 
#         camera_id="Cam-01",
#         deduplication_mode=deduplication_mode
#     )

# if __name__ == "__main__":
#     main()