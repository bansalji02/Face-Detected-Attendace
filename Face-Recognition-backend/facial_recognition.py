import cv2
import face_recognition
import numpy as np
import datetime
import os
from typing import List, Dict, Tuple, Set
import logging
from database import Database
import dlib
import time
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import tensorflow as tf


logger = logging.getLogger(__name__)

class FacialRecognitionSystem:
    def __init__(
                self, 
                database: Database,
                known_faces_dir: str = "known_faces",
                detection_cooldown: int = 60,
                confidence_threshold: float = 0.5,
                multi_angle: bool = True,
                use_tracking: bool = True
                ):
        """
        Initialize the facial recognition system.
        
        Args:
            database: Database instance for storing detections
            known_faces_dir: Directory containing known face images
            detection_cooldown: Cooldown period in seconds between detections of same person
            confidence_threshold: Minimum confidence level for face recognition
            multi_angle: Whether to use multi-angle face detection
            use_tracking: Whether to use object tracking for continuous detection
        """
        self.database = database
        self.known_faces_dir = known_faces_dir
        
        self.detection_cooldown = detection_cooldown
        self.confidence_threshold = confidence_threshold
        self.multi_angle = multi_angle
        self.use_tracking = use_tracking
        
        # Track last detection times for each person to avoid duplicate entries
        self.last_detection_times = {}
        # Current processing date for daily reset
        self.current_processing_date = datetime.date.today()
        # Set to track people already detected today (for daily one-time tracking)
        self.detected_today = set()
        
        # Initialize face detector with additional models
        self.face_detector = dlib.get_frontal_face_detector()
        # For profile detection
        self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Initialize face trackers for continuous tracking
        self.trackers = []
        self.tracker_names = []
        self.tracking_faces = {}
        self.last_detection_time = time.time()
        self.tracker_timeout = 3.0  # Reset trackers after 3 seconds
        
        # Initialize multi-angle face database
        self.face_encodings_db = {}  # Person name -> list of encodings from different angles
        
        # Initialize KNN classifier for improved matching
        self.knn_classifier = None
        self.use_knn = False
        
        # Load known faces
        self.load_known_faces()
        
        # Build KNN classifier if we have enough data
        if len(self.face_encodings_db) > 0:
            self.build_knn_classifier()
    
    def build_knn_classifier(self):
        """Build KNN classifier from stored face encodings."""
        X = []
        y = []
        
        for name, encodings in self.face_encodings_db.items():
            for encoding in encodings:
                X.append(encoding)
                y.append(name)
        
        if len(X) > 0:
            self.knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
            self.knn_classifier.fit(X, y)
            self.use_knn = True
            logger.info(f"Built KNN classifier with {len(X)} face encodings from {len(self.face_encodings_db)} people")
    
    def load_known_faces(self) -> None:
        """Load all known faces from the directory, including various angles if available."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            logger.info(f"Created directory {self.known_faces_dir} for storing known faces.")
            return
            
        # Create person-specific subdirectories if they don't exist
        for filename in os.listdir(self.known_faces_dir):
            # If it's a direct image file (old format)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.known_faces_dir, filename)
                
                # Create a directory for this person
                person_dir = os.path.join(self.known_faces_dir, name)
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                    
                    # Move the image to the new directory
                    new_path = os.path.join(person_dir, "frontal.jpg")
                    os.rename(image_path, new_path)
                    logger.info(f"Migrated {name}'s face image to dedicated directory")
            
        # Now process each person's directory
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_dir):
                continue
                
            # Process all images in the person's directory
            person_encodings = []
            for img_file in os.listdir(person_dir):
                if img_file.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(person_dir, img_file)
                    
                    # Load image and get face encoding
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        person_encodings.append(face_encodings[0])
                        logger.info(f"Loaded face variant for {person_name}: {img_file}")
                    else:
                        logger.warning(f"No face found in {image_path}")
            
            # Store all encodings for this person
            if person_encodings:
                self.face_encodings_db[person_name] = person_encodings
                logger.info(f"Loaded {len(person_encodings)} face variants for {person_name}")
    
    def add_new_face(self, image, name: str, angle_label: str = "frontal") -> bool:
        """
        Add a new known face to the system.
        
        Args:
            image: Image containing the face
            name: Name of the person
            angle_label: Label for the face angle (e.g., "frontal", "profile_left", "profile_right")
            
        Returns:
            bool: True if face was added successfully
        """
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) == 0:
            return False
        
        # Create person directory if it doesn't exist
        person_dir = os.path.join(self.known_faces_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        # Save face image
        filename = f"{angle_label}.jpg"
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, image)
        
        # Add to face encodings database
        if name in self.face_encodings_db:
            self.face_encodings_db[name].append(face_encodings[0])
        else:
            self.face_encodings_db[name] = [face_encodings[0]]
            
        # Rebuild KNN classifier with the new data
        self.build_knn_classifier()
        
        return True
    
    def add_multi_angle_faces(self, frames, name: str) -> int:
        """
        Add multiple angle face images for a person.
        
        Args:
            frames: List of frames containing the face from different angles
            name: Name of the person
            
        Returns:
            int: Number of successfully added face angles
        """
        successful_adds = 0
        angles = ["frontal", "slight_left", "profile_left", "slight_right", "profile_right", "tilted_up", "tilted_down"]
        
        for i, frame in enumerate(frames):
            if i < len(angles):
                angle_label = angles[i]
            else:
                angle_label = f"extra_{i}"
                
            if self.add_new_face(frame, name, angle_label):
                successful_adds += 1
                
        return successful_adds
    
    def detect_faces_multi_method(self, frame) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Detect faces using multiple detection methods to improve side profile detection.
        
        Args:
            frame: Input frame
            
        Returns:
            List of tuples containing face locations (top, right, bottom, left) and detector type
        """
        # Convert to grayscale for Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # List to store all detected face locations
        all_face_locations = []
        
        # Method 1: HOG-based detection (face_recognition's default)
        face_locations = face_recognition.face_locations(frame)
        for face_location in face_locations:
            all_face_locations.append((face_location, "hog"))
        
        # Method 2: Frontal face detection with dlib
        if self.multi_angle:
            dlib_faces = self.face_detector(gray, 1)
            for face in dlib_faces:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                # face_recognition format is (top, right, bottom, left)
                all_face_locations.append(((top, right, bottom, left), "dlib"))
            
            # Method 3: Profile face detection with Haar cascades
            # Detect right-facing profiles
            profile_faces = self.profile_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in profile_faces:
                left = x
                top = y
                right = x + w
                bottom = y + h
                all_face_locations.append(((top, right, bottom, left), "profile_right"))
            
            # Flip the image to detect left-facing profiles
            flipped = cv2.flip(gray, 1)
            flipped_profile_faces = self.profile_detector.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in flipped_profile_faces:
                # Adjust coordinates for the flipped image
                frame_width = frame.shape[1]
                left = frame_width - (x + w)
                top = y
                right = frame_width - x
                bottom = y + h
                all_face_locations.append(((top, right, bottom, left), "profile_left"))
        
        # Remove duplicates (overlapping detections)
        return self.remove_overlapping_faces(all_face_locations)
    
    def remove_overlapping_faces(self, face_locations_with_method):
        """Remove overlapping face detections by keeping the one with larger area."""
        filtered_faces = []
        
        # Sort faces by area (largest first)
        sorted_faces = sorted(
            face_locations_with_method,
            key=lambda x: (x[0][2] - x[0][0]) * (x[0][1] - x[0][3]),
            reverse=True
        )
        
        for face_location, method in sorted_faces:
            top, right, bottom, left = face_location
            
            # Check if this face overlaps significantly with any face we're keeping
            overlaps = False
            for kept_face, _ in filtered_faces:
                k_top, k_right, k_bottom, k_left = kept_face
                
                # Calculate overlap area
                x_overlap = max(0, min(right, k_right) - max(left, k_left))
                y_overlap = max(0, min(bottom, k_bottom) - max(top, k_top))
                overlap_area = x_overlap * y_overlap
                
                # Calculate areas
                face_area = (bottom - top) * (right - left)
                kept_face_area = (k_bottom - k_top) * (k_right - k_left)
                
                # If overlap is more than 50% of the smaller face, consider it a duplicate
                if overlap_area > 0.5 * min(face_area, kept_face_area):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_faces.append((face_location, method))
                
        return filtered_faces
    
    def should_store_detection(self, name: str, mode: str = "cooldown") -> bool:
        """
        Determine if a detection should be stored based on selected deduplication mode.
        
        Args:
            name: Name of the detected person
            mode: Deduplication mode:
                  "cooldown" - Store detection after cooldown period
                  "once_per_day" - Store only one detection per person per day
                  "always" - Store every detection
                  
        Returns:
            bool: True if the detection should be stored
        """
        current_time = datetime.datetime.now()
        current_date = current_time.date()
        
        # Check if date has changed, reset daily tracking if it has
        if current_date != self.current_processing_date:
            self.detected_today.clear()
            self.current_processing_date = current_date
        
        if mode == "always":
            return True
            
        elif mode == "once_per_day":
            # Only store if this person hasn't been detected today
            if name not in self.detected_today:
                self.detected_today.add(name)
                return True
            return False
            
        elif mode == "cooldown":
            # Check if this is the first detection or if cooldown period has passed
            if name not in self.last_detection_times:
                self.last_detection_times[name] = current_time
                return True
                
            last_time = self.last_detection_times[name]
            time_diff = (current_time - last_time).total_seconds()
            
            if time_diff > self.detection_cooldown:
                self.last_detection_times[name] = current_time
                return True
            return False
            
        # Default behavior is to store
        return True
    
    def update_trackers(self, frame):
        """Update all active face trackers and remove failed ones."""
        current_time = time.time()
        
        # Check if we should reset trackers
        if current_time - self.last_detection_time > self.tracker_timeout:
            self.trackers = []
            self.tracker_names = []
            return []
        
        tracked_faces = []
        successful_trackers = []
        successful_names = []
        
        # Update each tracker
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            
            if success:
                left = int(bbox[0])
                top = int(bbox[1])
                right = int(bbox[0] + bbox[2])
                bottom = int(bbox[1] + bbox[3])
                
                tracked_faces.append(((top, right, bottom, left), self.tracker_names[i]))
                successful_trackers.append(tracker)
                successful_names.append(self.tracker_names[i])
        
        # Update the list of active trackers
        self.trackers = successful_trackers
        self.tracker_names = successful_names
        
        return tracked_faces
    
    def identify_face(self, face_encoding) -> Tuple[str, float]:
        """
        Identify a face using either KNN or traditional distance comparison.
        
        Args:
            face_encoding: Face encoding to identify
            
        Returns:
            Tuple of (name, confidence)
        """
        if self.use_knn and self.knn_classifier is not None:
            # Use KNN classifier for better matching
            proba = self.knn_classifier.predict_proba([face_encoding])[0]
            best_idx = np.argmax(proba)
            confidence = proba[best_idx]
            
            if confidence > self.confidence_threshold:
                name = self.knn_classifier.classes_[best_idx]
                return name, confidence
            else:
                return "Unknown", confidence
        else:
            # Traditional face_recognition approach
            best_match = "Unknown"
            best_confidence = 0.0
            
            for name, encodings in self.face_encodings_db.items():
                # Calculate distances to all encodings for this person
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                
                # Get the best match
                if len(face_distances) > 0:
                    min_distance = min(face_distances)
                    confidence = 1 - min_distance
                    
                    if confidence > self.confidence_threshold and confidence > best_confidence:
                        best_match = name
                        best_confidence = confidence
            
            return best_match, best_confidence
    
    def process_frame(self, frame, location: str = None, camera_id: str = None, 
                     deduplication_mode: str = "cooldown") -> Tuple[np.ndarray, List[str]]:
        """
        Process a single frame from the video stream.
        
        Args:
            frame: Video frame to process
            location: Optional location information to store with detections
            camera_id: Optional camera identifier
            deduplication_mode: Mode for deduplication
            
        Returns:
            Tuple containing the annotated frame and list of detected names
        """
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Track faces if enabled and if we have trackers
        tracked_faces = []
        if self.use_tracking and self.trackers:
            tracked_faces = self.update_trackers(frame)
        
        # Every few frames, or if we have no tracked faces, run full detection
        current_time = time.time()
        if not tracked_faces or current_time - self.last_detection_time > 0.5:  # Run detection every 0.5 seconds
            self.last_detection_time = current_time
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert from BGR (OpenCV) to RGB (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Multi-method face detection
            face_locations_with_method = self.detect_faces_multi_method(rgb_small_frame)
            
            # Extract just the locations for encoding
            face_locations = [loc for loc, _ in face_locations_with_method]
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Reset trackers
            self.trackers = []
            self.tracker_names = []
            
            # Process each detected face
            for (face_loc, method), face_encoding in zip(face_locations_with_method, face_encodings):
                # Scale back up face locations
                top, right, bottom, left = face_loc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Identify the face
                name, confidence = self.identify_face(face_encoding)
                
                # If tracking is enabled, create a new tracker for this face
                if self.use_tracking and name != "Unknown":
                    tracker = cv2.TrackerKCF.create()  # Use this instead of cv2.TrackerKCF_create()
                    bbox = (left, top, right - left, bottom - top)
                    success = tracker.init(frame, bbox)
                    
                    if success:
                        self.trackers.append(tracker)
                        self.tracker_names.append(name)
                
                # Store in database if known person (with deduplication)
                if name != "Unknown":
                    if self.should_store_detection(name, deduplication_mode):
                        self.database.store_detection(name, confidence, location=location, camera_id=camera_id)
                    
                    # Draw detection information
                    method_colors = {
                        "hog": (0, 255, 0),      # Green for HOG detector
                        "dlib": (255, 0, 0),     # Blue for dlib
                        "profile_left": (0, 0, 255),  # Red for left profile
                        "profile_right": (0, 255, 255),  # Yellow for right profile
                    }
                    color = method_colors.get(method, (0, 255, 0))
                    
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            detected_names = [self.tracker_names[i] for i in range(len(self.trackers))]
        else:
            # Use tracked faces
            detected_names = []
            for (top, right, bottom, left), name in tracked_faces:
                # Draw rectangle for tracked face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (255, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, f"{name} (tracked)", (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                detected_names.append(name)
        
        return display_frame, detected_names
    
    def capture_multi_angle_faces(self, video_source=0):
        """
        Interactive utility to capture multiple face angles for a person.
        
        Args:
            video_source: Camera index or video file path
        """
        name = input("Enter person's name: ")
        print(f"Please position the person to capture multiple face angles.")
        print("Press SPACE to capture the current angle.")
        print("Angles to capture: frontal, slight left, profile left, slight right, profile right, up, down")
        print("Press ESC when done capturing all angles.")
        
        # Open video capture
        video_capture = cv2.VideoCapture(video_source)
        if not video_capture.isOpened():
            logger.error(f"Could not open video source {video_source}")
            return
        
        captured_frames = []
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Show angle count
                cv2.putText(frame, f"Captured angles: {len(captured_frames)}/7", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Capture Multiple Angles', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    captured_frames.append(frame.copy())
                    print(f"Captured angle {len(captured_frames)}")
                    
                    # If we have all angles, break
                    if len(captured_frames) >= 7:
                        print("All angles captured!")
                        break
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
        
        # Add the captured frames to the database
        if captured_frames:
            count = self.add_multi_angle_faces(captured_frames, name)
            print(f"Successfully added {count} face angles for {name}")
    
    def process_video_stream(self, video_source=0, location: str = None, camera_id: str = None,
                           detection_cooldown: int = None, deduplication_mode: str = "cooldown") -> None:
        """
        Process a video stream from camera or file.
        
        Args:
            video_source: Camera index or video file path
            location: Optional location information to store with detections
            camera_id: Optional camera identifier (also used as MongoDB collection name)
            detection_cooldown: Override the default cooldown period in seconds
            deduplication_mode: Mode for deduplication ("cooldown", "once_per_day", or "always")
        """
        # Ensure camera_id is set (needed for collection name)
        if not camera_id:
            camera_id = f"Cam-{video_source}" if isinstance(video_source, int) else "default"
            logger.info(f"No camera_id provided, using: {camera_id}")
        
        # Override cooldown if provided
        if detection_cooldown is not None:
            self.detection_cooldown = detection_cooldown
        
    
        # Open video capture
        video_capture = cv2.VideoCapture(video_source)
        
        if not video_capture.isOpened():
            logger.error(f"Could not open video source {video_source}")
            return
        
        logger.info(f"Processing video stream from camera: {camera_id}")
        logger.info(f"Press 'q' to quit, 'a' to add a new face, 'r' to see recent detections.")
        print(f"Processing video stream from camera: {camera_id}")
        print(f"Press 'q' to quit, 'a' to add a new face, 'm' to capture multi-angle faces")
        print(f"Press 'r' to see recent detections, 't' to toggle tracking, 'p' to toggle profile detection")
        print(f"Deduplication mode: {deduplication_mode}")
        if deduplication_mode == "cooldown":
            print(f"Detection cooldown set to {self.detection_cooldown} seconds.")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                    
                # Process frame
                processed_frame, detected_names = self.process_frame(
                    frame, location, camera_id, deduplication_mode
                )
                
                # Show tracking status
                tracking_status = "ON" if self.use_tracking else "OFF"
                profile_status = "ON" if self.multi_angle else "OFF"
                cv2.putText(processed_frame, f"Tracking: {tracking_status} | Profile: {profile_status}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow('Video', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Capture a frame for adding a new face
                    ret, new_face_frame = video_capture.read()
                    if ret:
                        name = input("Enter name for the new face: ")
                        if self.add_new_face(new_face_frame, name):
                            print(f"Added new face: {name}")
                        else:
                            print("No face detected in the frame")
                elif key == ord('m'):
                    # Capture multi-angle faces
                    video_capture.release()
                    cv2.destroyAllWindows()
                    self.capture_multi_angle_faces(video_source)
                    # Reopen the capture
                    video_capture = cv2.VideoCapture(video_source)
                elif key == ord('r'):
                    # Show recent detections for current camera
                    recent = self.database.get_recent_detections(camera_id)
                    print(f"\nRecent detections for camera {camera_id}:")
                    for i, detection in enumerate(recent):
                        print(f"{i+1}. {detection['person_name']} - {detection['timestamp']} - Confidence: {detection['confidence']:.2f}")
                    print()
                elif key == ord('t'):
                    # Toggle tracking
                    self.use_tracking = not self.use_tracking
                    print(f"\nTracking {'enabled' if self.use_tracking else 'disabled'}")
                    # Reset trackers when disabling tracking
                    if not self.use_tracking:
                        self.trackers = []
                        self.tracker_names = []
                elif key == ord('p'):
                    # Toggle profile detection
                    self.multi_angle = not self.multi_angle
                    print(f"\nProfile detection {'enabled' if self.multi_angle else 'disabled'}")
                elif key == ord('d'):
                    # Show today's detections for current camera
                    today = datetime.date.today().strftime("%Y-%m-%d")
                    detections = self.database.get_daily_detections(today, camera_id)
                    print(f"\nDetections for today ({today}) on camera {camera_id}:")
                    for i, detection in enumerate(detections):
                        print(f"{i+1}. {detection['person_name']} - {detection['timestamp']} - Confidence: {detection['confidence']:.2f}")
                    print()
                elif key == ord('c'):
                    # Lower/raise confidence threshold
                    self.confidence_threshold = max(0.3, min(0.9, self.confidence_threshold - 0.05))
                    print(f"\nConfidence threshold adjusted to: {self.confidence_threshold:.2f}")
        finally:
            # Release resources
            video_capture.release()
            cv2.destroyAllWindows()
    
    def get_office_presence_data(self, person_name=None, limit=10) -> List[Dict]:
        """
        Get presence data for all people or a specific person.
        
        Args:
            person_name: Optional name to filter results
            limit: Maximum number of records per person
            
        Returns:
            List of presence records
        """
        results = []
        
        # If a specific person is requested
        if person_name:
            # Get the person's detection history from all cameras
            detections = self.database.get_person_history(person_name, limit=limit)
            return [{
                "name": person_name,
                "last_seen": detection.get("timestamp"),
                "location": detection.get("location"),
                "camera": detection.get("camera_id"),
                "confidence": detection.get("confidence")
            } for detection in detections]
        
        # Get data for all known people
        for name in self.face_encodings_db.keys():
            # Get the most recent detection for each person
            recent = self.database.get_person_history(name, limit=1)
            if recent:
                detection = recent[0]
                results.append({
                   "name": name,
                    "last_seen": detection.get("timestamp"),
                    "location": detection.get("location"),
                    "camera": detection.get("camera_id"),
                    "confidence": detection.get("confidence")
                })
            else:
                # No detections for this person
                results.append({
                    "name": name,
                    "last_seen": None,
                    "location": None,
                    "camera": None,
                    "confidence": None
                })
                
        return results
    
    def generate_presence_report(self, date=None, format="text") -> str:
        """
        Generate a report of office presence for a specific date.
        
        Args:
            date: Date to report for (defaults to today)
            format: Output format ("text", "html", or "json")
            
        Returns:
            Formatted report as string
        """
        if date is None:
            date = datetime.date.today().strftime("%Y-%m-%d")
            
        # Get all detections for the date
        detections = self.database.get_daily_detections(date)
        
        # Group by person
        people_data = {}
        for detection in detections:
            name = detection.get("person_name")
            if name not in people_data:
                people_data[name] = []
            people_data[name].append(detection)
        
        # Generate report in requested format
        if format == "json":
            import json
            return json.dumps(people_data, default=str, indent=2)
            
        elif format == "html":
            html = f"<h1>Presence Report for {date}</h1>\n"
            html += "<table border='1'>\n"
            html += "<tr><th>Name</th><th>First Seen</th><th>Last Seen</th><th>Total Detections</th><th>Locations</th></tr>\n"
            
            for name, detections in people_data.items():
                # Sort by timestamp
                detections.sort(key=lambda x: x.get("timestamp"))
                first_seen = detections[0].get("timestamp")
                last_seen = detections[-1].get("timestamp")
                count = len(detections)
                
                # Get unique locations
                locations = set()
                for detection in detections:
                    loc = detection.get("location")
                    if loc:
                        locations.add(loc)
                        
                locations_str = ", ".join(locations) if locations else "Unknown"
                
                html += f"<tr><td>{name}</td><td>{first_seen}</td><td>{last_seen}</td><td>{count}</td><td>{locations_str}</td></tr>\n"
                
            html += "</table>"
            return html
            
        else:  # text format
            report = f"Presence Report for {date}\n"
            report += "=" * 50 + "\n\n"
            
            for name, detections in people_data.items():
                # Sort by timestamp
                detections.sort(key=lambda x: x.get("timestamp"))
                first_seen = detections[0].get("timestamp")
                last_seen = detections[-1].get("timestamp")
                count = len(detections)
                
                # Get unique locations
                locations = set()
                for detection in detections:
                    loc = detection.get("location")
                    if loc:
                        locations.add(loc)
                        
                locations_str = ", ".join(locations) if locations else "Unknown"
                
                report += f"Name: {name}\n"
                report += f"  First seen: {first_seen}\n"
                report += f"  Last seen: {last_seen}\n"
                report += f"  Total detections: {count}\n"
                report += f"  Locations: {locations_str}\n\n"
                
            return report
    
    def get_attendance_record(self, start_date=None, end_date=None) -> Dict[str, List[str]]:
        """
        Generate attendance record for a date range.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to lists of people detected
        """
        if start_date is None:
            # Default to last 7 days
            end_dt = datetime.datetime.now()
            start_dt = end_dt - datetime.timedelta(days=7)
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
        elif end_date is None:
            # If only start_date provided, use just that date
            end_date = start_date
            
        # Get all detections in the date range
        detections = self.database.get_detections_in_date_range(start_date, end_date)
        
        # Group by date
        attendance = {}
        for detection in detections:
            # Extract date from timestamp
            timestamp = detection.get("timestamp")
            if isinstance(timestamp, str):
                date_str = timestamp.split()[0]  # Extract YYYY-MM-DD part
            else:
                date_str = timestamp.strftime("%Y-%m-%d")
                
            # Initialize date entry if needed
            if date_str not in attendance:
                attendance[date_str] = set()
                
            # Add person to the attendance for this date
            attendance[date_str].add(detection.get("person_name"))
            
        # Convert sets to sorted lists for better readability
        return {date: sorted(list(people)) for date, people in attendance.items()}
    
    def export_data(self, format="csv", filename=None) -> str:
        """
        Export detection data to file.
        
        Args:
            format: Export format ("csv", "json", or "excel")
            filename: Output filename (optional)
            
        Returns:
            Path to the exported file
        """
        # Get all detection data
        all_data = self.database.get_all_detections()
        
        # Generate default filename if none provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_detection_export_{timestamp}.{format}"
            
        if format == "json":
            import json
            with open(filename, 'w') as f:
                json.dump(all_data, f, default=str, indent=2)
                
        elif format == "excel":
            try:
                df = pd.DataFrame(all_data)
                df.to_excel(filename, index=False)
            except ImportError:
                logger.error("pandas is required for Excel export. Please install with: pip install pandas openpyxl")
                return None
                
        else:  # csv format
            import csv
            with open(filename, 'w', newline='') as f:
                # Determine all possible fields from the data
                fields = set()
                for record in all_data:
                    fields.update(record.keys())
                
                writer = csv.DictWriter(f, fieldnames=sorted(fields))
                writer.writeheader()
                writer.writerows(all_data)
                
        logger.info(f"Exported {len(all_data)} records to {filename}")
        return filename
    
    def optimize_performance(self):
        """Optimize face recognition performance based on system capabilities."""
        # Check available CPU cores
        try:
            import multiprocessing
            cores = multiprocessing.cpu_count()
            logger.info(f"Detected {cores} CPU cores")
            
            # If we have many cores, enable parallel processing
            if cores >= 4:
                face_recognition.api._raw_face_locations = face_recognition.api.cnn_face_locations
                logger.info("Switched to CNN face detection for better accuracy")
            else:
                logger.info("Using HOG face detection for better performance")
                
        except ImportError:
            logger.warning("multiprocessing module not available, using default settings")
        
        # Check if we should reduce processing frequency
        if len(self.face_encodings_db) > 50:
            logger.info("Large face database detected, adjusting processing settings")
            self.tracker_timeout = 5.0  # Increase tracker timeout
            
        # Check GPU availability
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPU detected: {gpus[0].name}")
                # Enable GPU acceleration if available
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logger.info("GPU acceleration enabled")
        except ImportError:
            logger.info("TensorFlow not installed, GPU acceleration not available")
            
        logger.info("Performance optimization complete")

    def run_scheduled_report(self, schedule="daily", report_type="presence", output_dir="./reports/"):
        """
        Schedule automatic report generation.
        
        Args:
            schedule: Frequency ("daily", "weekly", "monthly")
            report_type: Type of report ("presence", "attendance")
            output_dir: Directory to save reports
        """
        import threading
        import time
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        def generate_report():
            while True:
                today = datetime.date.today()
                
                # Determine if we should generate a report today
                should_run = False
                if schedule == "daily":
                    should_run = True
                elif schedule == "weekly" and today.weekday() == 0:  # Monday
                    should_run = True
                elif schedule == "monthly" and today.day == 1:  # First day of month
                    should_run = True
                    
                if should_run:
                    logger.info(f"Generating scheduled {report_type} report")
                    
                    if report_type == "presence":
                        # Generate presence report for yesterday
                        yesterday = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                        report = self.generate_presence_report(date=yesterday, format="html")
                        
                        # Save report
                        filename = os.path.join(output_dir, f"presence_report_{yesterday}.html")
                        with open(filename, 'w') as f:
                            f.write(report)
                            
                    elif report_type == "attendance":
                        # For weekly/monthly reports, get data for appropriate date range
                        if schedule == "daily":
                            yesterday = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                            attendance = self.get_attendance_record(yesterday, yesterday)
                        elif schedule == "weekly":
                            # Get last week's data
                            end_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                            start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                            attendance = self.get_attendance_record(start_date, end_date)
                        else:  # monthly
                            # Get last month's data
                            last_month = today.replace(day=1) - datetime.timedelta(days=1)
                            start_date = last_month.replace(day=1).strftime("%Y-%m-%d")
                            end_date = last_month.strftime("%Y-%m-%d")
                            attendance = self.get_attendance_record(start_date, end_date)
                            
                        # Generate HTML report
                        html = f"<h1>Attendance Report: {start_date} to {end_date}</h1>\n"
                        html += "<table border='1'>\n"
                        html += "<tr><th>Date</th><th>People Present</th></tr>\n"
                        
                        for date in sorted(attendance.keys()):
                            people = ", ".join(attendance[date])
                            html += f"<tr><td>{date}</td><td>{people}</td></tr>\n"
                            
                        html += "</table>"
                        
                        # Save report
                        period = f"{start_date}_to_{end_date}" if start_date != end_date else start_date
                        filename = os.path.join(output_dir, f"attendance_report_{period}.html")
                        with open(filename, 'w') as f:
                            f.write(html)
                    
                    logger.info(f"Report saved to {filename}")
                
                # Sleep until next check (every hour)
                time.sleep(3600)
                
        # Start the reporting thread
        thread = threading.Thread(target=generate_report, daemon=True)
        thread.start()
        logger.info(f"Scheduled {schedule} {report_type} reports enabled")

















# import cv2
# import face_recognition
# import numpy as np
# import datetime
# import os
# from typing import List, Dict, Tuple, Set
# import logging
# from database import Database

# logger = logging.getLogger(__name__)

# class FacialRecognitionSystem:
#     def __init__(
#                 self, 
#                 database: Database,
#                 known_faces_dir: str = "known_faces",
#                 detection_cooldown: int = 60,
#                 confidence_threshold: float = 0.6
#                 ):
#         """
#         Initialize the facial recognition system.
        
#         Args:
#             database: Database instance for storing detections
#             known_faces_dir: Directory containing known face images
#             detection_cooldown: Cooldown period in seconds between detections of same person
#             confidence_threshold: Minimum confidence level for face recognition
#         """
#         self.database = database
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.known_faces_dir = known_faces_dir
#         self.detection_cooldown = detection_cooldown
#         self.confidence_threshold = confidence_threshold
        
#         # Track last detection time for each person to avoid duplicate entries
#         self.last_detection_times = {}
#         # Current processing date for daily reset
#         self.current_processing_date = datetime.date.today()
#         # Set to track people already detected today (for daily one-time tracking)
#         self.detected_today = set()
        
#         # Load known faces
#         self.load_known_faces()
    
#     def load_known_faces(self) -> None:
#         """Load all known faces from the directory."""
#         if not os.path.exists(self.known_faces_dir):
#             os.makedirs(self.known_faces_dir)
#             logger.info(f"Created directory {self.known_faces_dir} for storing known faces.")
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
#                     logger.info(f"Loaded known face: {name}")
#                 else:
#                     logger.warning(f"No face found in {filename}")
    
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
                
#                 if matches[best_match_index] and confidence > self.confidence_threshold:
#                     name = self.known_face_names[best_match_index]
            
#             # Store in database if known person (with deduplication)
#             if name != "Unknown":
#                 if self.should_store_detection(name, deduplication_mode):
#                     self.database.store_detection(name, confidence, location=location, camera_id=camera_id)
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
#             camera_id: Optional camera identifier (also used as MongoDB collection name)
#             detection_cooldown: Override the default cooldown period in seconds
#             deduplication_mode: Mode for deduplication ("cooldown", "once_per_day", or "always")
#         """
#         # Ensure camera_id is set (needed for collection name)
#         if not camera_id:
#             camera_id = f"Cam-{video_source}" if isinstance(video_source, int) else "default"
#             logger.info(f"No camera_id provided, using: {camera_id}")
        
#         # Override cooldown if provided
#         if detection_cooldown is not None:
#             self.detection_cooldown = detection_cooldown
        
#         # Open video capture
#         video_capture = cv2.VideoCapture(video_source)
        
#         if not video_capture.isOpened():
#             logger.error(f"Could not open video source {video_source}")
#             return
        
#         logger.info(f"Processing video stream from camera: {camera_id}")
#         logger.info(f"Press 'q' to quit, 'a' to add a new face, 'r' to see recent detections.")
#         print(f"Processing video stream from camera: {camera_id}")
#         print(f"Press 'q' to quit, 'a' to add a new face, 'r' to see recent detections.")
#         print(f"Deduplication mode: {deduplication_mode}")
#         if deduplication_mode == "cooldown":
#             print(f"Detection cooldown set to {self.detection_cooldown} seconds.")
        
#         try:
#             while True:
#                 # Capture frame-by-frame
#                 ret, frame = video_capture.read()
                
#                 if not ret:
#                     logger.error("Failed to capture frame")
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
#                     # Show recent detections for current camera
#                     recent = self.database.get_recent_detections(camera_id)
#                     print(f"\nRecent detections for camera {camera_id}:")
#                     for i, detection in enumerate(recent):
#                         print(f"{i+1}. {detection['person_name']} - {detection['timestamp']} - Confidence: {detection['confidence']:.2f}")
#                     print()
#                 elif key == ord('d'):
#                     # Show today's detections for current camera
#                     today = datetime.date.today().strftime("%Y-%m-%d")
#                     detections = self.database.get_daily_detections(today, camera_id)
#                     print(f"\nDetections for today ({today}) on camera {camera_id}:")
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
    
#     def get_office_presence_data(self, person_name=None, limit=10) -> List[Dict]:
#         """
#         Get presence data for all people or a specific person.
        
#         Args:
#             person_name: Optional name to filter results
#             limit: Maximum number of records per person
            
#         Returns:
#             List of presence records
#         """
#         results = []
        
#         # If a specific person is requested
#         if person_name:
#             # Get the person's detection history from all cameras
#             detections = self.database.get_person_history(person_name, limit=limit)
#             return [{
#                 "name": person_name,
#                 "last_seen": detection.get("timestamp"),
#                 "location": detection.get("location"),
#                 "camera": detection.get("camera_id"),
#                 "confidence": detection.get("confidence")
#             } for detection in detections]
        
#         # Get data for all known people
#         for name in self.known_face_names:
#             # Get the most recent detection for each person
#             recent = self.database.get_person_history(name, limit=1)
#             if recent:
#                 detection = recent[0]
#                 results.append({
#                     "name": name,
#                     "last_seen": detection.get("timestamp"),
#                     "location": detection.get("location"),
#                     "camera": detection.get("camera_id"),
#                     "confidence": detection.get("confidence")
#                 })
        
#         return results
    
#     def get_office_summary(self, date_str=None) -> Dict:
#         """
#         Get summary of office presence for a specific date.
        
#         Args:
#             date_str: Date string in format 'YYYY-MM-DD', or None for today
            
#         Returns:
#             Dictionary with presence summary information
#         """
#         # Get all detections for the specified date
#         detections = self.database.get_daily_detections(date_str)
        
#         # Process detections to create summary
#         unique_people = set()
#         camera_counts = {}
#         hourly_counts = {i: 0 for i in range(24)}
        
#         for detection in detections:
#             # Count unique people
#             unique_people.add(detection.get("person_name"))
            
#             # Count detections by camera
#             camera_id = detection.get("camera_id")
#             if camera_id in camera_counts:
#                 camera_counts[camera_id] += 1
#             else:
#                 camera_counts[camera_id] = 1
            
#             # Count detections by hour
#             hour = detection.get("timestamp").hour
#             hourly_counts[hour] += 1
        
#         # Create summary dict
#         summary = {
#             "date": date_str or datetime.date.today().strftime("%Y-%m-%d"),
#             "unique_visitors": len(unique_people),
#             "visitors_list": list(unique_people),
#             "total_detections": len(detections),
#             "camera_statistics": camera_counts,
#             "hourly_distribution": hourly_counts
#         }
        
#         return summary