import datetime
from typing import List, Dict
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, mongo_uri: str, mongo_db: str):
        """Initialize database connection."""
        self.mongo_uri = mongo_uri
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[mongo_db]
        logger.info(f"Connected to MongoDB database: {mongo_db}")
    
    def store_detection(self, name: str, confidence: float, timestamp=None, 
                       location: str = None, camera_id: str = None) -> bool:
        """Store a face detection in the database."""
        try:
            if timestamp is None:
                timestamp = datetime.datetime.now()
                
            date_str = timestamp.strftime("%Y-%m-%d")
            
            # Use camera_id as collection name
            if not camera_id:
                camera_id = "default"
                
            # Get or create collection for this camera
            collection = self.mongo_db[camera_id]
            
            detection_record = {
                "person_name": name,
                "timestamp": timestamp,
                "date": timestamp,
                "date_str": date_str,
                "confidence": float(confidence),
                "location": location,
                "camera_id": camera_id
            }
            collection.insert_one(detection_record)
            
            logger.debug(f"Stored detection: {name} at {timestamp}, camera: {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
            return False
    
    def get_recent_detections(self, camera_id: str = None, limit: int = 10) -> List[Dict]:
        """Get recent face detections from the database."""
        results = []
        
        try:
            if camera_id:
                # Query specific camera collection
                if camera_id in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[camera_id]
                    cursor = collection.find().sort("timestamp", -1).limit(limit)
                    results = list(cursor)
            else:
                # Query all camera collections
                for coll_name in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[coll_name]
                    cursor = collection.find().sort("timestamp", -1).limit(limit)
                    results.extend(list(cursor))
                
                # Sort combined results by timestamp
                results.sort(key=lambda x: x["timestamp"], reverse=True)
                
                # Limit after combining
                if len(results) > limit:
                    results = results[:limit]
        except Exception as e:
            logger.error(f"Error retrieving recent detections: {e}")
            
        return results
    
    def get_daily_detections(self, date_str: str = None, camera_id: str = None) -> List[Dict]:
        """Get all detections for a specific date."""
        if date_str is None:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            
        try:
            # Parse the date string to get year, month, day
            year, month, day = map(int, date_str.split('-'))
            
            # Create datetime objects for the start and end of the day
            start_date = datetime.datetime(year, month, day, 0, 0, 0)
            end_date = datetime.datetime(year, month, day, 23, 59, 59, 999999)
            
            results = []
            
            if camera_id:
                # Query specific camera collection
                if camera_id in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[camera_id]
                    cursor = collection.find({
                        "timestamp": {"$gte": start_date, "$lte": end_date}
                    }).sort("timestamp", 1)
                    results = list(cursor)
            else:
                # Query all camera collections
                for coll_name in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[coll_name]
                    cursor = collection.find({
                        "timestamp": {"$gte": start_date, "$lte": end_date}
                    }).sort("timestamp", 1)
                    results.extend(list(cursor))
                
                # Sort combined results by timestamp
                results.sort(key=lambda x: x["timestamp"])
        except Exception as e:
            logger.error(f"Error retrieving daily detections: {e}")
            results = []
            
        return results
    
    def get_person_history(self, name: str, camera_id: str = None, limit: int = 10) -> List[Dict]:
        """Get detection history for a specific person."""
        results = []
        
        try:
            if camera_id:
                # Query specific camera collection
                if camera_id in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[camera_id]
                    cursor = collection.find({"person_name": name}).sort("timestamp", -1).limit(limit)
                    results = list(cursor)
            else:
                # Query all camera collections
                for coll_name in self.mongo_db.list_collection_names():
                    collection = self.mongo_db[coll_name]
                    cursor = collection.find({"person_name": name}).sort("timestamp", -1).limit(limit)
                    results.extend(list(cursor))
                
                # Sort combined results by timestamp
                results.sort(key=lambda x: x["timestamp"], reverse=True)
                
                # Limit after combining
                if len(results) > limit:
                    results = results[:limit]
        except Exception as e:
            logger.error(f"Error retrieving person history: {e}")
                    
        return results
    
    def close(self):
        """Close database connection."""
        self.mongo_client.close()